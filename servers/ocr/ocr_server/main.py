import base64
import json
import logging
import os
import re
import shutil
import signal
import tempfile
import time
from pathlib import Path

import fitz  # PyMuPDF
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from transformers import AutoModel, AutoTokenizer

from ocr_server import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ocr-server")

_PDF_CONTENT_TYPES = {"application/pdf"}
_RESULT_EXTENSIONS = (".mmd", ".md", ".txt")

# Coordinates in `[x0, y0, x1, y1]` are on a 0-1000 scale relative to the page
# image (not the model's internal resize canvas) - matching the DeepSeek-OCR
# family's grounding convention this model's <|det|> markers follow.
_DET_RE = re.compile(r"<\|det\|>([^<\s\[]+)(?:\s*\[([^\]]*)\])?\s*<\|/det\|>(.*)", re.DOTALL)


def _handle_sigint(signum, frame):
    logger.info("SIGINT received, forcing exit")
    os._exit(0)


def _parse_blocks(raw: str) -> list[dict]:
    """Parse <|det|>type [x0, y0, x1, y1]<|/det|> markers from the model's raw
    output into structured blocks, grouping lines belonging to the same block.
    Mirrors the post-processing recipe from the model card, but keeps the
    category/bbox instead of discarding them. bbox coordinates are on a
    0-1000 scale relative to the page image the model was given."""
    blocks: list[dict] = []
    cur: dict | None = None
    for line in raw.splitlines():
        line = line.rstrip()
        if not line:
            continue
        m = _DET_RE.match(line)
        if m:
            category, bbox_raw, content = m.group(1).strip(), m.group(2), m.group(3).strip()
            bbox = None
            if bbox_raw:
                try:
                    bbox = [int(v.strip()) for v in bbox_raw.split(",")]
                except ValueError:
                    bbox = None
            if cur is not None:
                blocks.append(cur)
            cur = {"category": category, "bbox": bbox, "lines": [content] if content else []}
            continue
        if cur is None:
            cur = {"category": None, "bbox": None, "lines": []}
        cur["lines"].append(line)
    if cur is not None:
        blocks.append(cur)

    result = []
    for b in blocks:
        result.append({"category": b["category"], "bbox": b["bbox"], "text": "\n".join(b["lines"]).strip()})
    return result


def _blocks_to_text(blocks: list[dict]) -> str:
    return "\n\n".join(b["text"] for b in blocks if b["category"] != "image" and b["text"]).strip()


def _is_pdf(filename: str, content_type: str | None) -> bool:
    if content_type in _PDF_CONTENT_TYPES:
        return True
    return Path(filename).suffix.lower() == ".pdf"


def _pdf_to_images(pdf_path: str, out_dir: str, dpi: int) -> list[str]:
    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    paths = []
    for i, page in enumerate(doc):
        out_path = os.path.join(out_dir, f"page_{i + 1:04d}.png")
        page.get_pixmap(matrix=mat).save(out_path)
        paths.append(out_path)
    doc.close()
    return paths


def _extract_blocks(infer_return, output_dir: str) -> list[dict]:
    """The model's .infer()/.infer_multi() write parsed output to output_dir
    when save_results=True; some trust_remote_code revisions also return the
    text directly. Prefer the direct return value, otherwise read the most
    recently written result file in output_dir."""
    if isinstance(infer_return, str) and infer_return.strip():
        logger.info(f"infer() returned a str directly ({len(infer_return)} chars); raw head: {infer_return[:300]!r}")
        return _parse_blocks(infer_return)
    if isinstance(infer_return, dict):
        for key in ("text", "result", "content"):
            value = infer_return.get(key)
            if isinstance(value, str) and value.strip():
                logger.info(f"infer() returned dict key '{key}' ({len(value)} chars); raw head: {value[:300]!r}")
                return _parse_blocks(value)
        logger.info(f"infer() returned a dict with no usable text key; keys={list(infer_return.keys())}")

    all_files = sorted(Path(output_dir).glob("*"))
    logger.info(
        f"infer() return value unused (type={type(infer_return)!r}); "
        f"output_dir contents: {[(p.name, p.stat().st_size) for p in all_files]}"
    )
    candidates = sorted(
        (p for p in all_files if p.suffix.lower() in _RESULT_EXTENSIONS),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError(f"OCR model produced no readable output in {output_dir}")
    chosen = candidates[0]
    raw = chosen.read_text(encoding="utf-8")
    logger.info(f"Reading result file '{chosen.name}' ({len(raw)} chars); raw head: {raw[:300]!r}")
    return _parse_blocks(raw)


def create_app() -> FastAPI:
    app = FastAPI(title="Local OCR API")

    logger.info(f"Loading OCR model '{config.MODEL}' on {config.DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        config.MODEL,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=getattr(torch, config.DTYPE),
    )
    model = model.eval().to(config.DEVICE)

    # The vendored `.infer()` calls self.generate(..., eos_token_id=tokenizer.eos_token_id, ...)
    # without ever passing pad_token_id or attention_mask, so generate() falls back to treating
    # eos as pad — even though the tokenizer defines a real, distinct pad token. Sync it onto the
    # model's generation config so the attention mask is built correctly instead of guessed.
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    logger.info("OCR model loaded successfully!")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": config.MODEL, "device": config.DEVICE}

    @app.post("/v1/ocr/parse")
    def parse_document(file: UploadFile = File(...)):
        suffix = Path(file.filename).suffix or ".bin"
        work_dir = tempfile.mkdtemp(prefix="ocr_")
        upload_path = os.path.join(work_dir, f"upload{suffix}")
        with open(upload_path, "wb") as f:
            f.write(file.file.read())

        is_pdf = _is_pdf(file.filename, file.content_type)
        logger.info(f"Received {file.filename} (pdf={is_pdf}) -> {upload_path}")

        def generate():
            start = time.time()
            try:
                if is_pdf:
                    pages = _pdf_to_images(upload_path, work_dir, config.PDF_DPI)
                else:
                    pages = [upload_path]

                total = len(pages)
                yield json.dumps({"type": "info", "pages": total}) + "\n"

                page_texts = []
                for i, page_path in enumerate(pages, start=1):
                    page_output_dir = os.path.join(work_dir, f"out_{i:04d}")
                    os.makedirs(page_output_dir, exist_ok=True)

                    infer_return = model.infer(
                        tokenizer,
                        prompt="<image>document parsing.",
                        image_file=page_path,
                        output_path=page_output_dir,
                        base_size=config.BASE_SIZE,
                        image_size=config.IMAGE_SIZE,
                        crop_mode=config.CROP_MODE,
                        max_length=config.MAX_LENGTH,
                        no_repeat_ngram_size=35,
                        ngram_window=128,
                        save_results=True,
                    )
                    blocks = _extract_blocks(infer_return, page_output_dir)
                    page_text = _blocks_to_text(blocks)
                    page_texts.append(page_text)
                    image_b64 = base64.b64encode(Path(page_path).read_bytes()).decode("ascii")

                    logger.info(f"Page {i}/{total} parsed ({len(page_text)} chars, {len(blocks)} block(s))")
                    if blocks and not page_text:
                        logger.warning(f"All {len(blocks)} block(s) had empty/filtered text: {blocks}")
                    yield json.dumps({
                        "type": "page", "index": i, "total": total, "text": page_text,
                        "blocks": blocks, "image_b64": image_b64,
                    }) + "\n"

                full_text = "\n\n".join(page_texts).strip()
                elapsed = time.time() - start
                logger.info(f"Done in {elapsed:.1f}s, {len(full_text)} chars, {total} page(s)")
                yield json.dumps({
                    "type": "done", "text": full_text, "pages": total, "elapsed": elapsed,
                }) + "\n"
            except Exception as e:
                logger.exception("OCR parsing failed")
                yield json.dumps({"type": "error", "message": str(e)}) + "\n"
            finally:
                shutil.rmtree(work_dir, ignore_errors=True)

        return StreamingResponse(generate(), media_type="application/x-ndjson")

    return app


def run():
    signal.signal(signal.SIGINT, _handle_sigint)
    app = create_app()
    uvicorn.run(app, host=config.HOST, port=config.PORT)


if __name__ == "__main__":
    run()
