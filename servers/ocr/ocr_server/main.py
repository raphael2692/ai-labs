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

_DET_RE = re.compile(r"<\|det\|>([^<\s]+)(?:\s*\[[^\]]*\])?\s*<\|/det\|>(.*)", re.DOTALL)


def _handle_sigint(signum, frame):
    logger.info("SIGINT received, forcing exit")
    os._exit(0)


def _remove_det(raw: str) -> str:
    """Strip <|det|>type [bbox]<|/det|> markers from the model's raw output,
    grouping lines belonging to the same block and separating blocks with a
    blank line. Mirrors the post-processing recipe from the model card."""
    blocks: list[list[str]] = []
    cur: list[str] | None = None
    for line in raw.splitlines():
        line = line.rstrip()
        if not line:
            continue
        m = _DET_RE.match(line)
        if m:
            category, content = m.group(1).strip(), m.group(2).strip()
            if category == "image":
                continue
            if cur is not None:
                blocks.append(cur)
            cur = [content] if content else []
            continue
        if cur is None:
            cur = []
        cur.append(line)
    if cur is not None:
        blocks.append(cur)
    return "\n\n".join("\n".join(b) for b in blocks).strip()


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


def _extract_result(infer_return, output_dir: str) -> str:
    """The model's .infer()/.infer_multi() write parsed output to output_dir
    when save_results=True; some trust_remote_code revisions also return the
    text directly. Prefer the direct return value, otherwise read the most
    recently written result file in output_dir."""
    if isinstance(infer_return, str) and infer_return.strip():
        return _remove_det(infer_return)
    if isinstance(infer_return, dict):
        for key in ("text", "result", "content"):
            value = infer_return.get(key)
            if isinstance(value, str) and value.strip():
                return _remove_det(value)

    candidates = sorted(
        (p for p in Path(output_dir).glob("*") if p.suffix.lower() in _RESULT_EXTENSIONS),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError(f"OCR model produced no readable output in {output_dir}")
    return _remove_det(candidates[0].read_text(encoding="utf-8"))


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
                        prompt="<image>\n<|grounding|>Convert the document to markdown.",
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
                    page_text = _extract_result(infer_return, page_output_dir)
                    page_texts.append(page_text)

                    logger.info(f"Page {i}/{total} parsed ({len(page_text)} chars)")
                    yield json.dumps({
                        "type": "page", "index": i, "total": total, "text": page_text,
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
