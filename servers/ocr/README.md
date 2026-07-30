# OCR Server

GPU-backed document OCR HTTP API, built on [baidu/Unlimited-OCR](https://github.com/baidu/Unlimited-OCR)
(loaded directly via `transformers`) + FastAPI. Streams per-page parsing progress as newline-delimited
JSON so callers can show live progress while a multi-page PDF is processed.

This folder is self-contained: it can be copied alone to any GPU machine (it does not need the rest of
the `ai-lab` repo) and run independently. That machine only needs `uv` and a CUDA-capable GPU + driver
(the model is loaded in `bfloat16` on `cuda` by default).

## Run

```
./start.sh      # Linux/macOS
./start.ps1     # Windows (PowerShell)
```

Both scripts copy `.env.example` to `.env` on first run, then `uv sync` and launch the server. The first
run downloads the `baidu/Unlimited-OCR` weights from the Hugging Face Hub.

## Configuration

Edit `.env` (see `.env.example`):

| Variable          | Default              | Notes                                             |
|-------------------|----------------------|----------------------------------------------------|
| `OCR_HOST`        | `0.0.0.0`            | bind address                                       |
| `OCR_PORT`        | `9100`               | bind port                                          |
| `OCR_MODEL`       | `baidu/Unlimited-OCR`| HF repo id or local path                           |
| `OCR_DEVICE`      | `cuda`               | `cuda` or `cpu`                                    |
| `OCR_DTYPE`       | `bfloat16`           | torch dtype used to load the model                 |
| `OCR_BASE_SIZE`   | `1024`               | base/global-view size, used by both image_modes    |
| `OCR_IMAGE_SIZE`  | `640`                | detail-tile size, used by "gundam" image_mode only |
| `OCR_MAX_LENGTH`  | `32768`              | max generated tokens per page                      |
| `OCR_PDF_DPI`     | `300`                | DPI used to rasterize PDF pages before OCR         |

## API

- `GET /health` — readiness probe.
- `POST /v1/ocr/parse` — multipart form upload: `file` (a PDF or image) and optional `image_mode`
  (`"gundam"` default, or `"base"`). Returns `application/x-ndjson` with
  `{"type": "info" | "page_start" | "chunk" | "page" | "done" | "error", ...}` lines:
  - `info` — `{"pages": N}`, once at the start.
  - `page_start` — `{"index", "total", "image"}`, emitted right after a page is rasterized, before
    inference starts. `image` is a base64 data URI of exactly the image sent to the model, so callers can
    display the source page immediately and later overlay `<|det|>` boxes at pixel-accurate coordinates.
  - `chunk` — `{"index", "total", "text"}`, the model's raw output as it's generated (unprocessed,
    `<|det|>` markers included), for live token-streaming display.
  - `page` / `done` — as before: `text` (grounding markers stripped) and `raw_text` (unprocessed, markers
    included); `done`'s versions are the full-document concatenation of all pages.

  PDFs are rasterized page-by-page (via PyMuPDF) and each page is OCR'd independently, so pages stream in
  as they finish. The model is prompted with the model card's documented `document parsing.` prompt, which
  emits `<|det|>`-tagged grounding output by default.

  `image_mode` controls the per-page inference preset: `"gundam"` crops each page into detail tiles
  (`OCR_IMAGE_SIZE`) plus a global view (`OCR_BASE_SIZE`) — best for a single, detail-heavy page/image.
  `"base"` runs one uncropped pass at `OCR_BASE_SIZE` — faster, better suited to multi-page/PDF throughput.

Point any app (e.g. the `apps/ocr` Streamlit app) at `http://<this-machine-ip>:<OCR_PORT>/v1/ocr/parse`.
