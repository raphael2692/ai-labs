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
| `OCR_BASE_SIZE`   | `1024`               | per-page "gundam" preset base size                 |
| `OCR_IMAGE_SIZE`  | `640`                | per-page "gundam" preset image size                |
| `OCR_CROP_MODE`   | `true`               | per-page "gundam" preset crop mode                 |
| `OCR_MAX_LENGTH`  | `32768`              | max generated tokens per page                      |
| `OCR_PDF_DPI`     | `300`                | DPI used to rasterize PDF pages before OCR         |

## API

- `GET /health` — readiness probe.
- `POST /v1/ocr/parse` — multipart form upload (`file`, a PDF or image), returns `application/x-ndjson`
  with `{"type": "info" | "page" | "done" | "error", ...}` lines. PDFs are rasterized page-by-page (via
  PyMuPDF) and each page is OCR'd independently, so pages stream in as they finish; the final `done` event
  carries the concatenated full document as markdown (the model is prompted with
  `<|grounding|>Convert the document to markdown.`, and `<|det|>`/bbox grounding markers are stripped from
  the output before it's returned).

Point any app (e.g. the `apps/ocr` Streamlit app) at `http://<this-machine-ip>:<OCR_PORT>/v1/ocr/parse`.
