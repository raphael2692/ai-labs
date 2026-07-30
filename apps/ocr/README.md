# OCR

Upload a PDF (or image) and get it parsed to markdown by the OCR server, page by page with live progress.

## Run

From the repo root:

```
uv run --package ocr-app streamlit run apps/ocr/app.py
```

or use the helper script: `./scripts/run_app.sh ocr` / `./scripts/run_app.ps1 ocr`.

Configure the OCR endpoint via the sidebar, or set it once in the repo-root `.env` (`OCR_API_URL`).
