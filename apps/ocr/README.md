# OCR

Upload a PDF (or image) and watch it get parsed page by page: the source page and its live-streamed text
show side by side, with the model's `<|det|>` layout boxes drawn on the image as they're detected. Once a
page finishes, review any page with the slider, and download the full text or a ZIP (box-overlay image +
raw output + clean text per page).

## Run

From the repo root:

```
uv run --package ocr-app streamlit run apps/ocr/app.py
```

or use the helper script: `./scripts/run_app.sh ocr` / `./scripts/run_app.ps1 ocr`.

Configure the OCR endpoint via the sidebar, or set it once in the repo-root `.env` (`OCR_API_URL`). The
sidebar's `image_mode` selector picks the OCR server's per-page inference preset: `gundam` (cropped detail
tiles, best for a single detail-heavy page) or `base` (one uncropped pass, faster for multi-page/PDF) — see
`servers/ocr/README.md`.
