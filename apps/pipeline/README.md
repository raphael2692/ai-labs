# Pipeline

Upload an audio file, a PDF, or an image; watch it get extracted (via the Whisper server for audio, the OCR
server for PDF/image) with live streaming output; then run the extracted text through a stored, user-editable
prompt to produce a final Markdown document — also streamed live. Every stage can be stopped mid-flight, and
partial output from either stage is preserved instead of discarded.

## Run

From the repo root:

```
uv run --package pipeline-app streamlit run apps/pipeline/app.py
```

or use the helper script: `./scripts/run_app.sh pipeline` / `./scripts/run_app.ps1 pipeline`.

Configure the Whisper/OCR/Unsloth endpoints via the sidebar, or set them once in the repo-root `.env`. Saved
prompts live in `apps/pipeline/prompts.json` (gitignored — created on first save; ships with one default
preset).
