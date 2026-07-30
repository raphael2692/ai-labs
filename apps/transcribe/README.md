# Transcribe

Upload a recording, get it transcribed by the Whisper server.

## Run

From the repo root:

```
uv run --package transcribe streamlit run apps/transcribe/app.py
```

or use the helper script: `./scripts/run_app.sh transcribe` / `./scripts/run_app.ps1 transcribe`.

Configure the Whisper endpoint via the sidebar, or set it once in the repo-root `.env`
(`WHISPER_API_URL`).
