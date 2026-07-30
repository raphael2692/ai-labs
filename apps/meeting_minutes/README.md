# Meeting Minutes

Upload a recording, get it transcribed by the Whisper server, then summarized into structured meeting
minutes by the Unsloth server.

## Run

From the repo root:

```
uv run --package meeting-minutes streamlit run apps/meeting_minutes/app.py
```

or use the helper script: `./scripts/run_app.sh meeting_minutes` / `./scripts/run_app.ps1 meeting_minutes`.

Configure the Whisper/Unsloth endpoints via the sidebar, or set them once in the repo-root `.env`
(`WHISPER_API_URL`, `UNSLOTH_API_BASE`, `UNSLOTH_STUDIO_AUTH_TOKEN`, `UNSLOTH_MODEL`).
