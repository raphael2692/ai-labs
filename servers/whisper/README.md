# Whisper Server

GPU-backed speech-to-text HTTP API, built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper) + FastAPI.
Streams transcription progress as newline-delimited JSON so callers can show live progress.

This folder is self-contained: it can be copied alone to any GPU machine (it does not need the rest of
the `ai-lab` repo) and run independently. That machine only needs `uv` and, on Windows, a CUDA-capable GPU + driver.

## Run

```
./start.sh      # Linux/macOS
./start.ps1     # Windows (PowerShell)
```

Both scripts copy `.env.example` to `.env` on first run, then `uv sync` and launch the server.

## Configuration

Edit `.env` (see `.env.example`):

| Variable               | Default   | Notes                                      |
|-------------------------|-----------|---------------------------------------------|
| `WHISPER_HOST`          | `0.0.0.0` | bind address                                |
| `WHISPER_PORT`          | `9000`    | bind port                                   |
| `WHISPER_MODEL_SIZE`    | `medium`  | any faster-whisper model size/name          |
| `WHISPER_DEVICE`        | `cuda`    | `cuda` or `cpu`                             |
| `WHISPER_COMPUTE_TYPE`  | `float16` | e.g. `float16` (GPU), `int8` (CPU)          |

## API

- `GET /health` — readiness probe.
- `POST /v1/audio/transcriptions` — multipart form upload (`file`), returns `application/x-ndjson` with
  `{"type": "info" | "segment" | "done" | "error", ...}` lines.

Point any app (e.g. the `apps/meeting_minutes` Streamlit app) at
`http://<this-machine-ip>:<WHISPER_PORT>/v1/audio/transcriptions`.
