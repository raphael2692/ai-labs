import importlib.util
import json
import logging
import os
import signal
import sys
import tempfile
import time

import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from faster_whisper import WhisperModel

from whisper_server import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("whisper-server")


def _handle_sigint(signum, frame):
    logger.info("SIGINT received, forcing exit")
    os._exit(0)


def _fix_windows_cuda_dll_path() -> None:
    """faster-whisper's CTranslate2 backend needs cuBLAS/cuDNN DLLs on PATH.
    On Windows these ship inside the nvidia-* pip packages rather than system-wide,
    so we have to locate and register their `bin` folders manually."""
    if sys.platform != "win32":
        return
    for pkg_name in ("nvidia.cublas", "nvidia.cudnn"):
        spec = importlib.util.find_spec(pkg_name)
        if not spec or not spec.submodule_search_locations:
            logger.warning(f"Could not locate package: {pkg_name}")
            continue
        for loc in spec.submodule_search_locations:
            bin_dir = os.path.join(loc, "bin")
            if os.path.isdir(bin_dir):
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ["PATH"]
                os.add_dll_directory(bin_dir)
                logger.info(f"Added to PATH: {bin_dir}")
                break
        else:
            logger.warning(f"Found {pkg_name} but no 'bin' folder in {spec.submodule_search_locations}")


def create_app() -> FastAPI:
    app = FastAPI(title="Local Whisper ASR API")

    logger.info(f"Loading Whisper model '{config.MODEL_SIZE}' on {config.DEVICE}...")
    model = WhisperModel(config.MODEL_SIZE, device=config.DEVICE, compute_type=config.COMPUTE_TYPE)
    logger.info("Whisper model loaded successfully!")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": config.MODEL_SIZE, "device": config.DEVICE}

    @app.post("/v1/audio/transcriptions")
    def transcribe_audio(file: UploadFile = File(...)):
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file.file.read())
            tmp_path = tmp.name

        logger.info(f"Received {file.filename} -> {tmp_path}, starting transcription")

        def generate():
            start = time.time()
            try:
                segments, info = model.transcribe(tmp_path, beam_size=5)
                logger.info(f"Detected language={info.language} duration={info.duration:.1f}s")
                yield json.dumps({
                    "type": "info", "language": info.language, "duration": info.duration,
                }) + "\n"

                full_text = ""
                for segment in segments:
                    full_text += segment.text + " "
                    logger.info(f"[{segment.start:.1f}s -> {segment.end:.1f}s] {segment.text}")
                    yield json.dumps({
                        "type": "segment", "start": segment.start,
                        "end": segment.end, "text": segment.text,
                    }) + "\n"

                elapsed = time.time() - start
                logger.info(f"Done in {elapsed:.1f}s, {len(full_text)} chars")
                yield json.dumps({
                    "type": "done", "text": full_text.strip(),
                    "language": info.language, "duration": info.duration, "elapsed": elapsed,
                }) + "\n"
            except Exception as e:
                logger.exception("Transcription failed")
                yield json.dumps({"type": "error", "message": str(e)}) + "\n"
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        return StreamingResponse(generate(), media_type="application/x-ndjson")

    return app


def run():
    signal.signal(signal.SIGINT, _handle_sigint)
    _fix_windows_cuda_dll_path()
    app = create_app()
    uvicorn.run(app, host=config.HOST, port=config.PORT)


if __name__ == "__main__":
    run()
