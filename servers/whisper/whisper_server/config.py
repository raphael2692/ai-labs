import os

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


HOST = _env("WHISPER_HOST", "0.0.0.0")
PORT = int(_env("WHISPER_PORT", "9000"))
MODEL_SIZE = _env("WHISPER_MODEL_SIZE", "medium")
DEVICE = _env("WHISPER_DEVICE", "cuda")
COMPUTE_TYPE = _env("WHISPER_COMPUTE_TYPE", "float16")
