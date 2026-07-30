import os

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


HOST = _env("OCR_HOST", "0.0.0.0")
PORT = int(_env("OCR_PORT", "9100"))
MODEL = _env("OCR_MODEL", "baidu/Unlimited-OCR")
DEVICE = _env("OCR_DEVICE", "cuda")
DTYPE = _env("OCR_DTYPE", "bfloat16")

BASE_SIZE = int(_env("OCR_BASE_SIZE", "1024"))
IMAGE_SIZE = int(_env("OCR_IMAGE_SIZE", "640"))
CROP_MODE = _env_bool("OCR_CROP_MODE", True)
MAX_LENGTH = int(_env("OCR_MAX_LENGTH", "32768"))

PDF_DPI = int(_env("OCR_PDF_DPI", "300"))
