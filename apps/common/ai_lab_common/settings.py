from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Endpoints for the servers this app talks to.

    These servers typically run on other machines on the LAN, so every value
    here is meant to be overridden via `.env` (or the Streamlit sidebar) per
    deployment rather than hardcoded.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    whisper_api_url: str = "http://localhost:9000/v1/audio/transcriptions"
    ocr_api_url: str = "http://localhost:9100/v1/ocr/parse"
    unsloth_api_base: str = "http://localhost:8888/v1"
    unsloth_studio_auth_token: str = ""
    unsloth_model: str = "default"


@lru_cache
def get_settings() -> Settings:
    return Settings()
