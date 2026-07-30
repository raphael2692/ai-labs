from openai import OpenAI

from ai_lab_common.settings import Settings


def get_llm_client(settings: Settings) -> OpenAI:
    """Returns an OpenAI-compatible client pointed at the Unsloth server."""
    return OpenAI(base_url=settings.unsloth_api_base, api_key=settings.unsloth_studio_auth_token)
