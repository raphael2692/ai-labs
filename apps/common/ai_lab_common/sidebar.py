import streamlit as st

from ai_lab_common.settings import Settings


def render_endpoint_sidebar(settings: Settings) -> Settings:
    """Renders the standard 'Server Endpoints' sidebar section and returns a
    Settings instance reflecting any per-session overrides, since Whisper,
    OCR, and Unsloth may run on different machines than the one running this
    app."""

    st.sidebar.header("Server Endpoints")
    whisper_api_url = st.sidebar.text_input("Whisper API URL", value=settings.whisper_api_url)
    ocr_api_url = st.sidebar.text_input("OCR API URL", value=settings.ocr_api_url)
    unsloth_api_base = st.sidebar.text_input("Unsloth API Base URL", value=settings.unsloth_api_base)
    unsloth_model = st.sidebar.text_input("Unsloth Model", value=settings.unsloth_model)
    auth_token = st.sidebar.text_input(
        "Unsloth Auth Token", type="password", value=settings.unsloth_studio_auth_token
    )

    return settings.model_copy(
        update={
            "whisper_api_url": whisper_api_url,
            "ocr_api_url": ocr_api_url,
            "unsloth_api_base": unsloth_api_base,
            "unsloth_model": unsloth_model,
            "unsloth_studio_auth_token": auth_token,
        }
    )
