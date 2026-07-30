# ai-lab-common

Shared building blocks for the Streamlit apps in `apps/`:

- `ai_lab_common.settings` — `Settings` (pydantic-settings), reads server endpoints/tokens from the
  environment / a `.env` file, since the Whisper and Unsloth servers usually run on other machines.
- `ai_lab_common.sidebar` — renders the standard "Server Endpoints" sidebar section, letting the endpoints
  be overridden per-session without editing `.env`.
- `ai_lab_common.whisper_client` — `transcribe_stream(...)`, wraps the Whisper server's streaming NDJSON
  API.
- `ai_lab_common.llm_client` — `get_llm_client(...)`, returns an `openai.OpenAI` client pointed at the
  Unsloth server.

Every app depends on this package via the uv workspace (`ai-lab-common = { workspace = true }`), so changes
here are picked up by all apps immediately without publishing anything.
