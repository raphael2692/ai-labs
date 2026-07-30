import streamlit as st
from ai_lab_common import get_llm_client, get_settings
from ai_lab_common.sidebar import render_endpoint_sidebar

st.set_page_config(page_title="App Template", page_icon="🧪", layout="centered")
st.title("🧪 App Template")

settings = render_endpoint_sidebar(get_settings())

st.write("Replace this with your app. `settings` has the configured Whisper/Unsloth endpoints.")

if st.button("Test Unsloth connection"):
    try:
        client = get_llm_client(settings)
        response = client.chat.completions.create(
            model=settings.unsloth_model,
            messages=[{"role": "user", "content": "Say hello in five words or fewer."}],
        )
        st.success(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Could not reach Unsloth server at {settings.unsloth_api_base}: {e}")
