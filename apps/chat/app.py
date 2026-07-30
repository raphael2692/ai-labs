import streamlit as st
from ai_lab_common import get_llm_client, get_settings
from ai_lab_common.sidebar import render_endpoint_sidebar

st.set_page_config(page_title="Local LLM Chat", page_icon="💬", layout="centered")
st.title("💬 Local LLM Chat")

settings = render_endpoint_sidebar(get_settings())

if st.sidebar.button("Clear conversation"):
    st.session_state.pop("messages", None)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful AI assistant."}]

for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if user_input := st.chat_input("Type a message..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            client = get_llm_client(settings)
            stream = client.chat.completions.create(
                model=settings.unsloth_model,
                messages=st.session_state.messages,
                stream=True,
            )

            reply = ""
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    reply += chunk.choices[0].delta.content
                    placeholder.markdown(reply)

            st.session_state.messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            st.error(f"Could not reach Unsloth server at {settings.unsloth_api_base}: {e}")
