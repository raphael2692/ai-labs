from pathlib import Path

import streamlit as st
from ai_lab_common import get_settings, transcribe_stream
from ai_lab_common.sidebar import render_endpoint_sidebar
from ai_lab_common.whisper_client import WhisperTranscriptionError

st.set_page_config(page_title="Transcribe", page_icon="🎙️", layout="centered")

st.title("🎙️ Transcribe")
st.markdown("Upload an audio/video recording (**MP3, MKV, MP4**) to transcribe it via your Whisper server.")

settings = render_endpoint_sidebar(get_settings())

uploaded_file = st.file_uploader("Choose a file", type=["mp3", "mkv", "mp4"])

if uploaded_file is not None:
    file_extension = Path(uploaded_file.name).suffix.lower()
    if file_extension in [".mp4", ".mkv"]:
        st.video(uploaded_file)
    else:
        st.audio(uploaded_file)

    if st.button("🚀 Transcribe", type="primary"):
        with st.spinner("Sending file to Whisper server..."):
            status_placeholder = st.empty()
            transcript_placeholder = st.empty()
            transcript_text = ""
            lang = "unknown"
            duration = 0.0

            try:
                for event in transcribe_stream(
                    settings.whisper_api_url, uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type
                ):
                    if event["type"] == "info":
                        lang = event["language"]
                        duration = event["duration"]
                        status_placeholder.info(
                            f"Detected language: {lang} · Duration: {duration:.1f}s · Transcribing..."
                        )
                    elif event["type"] == "segment":
                        transcript_text += event["text"] + " "
                        status_placeholder.info(f"Transcribing... {event['end']:.1f}s / {duration:.1f}s")
                        transcript_placeholder.text_area("Live transcript", transcript_text, height=200)
                    elif event["type"] == "done":
                        transcript_text = event["text"]
                        status_placeholder.success(
                            f"Transcription complete! (Language: {lang}, Duration: {duration:.1f}s)"
                        )
            except WhisperTranscriptionError as e:
                st.error(f"Transcription failed: {e}")
                st.stop()
            except Exception as e:
                st.error(f"Could not reach Whisper server at {settings.whisper_api_url}: {e}")
                st.stop()

        st.markdown("### 📄 Transcript")
        st.text_area("Full transcript", transcript_text, height=300)

        st.download_button(
            label="Download Transcript (.txt)",
            data=transcript_text,
            file_name="transcript.txt",
            mime="text/plain",
        )
