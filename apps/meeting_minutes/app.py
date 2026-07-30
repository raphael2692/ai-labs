from pathlib import Path

import streamlit as st
from ai_lab_common import get_llm_client, get_settings, transcribe_stream
from ai_lab_common.sidebar import render_endpoint_sidebar
from ai_lab_common.whisper_client import WhisperTranscriptionError

st.set_page_config(page_title="Local Meeting Minutes Generator", page_icon="🎙️", layout="centered")

st.title("🎙️ Local Meeting Minutes & Transcription Suite")
st.markdown(
    "Upload an audio/video recording (**MP3, MKV, MP4**) to transcribe it via your Whisper server and"
    " generate meeting minutes via your Unsloth server."
)

settings = render_endpoint_sidebar(get_settings())

uploaded_file = st.file_uploader("Choose a meeting file", type=["mp3", "mkv", "mp4"])

if uploaded_file is not None:
    file_extension = Path(uploaded_file.name).suffix.lower()
    if file_extension in [".mp4", ".mkv"]:
        st.video(uploaded_file)
    else:
        st.audio(uploaded_file)

    if st.button("🚀 Process Recording", type="primary"):
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

        with st.expander("View Raw Transcript"):
            st.write(transcript_text)

        with st.spinner("Analyzing transcript and generating meeting minutes..."):
            try:
                client = get_llm_client(settings)

                prompt = (
                    "You are an executive assistant AI. Analyze the following meeting transcript and"
                    " generate professional meeting minutes containing:\n"
                    "1. **Executive Summary** (Overview of the meeting)\n"
                    "2. **Key Decisions Made** (Bullet points)\n"
                    "3. **Action Items** (Tasks with responsible owners if mentioned)\n\n"
                    f"Transcript:\n{transcript_text}"
                )

                response = client.chat.completions.create(
                    model=settings.unsloth_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You extract structured, concise business intelligence outputs from transcripts.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                )

                meeting_minutes = response.choices[0].message.content
            except Exception as e:
                st.error(f"Could not reach Unsloth server at {settings.unsloth_api_base}: {e}")
                st.stop()

        st.markdown("### 📋 Meeting Minutes")
        st.markdown(meeting_minutes)

        st.download_button(
            label="Download Meeting Minutes (.md)",
            data=meeting_minutes,
            file_name="meeting_minutes.md",
            mime="text/markdown",
        )
