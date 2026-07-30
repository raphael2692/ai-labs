import streamlit as st
from ai_lab_common import get_settings, ocr_stream
from ai_lab_common.ocr_client import OcrError
from ai_lab_common.sidebar import render_endpoint_sidebar

st.set_page_config(page_title="OCR", page_icon="📄", layout="centered")

st.title("📄 OCR")
st.markdown("Upload a **PDF** (or image) to parse its text via your OCR server, page by page.")

settings = render_endpoint_sidebar(get_settings())

uploaded_file = st.file_uploader("Choose a document", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        st.markdown(f"**{uploaded_file.name}** ({uploaded_file.size / 1024:.0f} KB)")
    else:
        st.image(uploaded_file)

    if st.button("🚀 Run OCR", type="primary"):
        status_placeholder = st.empty()

        st.markdown("### 📋 Extracted Text")
        tab_markdown, tab_raw = st.tabs(["Markdown", "Raw Output"])
        markdown_placeholder = tab_markdown.empty()
        raw_placeholder = tab_raw.empty()

        full_text = ""
        full_raw = ""
        text_chunks = []
        raw_chunks = []

        try:
            status_placeholder.info("Sending file to OCR server...")
            for event in ocr_stream(
                settings.ocr_api_url, uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type
            ):
                if event["type"] == "info":
                    status_placeholder.info(f"Parsing {event['pages']} page(s)...")
                elif event["type"] == "page":
                    status_placeholder.info(f"Parsed page {event['index']}/{event['total']}...")
                    text_chunks.append(event["text"])
                    raw_chunks.append(event["raw_text"].strip())
                    markdown_placeholder.markdown("\n\n".join(text_chunks))
                    raw_placeholder.text("\n\n".join(raw_chunks))
                elif event["type"] == "done":
                    full_text = event["text"]
                    full_raw = event["raw_text"]
                    markdown_placeholder.markdown(full_text)
                    raw_placeholder.text(full_raw)
                    status_placeholder.success(
                        f"OCR complete! ({event['pages']} page(s) in {event['elapsed']:.1f}s)"
                    )
        except OcrError as e:
            st.error(f"OCR failed: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Could not reach OCR server at {settings.ocr_api_url}: {e}")
            st.stop()

        st.download_button(
            label="Download Markdown (.md)",
            data=full_text,
            file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}.md",
            mime="text/markdown",
        )
