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
        pages_placeholder = st.container()
        full_text = ""
        total_pages = 0

        try:
            status_placeholder.info("Sending file to OCR server...")
            for event in ocr_stream(
                settings.ocr_api_url, uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type
            ):
                if event["type"] == "info":
                    total_pages = event["pages"]
                    status_placeholder.info(f"Parsing {total_pages} page(s)...")
                elif event["type"] == "page":
                    status_placeholder.info(f"Parsed page {event['index']}/{event['total']}...")
                    with pages_placeholder.expander(f"Page {event['index']}/{event['total']}", expanded=False):
                        st.text(event["text"])
                elif event["type"] == "done":
                    full_text = event["text"]
                    status_placeholder.success(
                        f"OCR complete! ({event['pages']} page(s) in {event['elapsed']:.1f}s)"
                    )
        except OcrError as e:
            st.error(f"OCR failed: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Could not reach OCR server at {settings.ocr_api_url}: {e}")
            st.stop()

        st.markdown("### 📋 Extracted Text")
        st.text_area("Full text", full_text, height=400)

        st.download_button(
            label="Download Text (.txt)",
            data=full_text,
            file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}.txt",
            mime="text/plain",
        )
