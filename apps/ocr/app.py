import base64
import io

import streamlit as st
from ai_lab_common import get_settings, ocr_stream
from ai_lab_common.ocr_client import OcrError
from ai_lab_common.sidebar import render_endpoint_sidebar
from PIL import Image, ImageDraw

st.set_page_config(page_title="OCR", page_icon="📄", layout="centered")

st.title("📄 OCR")
st.markdown("Upload a **PDF** (or image) to parse its text via your OCR server, page by page.")

settings = render_endpoint_sidebar(get_settings())

_CATEGORY_COLORS = {
    "title": "#3B82F6",
    "text": "#22C55E",
    "image": "#F97316",
}
_DEFAULT_COLOR = "#A855F7"


def _draw_boxes(image_b64: str, blocks: list[dict]) -> Image.Image:
    """Draws each block's bbox (0-1000 scale, relative to the page image) on
    a copy of that page's image."""
    image = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for block in blocks:
        bbox = block.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x0, y0, x1, y1 = bbox
        color = _CATEGORY_COLORS.get(block.get("category"), _DEFAULT_COLOR)
        draw.rectangle(
            [x0 / 1000 * width, y0 / 1000 * height, x1 / 1000 * width, y1 / 1000 * height],
            outline=color,
            width=max(2, width // 400),
        )
    return image


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
        pages_data = []

        try:
            status_placeholder.info("Sending file to OCR server...")
            for event in ocr_stream(
                settings.ocr_api_url, uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type
            ):
                if event["type"] == "info":
                    status_placeholder.info(f"Parsing {event['pages']} page(s)...")
                elif event["type"] == "page":
                    status_placeholder.info(f"Parsed page {event['index']}/{event['total']}...")
                    pages_data.append(event)
                    with pages_placeholder.expander(f"Page {event['index']}/{event['total']}", expanded=False):
                        st.markdown(event["text"])
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
        tab_labels = ["Rendered", "Raw Markdown"]
        if pages_data:
            tab_labels.append("Bounding Boxes")
        tabs = st.tabs(tab_labels)
        with tabs[0]:
            st.markdown(full_text)
        with tabs[1]:
            st.text_area("Raw markdown", full_text, height=400, label_visibility="collapsed")
        if pages_data:
            with tabs[2]:
                options = [p["index"] for p in pages_data]
                page_index = st.selectbox(
                    "Page", options, format_func=lambda i: f"Page {i}/{pages_data[0]['total']}"
                )
                page = next(p for p in pages_data if p["index"] == page_index)
                col_image, col_text = st.columns(2)
                with col_image:
                    st.image(_draw_boxes(page["image_b64"], page["blocks"]), use_container_width=True)
                with col_text:
                    st.markdown(page["text"])

        st.download_button(
            label="Download Markdown (.md)",
            data=full_text,
            file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}.md",
            mime="text/markdown",
        )
