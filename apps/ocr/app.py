import json
import time

import streamlit as st
import streamlit.components.v1 as components
from ai_lab_common import get_settings, ocr_stream
from ai_lab_common.ocr_client import OcrError
from ai_lab_common.sidebar import render_endpoint_sidebar

from dets import (
    decode_data_uri,
    draw_dets,
    parse_blocks,
    parse_dets,
    render_interactive_page,
    results_to_document,
    results_to_zip,
)

PANE_HEIGHT = 850
RENDER_THROTTLE = 0.8  # seconds between live redraws (each redraw re-mounts the component's iframe, causing a flash)

st.set_page_config(page_title="OCR", page_icon="📄", layout="wide")

st.session_state.setdefault("results", [])
st.session_state.setdefault("pagenum", 1)
st.session_state.setdefault("full_text", "")

with st.sidebar:
    st.title("📄 OCR")
    settings = render_endpoint_sidebar(get_settings())
    image_mode = st.selectbox(
        "image_mode",
        ["gundam", "base"],
        help="gundam: cropped detail tiles, best for a single detail-heavy page. "
        "base: one uncropped pass, faster for multi-page/PDF.",
    )
    show_boxes = st.checkbox("Draw detection boxes on image", value=True)

uploaded_file = st.file_uploader("Choose a document", type=["pdf", "png", "jpg", "jpeg"])
run = st.button("🚀 Run OCR", type="primary", disabled=uploaded_file is None)

status_ph = st.empty()
metrics_ph = st.empty()
nav_ph = st.empty()
view_ph = st.empty()


def _show_metrics(pages_label: str, elapsed: float, chars: int) -> None:
    metrics_ph.markdown(f"📄 **{pages_label}**   ·   ⏱ **{elapsed:.1f}s**   ·   🔤 **{chars:,}** chars")


def _render(image, blocks, text: str = "") -> None:
    with view_ph.container():
        components.html(
            render_interactive_page(image, blocks, text=text, show_boxes=show_boxes, height=PANE_HEIGHT),
            height=PANE_HEIGHT + 24,
            scrolling=False,
        )


def run_ocr() -> None:
    st.session_state.results = []
    st.session_state.pagenum = 1
    current_image = None
    page_raw_chunks: list[str] = []
    last_render = 0.0
    last_block_count = 0
    t_start = time.time()

    try:
        status_ph.info("Sending file to OCR server...")
        for event in ocr_stream(
            settings.ocr_api_url, uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type, image_mode
        ):
            if event["type"] == "info":
                status_ph.info(f"Parsing {event['pages']} page(s)...")

            elif event["type"] == "page_start":
                current_image = decode_data_uri(event["image"])
                page_raw_chunks = []
                last_render = 0.0
                last_block_count = 0
                _render(current_image, [])
                status_ph.info(f"Parsing page {event['index']}/{event['total']}...")

            elif event["type"] == "chunk":
                page_raw_chunks.append(event["text"])
                now = time.time()
                raw_so_far = "".join(page_raw_chunks)
                blocks_so_far = parse_blocks(raw_so_far)
                # Only re-mount the component (which flashes) when a new block actually
                # closed, or on the regular throttle tick for a fresh block of plain text.
                if current_image is not None and (
                    len(blocks_so_far) != last_block_count or now - last_render >= RENDER_THROTTLE
                ):
                    _render(current_image, blocks_so_far, text=raw_so_far)
                    last_render = now
                    last_block_count = len(blocks_so_far)

            elif event["type"] == "page":
                # Use the client's own accumulated stream as the source of truth for
                # <|det|> markers: the server's final `raw_text` (event["raw_text"]) is
                # sometimes re-extracted from a saved result file that already had the
                # markers stripped, which would silently drop every bounding box at the
                # moment the page finishes even though they were visible while streaming.
                raw_text = "".join(page_raw_chunks) or event["raw_text"]
                blocks = parse_blocks(raw_text)
                overlay = draw_dets(current_image, parse_dets(raw_text)) if current_image is not None else None
                if current_image is not None:
                    _render(current_image, blocks, text=event["text"])
                st.session_state.results.append(
                    {
                        "image": current_image,
                        "overlay": overlay,
                        "blocks": blocks,
                        "text": event["text"],
                        "raw_text": raw_text,
                    }
                )
                total_chars = sum(len(r["text"]) for r in st.session_state.results)
                _show_metrics(f"{event['index']}/{event['total']}", time.time() - t_start, total_chars)

            elif event["type"] == "done":
                st.session_state.full_text = event["text"]
                status_ph.success(f"OCR complete! ({event['pages']} page(s) in {event['elapsed']:.1f}s)")
    except OcrError as e:
        status_ph.error(f"OCR failed: {e}")
        st.stop()
    except Exception as e:
        status_ph.error(f"Could not reach OCR server at {settings.ocr_api_url}: {e}")
        st.stop()

    st.rerun()


def _navigate(n: int) -> int:
    if n <= 1:
        st.session_state.pagenum = 1
        return 1
    st.session_state.pagenum = nav_ph.slider("Page", 1, n, min(st.session_state.pagenum, n))
    return st.session_state.pagenum


def review() -> None:
    results = st.session_state.results
    page = _navigate(len(results))
    r = results[page - 1]
    _render(r["image"], r["blocks"], text=r["text"])
    with st.expander("Raw model output (with <|det|> markers)"):
        st.code(r["raw_text"])


if run and uploaded_file is not None:
    run_ocr()
elif st.session_state.results:
    review()
else:
    with view_ph.container():
        st.info("Upload a document and click Run OCR. Extracted text and detection boxes will appear here.")

if st.session_state.results:
    st.divider()
    stem = uploaded_file.name.rsplit(".", 1)[0] if uploaded_file is not None else "ocr"
    full_text = st.session_state.full_text or "\n\n".join(r["text"] for r in st.session_state.results)
    zip_bytes = results_to_zip(st.session_state.results)
    document_json = json.dumps(results_to_document(st.session_state.results), indent=2, ensure_ascii=False)

    c1, c2, c3 = st.columns(3)
    c1.download_button(
        "Download Text (.txt)", data=full_text, file_name=f"{stem}.txt", mime="text/plain"
    )
    c2.download_button(
        "Download Structure (.json)",
        data=document_json,
        file_name=f"{stem}_ocr.json",
        mime="application/json",
        help="Per page, per block: label (header, text, footer, title, table, ...), text, and normalized boxes.",
    )
    c3.download_button(
        "Download Pages (.zip)",
        data=zip_bytes,
        file_name=f"{stem}_ocr.zip",
        mime="application/zip",
        help="One folder per page: box-overlay image, raw model output, clean text, and blocks.json "
        "(label/text/boxes per block — header, text, footer, title, table, ...). Plus a top-level "
        "document.json with every page's blocks together.",
    )
