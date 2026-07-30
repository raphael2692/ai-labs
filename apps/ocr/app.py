import time

import streamlit as st
from ai_lab_common import get_settings, ocr_stream
from ai_lab_common.ocr_client import OcrError
from ai_lab_common.sidebar import render_endpoint_sidebar

from dets import decode_data_uri, draw_dets, parse_dets, results_to_zip

PANE_HEIGHT = 850
IMG_THROTTLE = 0.5  # seconds between box-overlay redraws while streaming (each redraw re-encodes a PNG)

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
col_img, col_txt = st.columns([2, 3])
col_img.caption("Source page")
col_txt.caption("Extracted text")
img_ph = col_img.container(height=PANE_HEIGHT).empty()
txt_ph = col_txt.container(height=PANE_HEIGHT).empty()


def _show_metrics(pages_label: str, elapsed: float, chars: int) -> None:
    metrics_ph.markdown(f"📄 **{pages_label}**   ·   ⏱ **{elapsed:.1f}s**   ·   🔤 **{chars:,}** chars")


def run_ocr() -> None:
    st.session_state.results = []
    st.session_state.pagenum = 1
    current_image = None
    page_raw_chunks: list[str] = []
    last_img_draw = 0.0
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
                last_img_draw = 0.0
                img_ph.image(current_image, use_container_width=True)
                txt_ph.text("")
                status_ph.info(f"Parsing page {event['index']}/{event['total']}...")

            elif event["type"] == "chunk":
                page_raw_chunks.append(event["text"])
                raw_so_far = "".join(page_raw_chunks)
                txt_ph.text(raw_so_far)
                now = time.time()
                if show_boxes and current_image is not None and now - last_img_draw >= IMG_THROTTLE:
                    dets = parse_dets(raw_so_far)
                    if dets:
                        img_ph.image(draw_dets(current_image, dets), use_container_width=True)
                    last_img_draw = now

            elif event["type"] == "page":
                overlay = current_image
                if show_boxes and current_image is not None:
                    overlay = draw_dets(current_image, parse_dets(event["raw_text"]))
                img_ph.image(overlay, use_container_width=True)
                txt_ph.text(event["text"])
                st.session_state.results.append(
                    {"overlay": overlay, "text": event["text"], "raw_text": event["raw_text"]}
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
    img_ph.image(r["overlay"], use_container_width=True)
    txt_ph.text(r["text"])
    with st.expander("Raw model output (with <|det|> markers)"):
        st.code(r["raw_text"])


if run and uploaded_file is not None:
    run_ocr()
elif st.session_state.results:
    review()
else:
    img_ph.info("Upload a document and click Run OCR.")
    txt_ph.write("Extracted text will appear here.")

if st.session_state.results:
    st.divider()
    stem = uploaded_file.name.rsplit(".", 1)[0] if uploaded_file is not None else "ocr"
    full_text = st.session_state.full_text or "\n\n".join(r["text"] for r in st.session_state.results)
    zip_bytes = results_to_zip(st.session_state.results)

    c1, c2 = st.columns(2)
    c1.download_button(
        "⬇️ Download Text (.txt)", data=full_text, file_name=f"{stem}.txt", mime="text/plain"
    )
    c2.download_button(
        "⬇️ Download Pages (.zip)",
        data=zip_bytes,
        file_name=f"{stem}_ocr.zip",
        mime="application/zip",
        help="One folder per page: box-overlay image, raw model output, and clean text.",
    )
