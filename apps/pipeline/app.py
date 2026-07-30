"""Extract -> optional content revision -> formatting, with live streaming
and abort at every stage.

Both extraction (OCR/Whisper) and each LLM generation step run in a
background thread that only pushes plain dicts onto a queue.Queue; the main
script polls that queue from an st.fragment(run_every=...) so the UI updates
without blocking, and a Stop button (also inside the fragment, so it's live
during the run) sets a threading.Event the worker checks between events. This
is the only way to get both live-streaming display and true mid-request abort
out of Streamlit's single-threaded rerun model.

Prompts (section 1) are configured up front and are independent of the
pipeline stage — not something the pipeline unlocks partway through. There
are two independent prompt slots — content revision (optional) and
formatting (always run) — each pulling from the same saved-prompt library but
tracked with its own selection.
"""

import json
import queue
import re
import threading

import streamlit as st
from ai_lab_common import get_llm_client, get_settings, ocr_stream, transcribe_stream
from ai_lab_common.ocr_client import OcrError
from ai_lab_common.sidebar import render_endpoint_sidebar
from ai_lab_common.whisper_client import WhisperTranscriptionError

from prompts import DEFAULT_FORMAT_PROMPT, DEFAULT_REVISION_PROMPT, load_prompts, save_prompts
from structure import parse_labeled_blocks

POLL_INTERVAL = "0.3s"

st.set_page_config(page_title="Pipeline", page_icon="🧵", layout="wide")

st.session_state.setdefault("stage", "idle")  # idle -> extracting -> extracted
#   -> [revising -> revised ->] formatting -> done   (revising/revised skipped if revision disabled)
st.session_state.setdefault("error", None)
st.session_state.setdefault("extract_pages", [])  # [{"index", "text", "blocks"}] for pdf/image
st.session_state.setdefault("extract_live_raw", "")
st.session_state.setdefault("extract_total_pages", None)
st.session_state.setdefault("extract_meta", "")  # info line (language/duration, or page progress)
st.session_state.setdefault("final_text", "")
st.session_state.setdefault("final_structure", None)  # list of page dicts, or None for audio
st.session_state.setdefault("revision_enabled", True)
st.session_state.setdefault("revised_text", "")
st.session_state.setdefault("current_step", None)  # "revision" | "format"
st.session_state.setdefault("md_text", "")
st.session_state.setdefault("file_kind", None)  # "audio" | "document"
st.session_state.setdefault("file_stem", "pipeline")
st.session_state.setdefault("prompts", load_prompts())

_prompts = st.session_state.prompts
st.session_state.setdefault(
    "selected_prompt_revision", DEFAULT_REVISION_PROMPT if DEFAULT_REVISION_PROMPT in _prompts else next(iter(_prompts))
)
st.session_state.setdefault(
    "selected_prompt_format", DEFAULT_FORMAT_PROMPT if DEFAULT_FORMAT_PROMPT in _prompts else next(iter(_prompts))
)

# A selectbox bound to key=<select_key> can't be reassigned after it's
# instantiated this run (Streamlit forbids mutating a widget's own key
# mid-run). Handlers that need to change a selection stash it here and rerun;
# this applies it *before* the corresponding selectbox is created below.
for _select_key, _target in st.session_state.pop("_pending_selects", {}).items():
    if _target in st.session_state.prompts:
        st.session_state[_select_key] = _target


# --- Background workers -----------------------------------------------------
# Run in a thread; must never touch st.* (not safe off the main thread). They
# only push {"type": ...} dicts onto q and check cancel_event between events.


def _extract_worker(kind, name, data, content_type, settings, image_mode, q, cancel_event):
    try:
        if kind == "audio":
            gen = transcribe_stream(settings.whisper_api_url, name, data, content_type)
        else:
            gen = ocr_stream(settings.ocr_api_url, name, data, content_type, image_mode)
        for event in gen:
            if cancel_event.is_set():
                gen.close()
                q.put({"type": "cancelled"})
                return
            q.put({"type": "event", "event": event})
        q.put({"type": "finished"})
    except (WhisperTranscriptionError, OcrError) as e:
        q.put({"type": "error", "message": str(e)})
    except Exception as e:
        q.put({"type": "error", "message": f"Could not reach the extraction server: {e}"})


def _generate_worker(client, model, system_prompt, user_content, q, cancel_event):
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            stream=True,
        )
        for chunk in stream:
            if cancel_event.is_set():
                stream.close()
                q.put({"type": "cancelled"})
                return
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                q.put({"type": "delta", "text": delta})
        q.put({"type": "finished"})
    except Exception as e:
        q.put({"type": "error", "message": f"Generation failed: {e}"})


def _build_user_content(text: str, structure: list | None) -> str:
    content = f"Text to process:\n\n{text}"
    if structure:
        content += (
            "\n\n---\nStructured extraction (JSON, per page/label):\n"
            + json.dumps(structure, ensure_ascii=False, indent=2)
        )
    return content


def _start_generation(step: str, system_prompt: str, user_content: str) -> None:
    cancel_event = threading.Event()
    q = queue.Queue()
    client = get_llm_client(settings)
    thread = threading.Thread(
        target=_generate_worker,
        args=(client, settings.unsloth_model, system_prompt, user_content, q, cancel_event),
        daemon=True,
    )
    thread.start()
    st.session_state.generate_queue = q
    st.session_state.generate_cancel = cancel_event
    st.session_state.current_step = step
    st.session_state.error = None
    if step == "revision":
        st.session_state.revised_text = ""
        st.session_state.stage = "revising"
    else:
        st.session_state.md_text = ""
        st.session_state.stage = "formatting"
    st.rerun()


def _reset_pipeline():
    for key in ("extract_pages", "extract_live_raw", "extract_meta", "final_text", "revised_text", "md_text"):
        st.session_state[key] = "" if isinstance(st.session_state[key], str) else []
    st.session_state.extract_total_pages = None
    st.session_state.final_structure = None
    st.session_state.current_step = None
    st.session_state.error = None
    st.session_state.stage = "idle"


_FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


def _render_markdown_preview(md_text: str) -> None:
    """Renders md_text for on-screen preview only — the downloaded file keeps
    the raw front matter block as-is. Streamlit's markdown has no notion of
    Jekyll-style front matter, so a leading `---\\ntitle: ...\\n---` block would
    otherwise show up as a stray horizontal rule and a plain-text paragraph;
    here title/subtitle render as an H1/H2 ahead of the body instead."""
    match = _FRONT_MATTER_RE.match(md_text)
    with st.container(border=True):
        if not match:
            st.markdown(md_text)
            return

        meta = {}
        for line in match.group(1).splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                meta[key.strip().lower()] = value.strip().strip("\"'")

        if meta.get("title"):
            st.markdown(f"# {meta['title']}")
        if meta.get("subtitle"):
            st.markdown(f"## {meta['subtitle']}")
        st.markdown(md_text[match.end() :])


def _prompt_editor(title: str, select_key: str, enabled: bool = True) -> str:
    """Renders one saved-prompt selector + editable instructions + save/delete,
    all pulling from the shared st.session_state.prompts library. Returns the
    (possibly edited, not-yet-saved) instructions text currently shown."""
    prompts = st.session_state.prompts
    col1, col2 = st.columns([3, 1])
    with col1:
        preset_name = st.selectbox(f"{title} — saved prompt", list(prompts.keys()), key=select_key, disabled=not enabled)
    prompt_text = st.text_area(
        f"{title} — instructions",
        value=prompts.get(preset_name, ""),
        height=160,
        key=f"{select_key}_editor_{preset_name}",
        disabled=not enabled,
    )
    with col2:
        new_name = st.text_input("Save as", key=f"{select_key}_save_as", disabled=not enabled)
        if st.button("💾 Save", key=f"{select_key}_save_btn", disabled=not enabled):
            name = new_name.strip() or preset_name
            prompts[name] = prompt_text
            save_prompts(prompts)
            st.session_state.prompts = prompts
            st.session_state["_pending_selects"] = {select_key: name}
            st.rerun()
        if len(prompts) > 1 and st.button("🗑 Delete", key=f"{select_key}_del_btn", disabled=not enabled):
            prompts.pop(preset_name, None)
            save_prompts(prompts)
            st.session_state.prompts = prompts
            st.session_state["_pending_selects"] = {select_key: next(iter(prompts))}
            st.rerun()
    return prompt_text


# --- UI ----------------------------------------------------------------------

with st.sidebar:
    st.title("🧵 Pipeline")
    settings = render_endpoint_sidebar(get_settings())
    image_mode = st.selectbox(
        "OCR image_mode",
        ["gundam", "base"],
        help="Only used for PDF/image input. gundam: cropped detail tiles, best for a single detail-heavy "
        "page. base: one uncropped pass, faster for multi-page/PDF.",
    )
    if st.session_state.stage != "idle" and st.button("↺ Start over"):
        _reset_pipeline()
        st.rerun()

st.title("Extract → Revise → Format")

# --- 1. Prompt setup — configured up front, independent of pipeline stage ---

st.subheader("1. Prompts")

st.checkbox(
    "Enable content revision step (optional pass before formatting)",
    key="revision_enabled",
)
revision_prompt_text = _prompt_editor(
    "Content revision", "selected_prompt_revision", enabled=st.session_state.revision_enabled
)
st.divider()
formatting_prompt_text = _prompt_editor("Formatting", "selected_prompt_format")

# --- 2. Upload & extract ------------------------------------------------------

st.divider()
st.subheader("2. Upload & extract")

uploaded_file = st.file_uploader(
    "Choose an audio file, PDF, or image",
    type=["wav", "mp3", "m4a", "ogg", "flac", "webm", "pdf", "png", "jpg", "jpeg"],
    disabled=st.session_state.stage != "idle",
)
start = st.button(
    "🚀 Extract",
    type="primary",
    disabled=uploaded_file is None or st.session_state.stage != "idle",
)

if start and uploaded_file is not None:
    kind = "audio" if uploaded_file.type.startswith("audio/") else "document"
    st.session_state.file_kind = kind
    st.session_state.file_stem = uploaded_file.name.rsplit(".", 1)[0]
    cancel_event = threading.Event()
    q = queue.Queue()
    thread = threading.Thread(
        target=_extract_worker,
        args=(kind, uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type, settings, image_mode, q,
              cancel_event),
        daemon=True,
    )
    thread.start()
    st.session_state.extract_queue = q
    st.session_state.extract_cancel = cancel_event
    st.session_state.extract_pages = []
    st.session_state.extract_live_raw = ""
    st.session_state.extract_total_pages = None
    st.session_state.extract_audio_duration = None
    st.session_state.extract_meta = ""
    st.session_state.error = None
    st.session_state.stage = "extracting"
    st.rerun()


@st.fragment(run_every=POLL_INTERVAL)
def _poll_extraction():
    q = st.session_state.extract_queue
    kind = st.session_state.file_kind
    drained = False
    while not q.empty():
        item = q.get()
        drained = True
        if item["type"] == "event":
            _handle_extract_event(kind, item["event"])
        elif item["type"] == "finished":
            _finish_extraction()
            st.rerun()
        elif item["type"] == "cancelled":
            _finish_extraction(cancelled=True)
            st.rerun()
        elif item["type"] == "error":
            st.session_state.error = item["message"]
            st.session_state.stage = "idle"
            st.rerun()

    st.info(st.session_state.extract_meta or "Sending file to the extraction server...")
    live_preview = st.session_state.extract_live_raw
    if kind == "document" and live_preview:
        live_preview = "\n\n".join(f"**{b['label']}**: {b['text']}" for b in parse_labeled_blocks(live_preview))
    pages_so_far = "\n\n".join(p["text"] for p in st.session_state.extract_pages)
    st.text_area("Live extraction", (pages_so_far + "\n\n" + live_preview).strip(), height=280, disabled=True)
    st.button("⏹ Stop extraction", on_click=lambda: st.session_state.extract_cancel.set())
    if not drained:
        return  # nothing new this tick, avoid a redundant rerun


def _handle_extract_event(kind, event):
    if kind == "audio":
        if event["type"] == "info":
            st.session_state.extract_audio_duration = event["duration"]
            st.session_state.extract_meta = f"Language: {event['language']} · Duration: {event['duration']:.1f}s"
        elif event["type"] == "segment":
            st.session_state.extract_live_raw += event["text"] + " "
            duration = st.session_state.get("extract_audio_duration")
            duration_str = f"{duration:.1f}s" if duration is not None else "?"
            st.session_state.extract_meta = f"Transcribing... {event['end']:.1f}s / {duration_str}"
        elif event["type"] == "done":
            st.session_state.final_text = event["text"]
    else:
        if event["type"] == "info":
            st.session_state.extract_total_pages = event["pages"]
            st.session_state.extract_meta = f"Parsing {event['pages']} page(s)..."
        elif event["type"] == "page_start":
            st.session_state.extract_live_raw = ""
            st.session_state.extract_meta = f"Parsing page {event['index']}/{event['total']}..."
        elif event["type"] == "chunk":
            st.session_state.extract_live_raw += event["text"]
        elif event["type"] == "page":
            raw = st.session_state.extract_live_raw or event["raw_text"]
            st.session_state.extract_pages.append(
                {"index": event["index"], "text": event["text"], "blocks": parse_labeled_blocks(raw)}
            )
            st.session_state.extract_live_raw = ""
        elif event["type"] == "done":
            st.session_state.final_text = event["text"]


def _finish_extraction(cancelled: bool = False):
    if not st.session_state.final_text:
        st.session_state.final_text = "\n\n".join(p["text"] for p in st.session_state.extract_pages)
    if st.session_state.file_kind == "document" and st.session_state.extract_pages:
        st.session_state.final_structure = [
            {"page": p["index"], "blocks": p["blocks"]} for p in st.session_state.extract_pages
        ]
    st.session_state.stage = "extracted"
    if cancelled:
        st.session_state.error = "Extraction stopped early — showing partial results."


if st.session_state.stage == "extracting":
    _poll_extraction()

if st.session_state.error:
    st.warning(st.session_state.error)

# --- 3. Extracted text --------------------------------------------------------

_POST_EXTRACT_STAGES = ("extracted", "revising", "revised", "formatting", "done")

if st.session_state.stage in _POST_EXTRACT_STAGES:
    st.divider()
    st.subheader("3. Extracted text")
    st.text_area("Extracted text", st.session_state.final_text, height=220, disabled=True)
    if st.session_state.final_structure:
        with st.expander(f"Structured extraction ({len(st.session_state.final_structure)} page(s), JSON)"):
            st.json(st.session_state.final_structure)

    if st.session_state.stage == "extracted":
        user_content = _build_user_content(st.session_state.final_text, st.session_state.final_structure)
        if st.session_state.error:
            # A generation error bounced the stage back to "extracted" — don't
            # auto-retry into the same failure, wait for an explicit retry.
            if st.button("🔁 Retry"):
                if st.session_state.revision_enabled:
                    _start_generation("revision", revision_prompt_text, user_content)
                else:
                    _start_generation("format", formatting_prompt_text, user_content)
        elif st.session_state.revision_enabled:
            _start_generation("revision", revision_prompt_text, user_content)
        else:
            _start_generation("format", formatting_prompt_text, user_content)


@st.fragment(run_every=POLL_INTERVAL)
def _poll_generation():
    step = st.session_state.current_step
    buffer_key = "revised_text" if step == "revision" else "md_text"
    q = st.session_state.generate_queue
    drained = False
    while not q.empty():
        item = q.get()
        drained = True
        if item["type"] == "delta":
            st.session_state[buffer_key] += item["text"]
        elif item["type"] == "finished":
            st.session_state.stage = "revised" if step == "revision" else "done"
            st.rerun()
        elif item["type"] == "cancelled":
            st.session_state.stage = "revised" if step == "revision" else "done"
            what = "Content revision" if step == "revision" else "Formatting"
            st.session_state.error = f"{what} stopped early — showing partial result."
            st.rerun()
        elif item["type"] == "error":
            st.session_state.error = item["message"]
            st.session_state.stage = "extracted" if step == "revision" or not st.session_state.revision_enabled else "revised"
            st.rerun()

    st.info("Revising content..." if step == "revision" else "Formatting...")
    if step == "revision":
        st.text_area("Live revision", st.session_state.revised_text, height=240, disabled=True)
    else:
        _render_markdown_preview(st.session_state.md_text)
    st.button("⏹ Stop", on_click=lambda: st.session_state.generate_cancel.set(), key=f"stop_{step}")
    if not drained:
        return


# --- 5. Content revision (optional) ------------------------------------------

if st.session_state.revision_enabled and st.session_state.stage in ("revising", "revised", "formatting", "done"):
    st.divider()
    st.subheader("5. Content revision")
    if st.session_state.stage == "revising":
        _poll_generation()
    else:
        st.text_area(
            "Revised content",
            key="revised_text",
            height=240,
            disabled=True,
        )
        if st.session_state.stage == "revised":
            user_content = _build_user_content(st.session_state.revised_text, None)
            if st.session_state.error:
                if st.button("🔁 Retry formatting"):
                    _start_generation("format", formatting_prompt_text, user_content)
            else:
                _start_generation("format", formatting_prompt_text, user_content)

# --- 6. Result -----------------------------------------------------------------

if st.session_state.stage in ("formatting", "done"):
    st.divider()
    st.subheader("6. Result")
    if st.session_state.stage == "formatting":
        _poll_generation()
    else:
        _render_markdown_preview(st.session_state.md_text)
        st.download_button(
            "⬇ Download Markdown",
            data=st.session_state.md_text,
            file_name=f"{st.session_state.file_stem}.md",
            mime="text/markdown",
            type="primary",
        )

        with st.expander("⬇ Download intermediate results (resume from an earlier step elsewhere)"):
            st.download_button(
                "Extracted text (.txt)",
                data=st.session_state.final_text,
                file_name=f"{st.session_state.file_stem}.extracted.txt",
                mime="text/plain",
                key="dl_extracted_text",
            )
            if st.session_state.final_structure:
                st.download_button(
                    "Structured extraction (.json)",
                    data=json.dumps(st.session_state.final_structure, ensure_ascii=False, indent=2),
                    file_name=f"{st.session_state.file_stem}.extracted.json",
                    mime="application/json",
                    key="dl_extracted_json",
                )
            if st.session_state.revision_enabled:
                st.download_button(
                    "Revised text (.txt)",
                    data=st.session_state.revised_text,
                    file_name=f"{st.session_state.file_stem}.revised.txt",
                    mime="text/plain",
                    key="dl_revised_text",
                )
