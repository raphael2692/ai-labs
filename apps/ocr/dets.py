"""Parses the OCR model's <|det|> grounding markers and renders them as
bounding-box overlays, mirroring the post-processing recipe from the model
card. Kept alongside app.py (not in ai_lab_common) since it's OCR-specific
and only this app consumes it; Streamlit adds the script's own directory to
sys.path, so `import dets` from app.py resolves this file directly."""

import ast
import base64
import html
import io
import json
import re
import zipfile

from PIL import Image, ImageDraw, ImageFont

# Matches e.g. "<|det|>header [123, 29, 322, 75]<|/det|>" — coordinates are
# normalized to a 0-999 scale relative to the image sent to the model.
_DET_RE = re.compile(r"<\|det\|>\s*([A-Za-z_][\w-]*)\s*(\[[0-9,\s.\[\]]+\])\s*<\|/det\|>")

LABEL_COLORS = {
    "title": (220, 40, 40),
    "header": (230, 140, 20),
    "text": (40, 110, 220),
    "image": (30, 170, 90),
    "image_caption": (20, 160, 160),
    "table": (150, 60, 200),
    "table_caption": (120, 80, 200),
    "list": (160, 100, 40),
    "figure": (30, 170, 90),
    "formula": (200, 60, 160),
    "page_number": (130, 130, 130),
    "footer": (130, 130, 130),
}
_DEFAULT_COLOR = (200, 60, 160)


def _boxes_from_str(box_str: str) -> list[tuple[float, float, float, float]]:
    try:
        val = ast.literal_eval(box_str)
    except (ValueError, SyntaxError):
        return []
    if not val:
        return []
    if isinstance(val[0], (int, float)):
        val = [val]
    return [tuple(float(x) for x in b[:4]) for b in val if isinstance(b, (list, tuple)) and len(b) >= 4]


def parse_dets(raw_text: str) -> list[tuple[str, list[tuple[float, float, float, float]]]]:
    """Extracts (label, boxes) pairs from raw model output. Coordinates stay
    normalized to 0-999; safe to call on partial/still-streaming text since
    only fully-closed <|det|>...<|/det|> markers match."""
    dets = []
    for label, box_str in _DET_RE.findall(raw_text):
        boxes = _boxes_from_str(box_str)
        if boxes:
            dets.append((label.strip(), boxes))
    return dets


def parse_blocks(raw_text: str) -> list[dict]:
    """Pairs each detection box with the text that follows it up to the next
    marker (or end of page), so a box and its corresponding text can be linked
    in the UI. Ordered list of {"label", "boxes", "text"}; grows monotonically
    as more of raw_text closes (only fully-closed markers match), so calling
    this again on a longer prefix reuses the same indices for prior blocks —
    safe to re-render from scratch on every streamed chunk."""
    matches = list(_DET_RE.finditer(raw_text))
    blocks = []
    for i, m in enumerate(matches):
        label = m.group(1).strip()
        boxes = _boxes_from_str(m.group(2))
        if not boxes or label == "image":
            continue
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        blocks.append({"label": label, "boxes": boxes, "text": raw_text[start:end].strip()})
    return blocks


def decode_data_uri(data_uri: str) -> Image.Image:
    _, b64 = data_uri.split(",", 1)
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _encode_image(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def render_interactive_page(
    image: Image.Image, blocks: list[dict], text: str = "", show_boxes: bool = True, height: int = 850
) -> str:
    """Builds a self-contained HTML fragment (for st.components.v1.html): the
    source image on the left with one clickable box per detected block, and
    the corresponding text blocks on the right. Clicking a box scrolls to and
    highlights its text, and vice versa — pure client-side JS, since both
    panes live in the same iframe and no Streamlit round-trip is needed.
    Falls back to plain `text` when no <|det|> blocks were parsed (e.g. the
    model didn't emit grounding markers for this page), so text is never
    silently hidden."""
    box_divs = []
    text_divs = []
    for i, block in enumerate(blocks):
        color = LABEL_COLORS.get(block["label"], _DEFAULT_COLOR)
        rgb = f"rgb({color[0]},{color[1]},{color[2]})"
        label = html.escape(block["label"])
        if show_boxes:
            border_w = 3 if block["label"] == "title" else 2
            for x1, y1, x2, y2 in block["boxes"]:
                left, top = min(x1, x2) / 999 * 100, min(y1, y2) / 999 * 100
                w, h = abs(x2 - x1) / 999 * 100, abs(y2 - y1) / 999 * 100
                if w <= 0 or h <= 0:
                    continue
                box_divs.append(
                    f'<div id="box-{i}" onclick="scrollToBlock({i})" title="{label}" '
                    f'style="position:absolute; left:{left:.2f}%; top:{top:.2f}%; '
                    f'width:{w:.2f}%; height:{h:.2f}%; border:{border_w}px solid {rgb}; '
                    f'background:rgba({color[0]},{color[1]},{color[2]},0.12); '
                    f'cursor:pointer; box-sizing:border-box;"></div>'
                )
        text_divs.append(
            f'<div id="block-{i}" onclick="scrollToBox({i})" '
            f'style="border-left:4px solid {rgb}; padding:4px 10px; margin-bottom:8px; cursor:pointer;">'
            f'<span style="font-size:11px; color:{rgb}; font-weight:600;">{label}</span><br>'
            f'<span style="white-space:pre-wrap;">{html.escape(block["text"])}</span></div>'
        )

    img_uri = _encode_image(image)
    if text_divs:
        text_html = "".join(text_divs)
    elif text.strip():
        text_html = f'<span style="white-space:pre-wrap;">{html.escape(text)}</span>'
    else:
        text_html = '<span style="opacity:0.6;">Extracted text will appear here.</span>'

    return f"""
<div style="display:flex; gap:12px; height:{height}px; font-family:sans-serif; background:#fff; color:#1a1a1a;">
  <div style="flex:2; overflow:auto; border:1px solid rgba(128,128,128,0.4); border-radius:6px; background:#fff;">
    <div style="position:relative;">
      <img src="{img_uri}" style="width:100%; display:block;">
      {"".join(box_divs)}
    </div>
  </div>
  <div style="flex:3; overflow:auto; border:1px solid rgba(128,128,128,0.4); border-radius:6px; padding:8px;
              font-size:14px; line-height:1.4; background:#fff; color:#1a1a1a;">
    {text_html}
  </div>
</div>
<script>
function scrollToBlock(i) {{
  var el = document.getElementById('block-' + i);
  if (!el) return;
  el.scrollIntoView({{behavior: 'smooth', block: 'center'}});
  var orig = el.style.backgroundColor;
  el.style.transition = 'background-color 0.3s';
  el.style.backgroundColor = 'rgba(255,225,0,0.35)';
  setTimeout(function() {{ el.style.backgroundColor = orig; }}, 1200);
}}
function scrollToBox(i) {{
  var el = document.getElementById('box-' + i);
  if (!el) return;
  el.scrollIntoView({{behavior: 'smooth', block: 'center'}});
  var orig = el.style.boxShadow;
  el.style.transition = 'box-shadow 0.3s';
  el.style.boxShadow = '0 0 0 4px rgba(255,225,0,0.9)';
  setTimeout(function() {{ el.style.boxShadow = orig || 'none'; }}, 1200);
}}
</script>
"""


def draw_dets(base_img: Image.Image, dets: list[tuple[str, list[tuple[float, float, float, float]]]]) -> Image.Image:
    """Overlays parsed detection boxes on a copy of base_img."""
    img = base_img.copy()
    draw = ImageDraw.Draw(img, "RGBA")
    width_px, height_px = img.size
    font = ImageFont.load_default()
    for label, boxes in dets:
        color = LABEL_COLORS.get(label, _DEFAULT_COLOR)
        outline_width = 4 if label == "title" else 2
        for x1, y1, x2, y2 in boxes:
            px1, px2 = sorted((int(x1 / 999 * width_px), int(x2 / 999 * width_px)))
            py1, py2 = sorted((int(y1 / 999 * height_px), int(y2 / 999 * height_px)))
            px1, px2 = max(0, px1), min(width_px - 1, px2)
            py1, py2 = max(0, py1), min(height_px - 1, py2)
            if px2 <= px1 or py2 <= py1:
                continue
            draw.rectangle([px1, py1, px2, py2], outline=color, width=outline_width)
            draw.rectangle([px1, py1, px2, py2], fill=color + (28,))
            tag_y = max(0, py1 - 13)
            tb = draw.textbbox((0, 0), label, font=font)
            tag_w, tag_h = tb[2] - tb[0], tb[3] - tb[1]
            draw.rectangle([px1, tag_y, px1 + tag_w + 4, tag_y + tag_h + 2], fill=(255, 255, 255, 220))
            draw.text((px1 + 2, tag_y), label, font=font, fill=color)
    return img


def results_to_document(results: list[dict]) -> list[dict]:
    """Builds the structured, labeled representation of every page: each
    block's label (header/text/footer/title/table/...), text, and normalized
    box coordinates, keyed by page number. This is what survives when the
    flattened .txt export would otherwise lose which piece of text was a
    header vs. a footer vs. body text."""
    return [
        {
            "page": i,
            "blocks": [{"label": b["label"], "text": b["text"], "boxes": b["boxes"]} for b in r.get("blocks", [])],
        }
        for i, r in enumerate(results, 1)
    ]


def results_to_zip(results: list[dict]) -> bytes:
    """Packs per-page results into a ZIP: one folder per page with the
    box-overlay image, the model's raw output, the clean text, and a
    blocks.json preserving each detected block's label (header/text/footer/
    title/table/...), text, and normalized box coordinates. Also writes a
    top-level document.json with every page's blocks together, so labeled
    structure survives the download instead of only the flattened text."""
    buf = io.BytesIO()
    document = results_to_document(results)
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for i, r in enumerate(results, 1):
            folder = f"page_{i:02d}"
            png_buf = io.BytesIO()
            r["overlay"].save(png_buf, format="PNG")
            z.writestr(f"{folder}/overlay.png", png_buf.getvalue())
            z.writestr(f"{folder}/raw.txt", r.get("raw_text", ""))
            z.writestr(f"{folder}/text.txt", r.get("text", ""))
            z.writestr(f"{folder}/blocks.json", json.dumps(document[i - 1]["blocks"], indent=2, ensure_ascii=False))
        z.writestr("document.json", json.dumps(document, indent=2, ensure_ascii=False))
    return buf.getvalue()
