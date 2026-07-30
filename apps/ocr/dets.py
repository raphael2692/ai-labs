"""Parses the OCR model's <|det|> grounding markers and renders them as
bounding-box overlays, mirroring the post-processing recipe from the model
card. Kept alongside app.py (not in ai_lab_common) since it's OCR-specific
and only this app consumes it; Streamlit adds the script's own directory to
sys.path, so `import dets` from app.py resolves this file directly."""

import ast
import base64
import io
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


def decode_data_uri(data_uri: str) -> Image.Image:
    _, b64 = data_uri.split(",", 1)
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


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


def results_to_zip(results: list[dict]) -> bytes:
    """Packs per-page results into a ZIP: one folder per page with the
    box-overlay image, the model's raw output, and the clean text."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for i, r in enumerate(results, 1):
            folder = f"page_{i:02d}"
            png_buf = io.BytesIO()
            r["overlay"].save(png_buf, format="PNG")
            z.writestr(f"{folder}/overlay.png", png_buf.getvalue())
            z.writestr(f"{folder}/raw.txt", r.get("raw_text", ""))
            z.writestr(f"{folder}/text.txt", r.get("text", ""))
    return buf.getvalue()
