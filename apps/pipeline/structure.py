"""Splits raw OCR output on the model's <|det|> grounding markers into labeled
text blocks, without the box-drawing/image side of apps/ocr/dets.py (this app
never displays the source page, only the extracted structure) — kept local
rather than shared since apps are independent packages by design."""

import re

_DET_RE = re.compile(r"<\|det\|>\s*([A-Za-z_][\w-]*)\s*\[[0-9,\s.\[\]]+\]\s*<\|/det\|>")


def parse_labeled_blocks(raw_text: str) -> list[dict]:
    """Ordered list of {"label", "text"} blocks, e.g. label="header"/"title"/
    "text"/"table"/"footer". Safe to call on partial/still-streaming text
    since only fully-closed markers match."""
    matches = list(_DET_RE.finditer(raw_text))
    blocks = []
    for i, m in enumerate(matches):
        label = m.group(1).strip()
        if label == "image":
            continue
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        text = raw_text[start:end].strip()
        if text:
            blocks.append({"label": label, "text": text})
    return blocks
