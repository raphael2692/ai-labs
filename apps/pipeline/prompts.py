"""Persists user-defined prompt presets to a local JSON file, kept alongside
app.py rather than in ai_lab_common since it's pipeline-specific state, not a
shared server client."""

import json
from pathlib import Path

PROMPTS_PATH = Path(__file__).parent / "prompts.json"

DEFAULT_REVISION_PROMPT = "Content Revision (light copy-edit)"
DEFAULT_FORMAT_PROMPT = "Clean Markdown Notes"

DEFAULT_PROMPTS = {
    DEFAULT_REVISION_PROMPT: (
        "You review raw extracted document/transcript text and lightly revise its content before it's "
        "formatted into Markdown by a later step.\n\n"
        "Rules:\n"
        "- Output plain revised text, NOT Markdown — no headings, no bullet formatting. That's the next "
        "step's job.\n"
        "- Fix obvious transcription/OCR artifacts (broken words, stray characters, misplaced line breaks, "
        "mis-transcribed terms) and tighten awkward phrasing.\n"
        "- Remove filler and duplicated content, but preserve every fact, name, number, and instruction "
        "faithfully — never invent or drop information.\n"
        "- Keep the original structure/order of ideas; you're editing for clarity, not reorganizing.\n"
    ),
    DEFAULT_FORMAT_PROMPT: (
        "You convert revised document/transcript text into clean, well-structured Markdown.\n\n"
        "Rules:\n"
        "- Output ONLY Markdown. No commentary before or after it, no code fences wrapping the whole thing.\n"
        "- Start with a single H1 title that summarizes the content.\n"
        "- Use H2 sections to break up major topics, bullet lists for enumerations, and bold for key terms.\n"
        "- Preserve the source's factual content faithfully; never invent information that isn't present.\n"
    ),
}


def load_prompts() -> dict[str, str]:
    if PROMPTS_PATH.exists():
        try:
            data = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data:
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return dict(DEFAULT_PROMPTS)


def save_prompts(prompts: dict[str, str]) -> None:
    PROMPTS_PATH.write_text(json.dumps(prompts, indent=2, ensure_ascii=False), encoding="utf-8")
