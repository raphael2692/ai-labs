import json
from collections.abc import Iterator

import requests


class OcrError(RuntimeError):
    pass


def ocr_stream(
    ocr_api_url: str, filename: str, file_bytes: bytes, content_type: str, image_mode: str = "gundam"
) -> Iterator[dict]:
    """Streams OCR parsing events from the OCR server.

    Yields dicts shaped like {"type": "info" | "page_start" | "chunk" | "page" | "done" | "error", ...},
    matching the NDJSON lines produced by servers/ocr. Raises OcrError on a
    non-200 response or a mid-stream "error" event.
    """
    files = {"file": (filename, file_bytes, content_type)}
    with requests.post(ocr_api_url, files=files, data={"image_mode": image_mode}, stream=True) as response:
        if response.status_code != 200:
            raise OcrError(f"OCR server returned {response.status_code}: {response.text}")

        for line in response.iter_lines():
            if not line:
                continue
            event = json.loads(line)
            if event["type"] == "error":
                raise OcrError(event["message"])
            yield event
