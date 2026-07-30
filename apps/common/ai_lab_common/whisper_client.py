import json
from collections.abc import Iterator

import requests


class WhisperTranscriptionError(RuntimeError):
    pass


def transcribe_stream(whisper_api_url: str, filename: str, file_bytes: bytes, content_type: str) -> Iterator[dict]:
    """Streams transcription events from the Whisper server.

    Yields dicts shaped like {"type": "info" | "segment" | "done" | "error", ...},
    matching the NDJSON lines produced by servers/whisper. Raises
    WhisperTranscriptionError on a non-200 response or a mid-stream "error" event.
    """
    files = {"file": (filename, file_bytes, content_type)}
    with requests.post(whisper_api_url, files=files, stream=True) as response:
        if response.status_code != 200:
            raise WhisperTranscriptionError(f"Whisper server returned {response.status_code}: {response.text}")

        for line in response.iter_lines():
            if not line:
                continue
            event = json.loads(line)
            if event["type"] == "error":
                raise WhisperTranscriptionError(event["message"])
            yield event
