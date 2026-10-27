"""Transcription through Mistral's API: a reference point, not a baseline.

Mistral's newest transcription models (Voxtral 3) have no open weights, so
they are scored through `POST /v1/audio/transcriptions`. The API key is read
from the `MISTRAL_API_KEY` environment variable and never stored. The API
has no documented temperature setting, so decoding cannot be pinned to
greedy.
"""

from __future__ import annotations

import io
import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import soundfile as sf

from voxnor.data import SAMPLE_RATE

URL = "https://api.mistral.ai/v1/audio/transcriptions"
# Rate limits, overload and dropped connections: worth another try.
RETRY_STATUS = {429, 500, 502, 503, 504}


class MistralApiTranscriber:
    """Sends each clip as a 16 kHz WAV file, `workers` requests at a time."""

    def __init__(
        self,
        model: str,
        language: str | None,
        api_key: str | None = None,
        workers: int = 8,
        attempts: int = 6,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError("Set MISTRAL_API_KEY to use the Mistral API")
        self.model, self.language = model, language
        self.workers, self.attempts = workers, attempts
        self.session = session or requests.Session()
        self.session.headers["Authorization"] = f"Bearer {api_key}"
        self.sleep = sleep

    def transcribe(self, audios: list[np.ndarray]) -> list[str]:
        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(self.transcribe_one, audios))

    def transcribe_one(self, audio: np.ndarray) -> str:
        buffer = io.BytesIO()
        sf.write(buffer, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        data = {"model": self.model}
        if self.language is not None:
            data["language"] = self.language
        for attempt in range(self.attempts):
            files = {"file": ("clip.wav", buffer.getvalue(), "audio/wav")}
            try:
                response = self.session.post(URL, data=data, files=files, timeout=120)
            except (requests.ConnectionError, requests.Timeout):
                if attempt == self.attempts - 1:
                    raise
            else:
                if response.status_code not in RETRY_STATUS or attempt == self.attempts - 1:
                    if response.status_code >= 400:
                        # The body says why (e.g. an unsupported language code).
                        raise requests.HTTPError(
                            f"{response.status_code} from the Mistral API: {response.text[:500]}",
                            response=response,
                        )
                    return response.json()["text"].strip()
            self.sleep(min(2**attempt, 60))
        raise AssertionError("unreachable")
