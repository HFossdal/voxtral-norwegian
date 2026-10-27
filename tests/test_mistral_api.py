import io

import numpy as np
import pytest
import requests
import soundfile as sf

from voxnor.data import SAMPLE_RATE
from voxnor.mistral_api import URL, MistralApiTranscriber
from voxnor.report import describe


class FakeResponse:
    def __init__(self, status: int, text: str = ""):
        self.status_code, self.text = status, text

    def json(self):
        return {"text": self.text, "language": None}


class FakeSession:
    """Answers each request with the next of `responses`; records what was sent."""

    def __init__(self, responses):
        self.responses, self.calls, self.headers = list(responses), [], {}

    def post(self, url, data, files, timeout):
        self.calls.append((url, data, files))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def transcriber(responses, language="no", **kwargs):
    session = FakeSession(responses)
    t = MistralApiTranscriber(
        "voxtral-mini-transcribe-3",
        language,
        api_key="test-key",
        workers=1,
        session=session,
        sleep=lambda s: None,
        **kwargs,
    )
    return t, session


def test_sends_a_wav_with_model_and_language():
    t, session = transcriber([FakeResponse(200, " Hei på deg. ")])
    audio = np.linspace(-0.5, 0.5, SAMPLE_RATE, dtype=np.float32)
    assert t.transcribe([audio]) == ["Hei på deg."]
    [(url, data, files)] = session.calls
    assert url == URL
    assert data == {"model": "voxtral-mini-transcribe-3", "language": "no"}
    assert session.headers["Authorization"] == "Bearer test-key"
    _, payload, mime = files["file"]
    sent, rate = sf.read(io.BytesIO(payload))
    assert (rate, len(sent), mime) == (SAMPLE_RATE, SAMPLE_RATE, "audio/wav")


def test_auto_language_sends_no_language_field():
    t, session = transcriber([FakeResponse(200, "x")], language=None)
    t.transcribe([np.zeros(1600, dtype=np.float32)])
    assert "language" not in session.calls[0][1]


def test_retries_rate_limits_and_dropped_connections():
    responses = [FakeResponse(429), requests.ConnectionError(), FakeResponse(200, "ok")]
    t, session = transcriber(responses)
    assert t.transcribe([np.zeros(1600, dtype=np.float32)]) == ["ok"]
    assert len(session.calls) == 3


def test_gives_up_after_the_last_attempt_and_on_client_errors():
    t, _ = transcriber([FakeResponse(503)] * 3, attempts=3)
    with pytest.raises(requests.HTTPError):
        t.transcribe([np.zeros(1600, dtype=np.float32)])
    t, session = transcriber([FakeResponse(400, '{"message": "invalid language"}')])
    with pytest.raises(requests.HTTPError, match=r"400 from the Mistral API: .*invalid language"):
        t.transcribe([np.zeros(1600, dtype=np.float32)])
    assert len(session.calls) == 1


def test_needs_an_api_key(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="MISTRAL_API_KEY"):
        MistralApiTranscriber("voxtral-mini-transcribe-3", None)


def test_report_names_the_api_system():
    model = {"family": "mistral_api", "hf_id": "voxtral-mini-transcribe-3", "tiny": False}
    assert describe({"model": model | {"language": "no"}}) == (
        "voxtral-mini-transcribe-3 (API, language: no)"
    )
