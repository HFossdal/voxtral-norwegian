import numpy as np

from voxnor.config import ModelConfig
from voxnor.model import load_model
from voxnor.realtime_stream import layout_from_processor
from voxnor.transcribe import WhisperTranscriber, build_transcriber, load_processor

WHISPER = ModelConfig(
    family="whisper", hf_id="NbAiLab/nb-whisper-large", tiny=True, dtype="float32", language="no"
)


def test_whisper_transcribes_a_batch_with_one_flag_per_clip():
    model, processor = load_model(WHISPER).eval(), load_processor(WHISPER)
    transcriber = build_transcriber(WHISPER, model, processor, max_new_tokens=5)
    assert isinstance(transcriber, WhisperTranscriber)
    rng = np.random.default_rng(0)
    audios = [rng.standard_normal(n).astype(np.float32) * 0.1 for n in (16_000, 24_000)]
    texts = transcriber.transcribe(audios)
    assert len(texts) == 2 and all(isinstance(t, str) for t in texts)
    # A random model rarely ends within 5 tokens, so these are cut off.
    assert transcriber.last_truncated == [True, True]


def test_realtime_processor_uses_the_configured_delay():
    for delay_ms in (480, 960):
        cfg = ModelConfig(
            hf_id="mistralai/Voxtral-Mini-4B-Realtime-2602", transcription_delay_ms=delay_ms
        )
        processor = load_processor(cfg)
        assert processor.num_delay_tokens == delay_ms // 80
        # The training targets read their delay from the processor too.
        assert layout_from_processor(processor).delay == delay_ms // 80
