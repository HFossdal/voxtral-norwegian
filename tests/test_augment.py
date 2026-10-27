import numpy as np
import pytest

from voxnor.align import WordTiming
from voxnor.augment import add_noise, augment, change_speed
from voxnor.data import SAMPLE_RATE

WORDS = [WordTiming("hei", 0.1, 0.4), WordTiming("på", 0.5, 0.7)]


def tone(hz=200.0, seconds=1.0):
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (0.5 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def peak_hz(audio):
    spectrum = np.abs(np.fft.rfft(audio))
    return np.fft.rfftfreq(len(audio), 1 / SAMPLE_RATE)[spectrum.argmax()]


@pytest.mark.parametrize("factor", [0.9, 1.1])
def test_speed_change_scales_length_pitch_and_timings(factor):
    audio = tone()
    out, timings = change_speed(audio, WORDS, factor)
    assert len(out) == pytest.approx(len(audio) / factor, abs=1)
    assert out.dtype == audio.dtype
    assert peak_hz(out) == pytest.approx(200 * factor, rel=0.01)
    assert timings[0].start == pytest.approx(0.1 / factor)
    assert timings[1].end == pytest.approx(0.7 / factor)
    assert [t.word for t in timings] == ["hei", "på"]


def test_speed_one_is_a_no_op():
    audio = tone()
    out, timings = change_speed(audio, WORDS, 1.0)
    assert out is audio and timings == WORDS


def test_noise_has_the_requested_snr():
    audio = tone(seconds=4.0)
    noisy = add_noise(audio, 20.0, np.random.default_rng(0))
    noise = noisy.astype(np.float64) - audio
    snr = 10 * np.log10(np.mean(audio.astype(np.float64) ** 2) / np.mean(noise**2))
    assert snr == pytest.approx(20.0, abs=0.2)
    silent = np.zeros(100, dtype=np.float32)
    assert add_noise(silent, 20.0, np.random.default_rng(0)) is silent


def test_augment_is_reproducible_from_the_seed():
    audio = tone()
    runs = [augment(audio, WORDS, np.random.default_rng([0, 7, 0])) for _ in range(2)]
    assert np.array_equal(runs[0][0], runs[1][0]) and runs[0][1] == runs[1][1]
    outputs = {len(augment(audio, WORDS, np.random.default_rng(s))[0]) for s in range(30)}
    assert len(outputs) == 3  # all three speeds occur
