"""Training-time audio augmentation: speed perturbation and additive noise.

Speed perturbation resamples the clip, so tempo and pitch change together
(Ko et al., 2015). Word timings are scaled by the same factor, so the
stream target still places every word at its spoken time. Noise is white
Gaussian noise at a random signal-to-noise ratio.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

from voxnor.align import WordTiming

SPEED_FACTORS = (0.9, 1.0, 1.1)
NOISE_PROBABILITY = 0.5
NOISE_SNR_DB = (10.0, 30.0)


def change_speed(
    audio: np.ndarray, timings: Sequence[WordTiming], factor: float
) -> tuple[np.ndarray, list[WordTiming]]:
    """Play the clip `factor` times as fast; the word timings move with it."""
    if factor == 1.0:
        return audio, list(timings)
    ratio = Fraction(factor).limit_denominator(100)
    # Slower (factor < 1) means more samples: upsample by the denominator.
    out = resample_poly(audio, ratio.denominator, ratio.numerator).astype(audio.dtype)
    return out, [WordTiming(t.word, t.start / factor, t.end / factor) for t in timings]


def add_noise(audio: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """White Gaussian noise at `snr_db` relative to the clip's mean power."""
    power = float(np.mean(audio.astype(np.float64) ** 2))
    if power == 0.0:
        return audio
    noise = rng.standard_normal(len(audio)) * np.sqrt(power / 10 ** (snr_db / 10))
    return (audio + noise).astype(audio.dtype)


def augment(
    audio: np.ndarray, timings: Sequence[WordTiming], rng: np.random.Generator
) -> tuple[np.ndarray, list[WordTiming]]:
    """A random speed from `SPEED_FACTORS`, then noise with `NOISE_PROBABILITY`."""
    audio, timings = change_speed(audio, timings, float(rng.choice(SPEED_FACTORS)))
    if rng.random() < NOISE_PROBABILITY:
        audio = add_noise(audio, rng.uniform(*NOISE_SNR_DB), rng)
    return audio, timings
