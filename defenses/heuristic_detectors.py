"""D_hidden: lightweight heuristic countermeasures (spec §十).

A small pool of CPU-only, non-learned detectors that exploit classic
DeepFake forensic cues (spectral flatness, high-band energy ratio, phase
discontinuity).  Each returns an uncalibrated raw score from :meth:`raw_score`;
the shared calibration base fits a linear+sigmoid mapping from training
scores so the heuristic detectors are actually discriminative rather than
relying on hand-picked thresholds.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import stft

from .base import BaseDefense


def _stft_mag(x, fs):
    f, _t, Z = stft(x, fs=fs, window="hann", nperseg=512, noverlap=384)
    return f, np.abs(Z)


class _CalibratedHeuristic(BaseDefense):
    """Base for heuristic detectors: fits ``z = (raw - thr) / scale`` and a
    sigmoid mapping so higher raw score => higher spoof probability."""

    params = {"seed": 0}

    def __init__(self, **params):
        super().__init__(**params)
        self._calib = {"thr": 0.0, "scale": 1.0, "direction": 1.0}

    # subclasses implement raw scoring (higher = more spoof-ish by default)
    def raw_score(self, audio, sample_rate) -> float:
        raise NotImplementedError

    def score(self, audio, sample_rate):
        return self.raw_score(audio, sample_rate)

    def train(self, bonafide_samples, spoof_samples, sample_rate=16000):
        b = np.array([self.raw_score(s, sample_rate) for s in bonafide_samples], dtype=np.float64)
        s = np.array([self.raw_score(s, sample_rate) for s in spoof_samples], dtype=np.float64)
        b = b[np.isfinite(b)]
        s = s[np.isfinite(s)]
        if b.size == 0 or s.size == 0:
            self.is_trained = True
            return
        mb, ms = float(np.mean(b)), float(np.mean(s))
        thr = 0.5 * (mb + ms)
        scale = float(0.5 * (np.std(b) + np.std(s)) + 1e-6)
        direction = 1.0 if ms >= mb else -1.0
        self._calib = {"thr": thr, "scale": scale, "direction": direction}
        self.is_trained = True

    def _to_probability(self, raw_score, audio, sample_rate):
        z = (raw_score - self._calib["thr"]) / self._calib["scale"]
        z *= self._calib["direction"]
        return 1.0 / (1.0 + np.exp(-z))


class SpectralFlatnessDefense(_CalibratedHeuristic):
    """Vocoded/spoofed audio tends to higher spectral flatness in the high
    band (vocoder buzz)."""
    name = "D_flat"

    def raw_score(self, audio, sample_rate):
        x = np.asarray(audio, dtype=np.float64)
        if x.size < 512:
            return 0.5
        f, mag = _stft_mag(x, sample_rate)
        hb = (f >= 4000) & (f <= 8000)
        if not hb.any():
            return 0.5
        m = mag[hb] + 1e-12
        geo = np.exp(np.mean(np.log(m), axis=0))
        arith = np.mean(m, axis=0)
        return float(np.mean(geo / arith))


class HighBandEnergyDefense(_CalibratedHeuristic):
    """Vocoder/codec artefacts change the high/low band energy ratio."""
    name = "D_hbe"

    def raw_score(self, audio, sample_rate):
        x = np.asarray(audio, dtype=np.float64)
        if x.size < 512:
            return 0.5
        f, mag = _stft_mag(x, sample_rate)
        e = (mag ** 2).sum(axis=1)
        lo = e[(f >= 0) & (f < 2000)].sum()
        hi = e[(f >= 4000) & (f <= 8000)].sum()
        return float(hi / (lo + 1e-12))


class PhaseDiscontinuityDefense(_CalibratedHeuristic):
    """Spoof concatenation / vocoder frames create phase jumps."""
    name = "D_phase"

    def raw_score(self, audio, sample_rate):
        x = np.asarray(audio, dtype=np.float64)
        if x.size < 1024:
            return 0.5
        f, _t, Z = stft(x, fs=sample_rate, window="hann", nperseg=512, noverlap=384)
        phase = np.angle(Z)
        dphi = np.diff(phase, axis=1)
        dphi = np.mod(dphi + np.pi, 2 * np.pi) - np.pi
        mb = (f >= 500) & (f <= 4000)
        if not mb.any():
            return 0.5
        return float(np.mean(np.abs(dphi[mb])))
