"""Audio I/O utilities for DeepGuard-Audio backend.

Provides loading, saving, resampling and format validation that work with the
guaranteed-available packages (numpy / scipy).  ``soundfile`` and ``librosa``
are used opportunistically when present so that more container formats (flac,
mp3, ...) can be read, but the core path only needs ``scipy.io.wavfile``.
"""
from __future__ import annotations

import os
import wave
import struct
from typing import Tuple

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

TARGET_SR = 16000


# --------------------------------------------------------------------------- #
# Optional dependency probes
# --------------------------------------------------------------------------- #
def _try_import_soundfile():
    try:
        import soundfile as sf  # noqa: WPS433
        return sf
    except Exception:  # pragma: no cover - optional dep
        return None


_SF = _try_import_soundfile()


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_audio(path: str, target_sr: int = TARGET_SR) -> Tuple[np.ndarray, int]:
    """Load an audio file as ``float32`` mono in ``[-1, 1]``.

    Returns ``(audio, sample_rate)``.  Resamples to ``target_sr`` if needed.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    if _SF is not None:
        try:
            audio, sr = _SF.read(path, dtype="float32", always_2d=False)
        except Exception:
            audio, sr = _wav_read_fallback(path)
    else:
        audio, sr = _wav_read_fallback(path)

    audio = _to_mono_float32(audio)
    if sr != target_sr:
        audio = resample_audio(audio, sr, target_sr)
        sr = target_sr
    return audio.astype(np.float32, copy=False), int(sr)


def _wav_read_fallback(path: str) -> Tuple[np.ndarray, int]:
    """Pure-stdlib/scipy wav reader returning float32 in [-1, 1]."""
    sr, data = wavfile.read(path)
    if data.dtype == np.float32 or data.dtype == np.float64:
        audio = data.astype(np.float32)
    elif data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        audio = (data.astype(np.float32) - 128.0) / 128.0
    else:
        audio = data.astype(np.float32)
    return audio, int(sr)


def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        # (frames, channels) -> mean across channels
        audio = audio.mean(axis=tuple(range(1, audio.ndim)))
    audio = np.clip(audio, -1.0, 1.0)
    return audio.astype(np.float32, copy=False)


# --------------------------------------------------------------------------- #
# Saving
# --------------------------------------------------------------------------- #
def save_audio(path: str, audio: np.ndarray, sample_rate: int = TARGET_SR) -> None:
    """Save ``float32`` mono audio to a wav file (int16 PCM)."""
    audio = _to_mono_float32(audio)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    int_data = np.round(audio * 32767.0).astype(np.int16)
    wavfile.write(path, int(sample_rate), int_data)


# --------------------------------------------------------------------------- #
# Resampling
# --------------------------------------------------------------------------- #
def resample_audio(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Polyphase resampling that avoids scipy.signal dependency edge cases."""
    if sr_in == sr_out:
        return audio.astype(np.float32, copy=False)
    # reduce by gcd to keep filter order small
    from math import gcd
    g = gcd(int(sr_in), int(sr_out))
    up = int(sr_out) // g
    down = int(sr_in) // g
    return resample_poly(audio, up, down).astype(np.float32)


# --------------------------------------------------------------------------- #
# Format validation / normalisation (spec §4, §6.1)
# --------------------------------------------------------------------------- #
def check_format(audio: np.ndarray, sample_rate: int = TARGET_SR) -> dict:
    """Return a dict of basic format sanity checks (spec §6.1)."""
    audio = np.asarray(audio)
    info = {
        "is_finite": bool(np.isfinite(audio).all()),
        "has_nan": bool(np.isnan(audio).any()),
        "has_inf": bool(np.isinf(audio).any()),
        "is_mono": audio.ndim == 1,
        "sample_rate": int(sample_rate),
        "duration": float(len(audio) / sample_rate) if sample_rate else 0.0,
        "dtype_ok": str(audio.dtype),
        "peak": float(np.max(np.abs(audio))) if audio.size else 0.0,
        "min": float(np.min(audio)) if audio.size else 0.0,
        "max": float(np.max(audio)) if audio.size else 0.0,
    }
    return info


def normalize_audio(audio: np.ndarray, target_sr: int = TARGET_SR,
                    peak_protect: bool = True) -> Tuple[np.ndarray, int]:
    """Apply the referee pre-check pipeline (spec §4): mono, float32, peak
    clip protect, resample to 16 kHz."""
    audio = _to_mono_float32(audio)
    if peak_protect:
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1.0:
            audio = audio / (peak + 1e-12)
    return audio.astype(np.float32, copy=False), int(target_sr)
