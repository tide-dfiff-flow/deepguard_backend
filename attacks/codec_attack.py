"""A_codec: lossy codec attack (spec §14 A0-1).

``WAV -> MP3/AAC/Opus -> WAV 16 kHz``.  Without ``ffmpeg``/``lameenc`` we
simulate the perceptual lossy codec with a PQMF-style sub-band quantiser that
introduces the same kind of band-limited quantisation noise the real codecs
produce.  When a real encoder is available (pydub+ffmpeg or soundfile with a
codec backend) it can be plugged in via ``real_codec=True``.
"""
from __future__ import annotations

import io
import numpy as np
from scipy.signal import resample_poly, butter, lfilter

from .base import BaseAttack


def _quantize(x: np.ndarray, bits: int) -> np.ndarray:
    """Symmetric mid-tread quantiser."""
    levels = 2 ** bits
    step = 2.0 / levels
    q = np.round(x / step) * step
    return q


class CodecAttack(BaseAttack):
    name = "A_codec"
    params = {
        "codec": "mp3",        # mp3 | aac | opus | auto
        "bitrate_kbps": 64,
        "bits": 8,             # quantiser bits for the simulator
        "n_bands": 32,         # PQMF sub-bands
        "lowpass_hz": 8000,    # band-limit (Opus/AAC narrowband)
        "real_codec": False,   # try real encoder if available
        "seed": 0,
    }

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        params = self.params
        codec = params["codec"]
        if params.get("real_codec", False):
            enc = _try_real_codec(audio, sample_rate, codec, params["bitrate_kbps"])
            if enc is not None:
                return enc

        # ---- simulator path ----
        x = audio.astype(np.float64)
        # 1. optional band-limit (codec low-pass)
        lp = params["lowpass_hz"]
        if lp and lp < sample_rate / 2:
            x = _lowpass(x, lp, sample_rate)

        # 2. PQMF sub-band split -> quantise -> recombine
        bits = max(2, int(params["bits"]) - self._codec_bit_penalty(codec, params["bitrate_kbps"]))
        x = _pqmf_quantize(x, params["n_bands"], bits)

        # 3. round-trip resample to a typical codec internal rate and back
        if codec == "opus":
            internal = 16000
        elif codec == "aac":
            internal = 22050
        else:  # mp3
            internal = 16000 if params["bitrate_kbps"] <= 64 else 22050
        if internal != sample_rate:
            from math import gcd
            g = gcd(int(sample_rate), int(internal))
            x = resample_poly(x, int(internal) // g, int(sample_rate) // g)
            g2 = gcd(int(internal), int(sample_rate))
            x = resample_poly(x, int(sample_rate) // g2, int(internal) // g2)
        return x.astype(np.float32)

    @staticmethod
    def _codec_bit_penalty(codec: str, bitrate: int) -> int:
        """Reduce quantiser bits for lower bitrates."""
        if bitrate <= 32:
            return 3
        if bitrate <= 64:
            return 2
        if bitrate <= 96:
            return 1
        return 0


def _lowpass(x, cutoff, sr, order=6):
    nyq = 0.5 * sr
    b, a = butter(order, cutoff / nyq, btype="low")
    return lfilter(b, a, x)


def _pqmf_quantize(x: np.ndarray, n_bands: int, bits: int) -> np.ndarray:
    """Pseudo-QMF: split into n_bands uniform sub-bands, quantise each,
    sum back.  Introduces band-shaped quantisation noise like real codecs."""
    n = len(x)
    if n == 0:
        return x
    # DCT-IV-like sub-band decomposition via type-III DCT (real, orthonormal-ish)
    from scipy.fft import dct, idct
    # frame the signal to keep sub-band structure local
    frame = 1024
    hop = frame // 2
    window = np.hanning(frame)
    out = np.zeros_like(x)
    norm = np.zeros_like(x)
    i = 0
    while i < n:
        end = min(i + frame, n)
        L = end - i
        seg = x[i:end]
        if L < frame:
            seg = np.pad(seg, (0, frame - L))
        segw = seg * window
        coeffs = dct(segw, type=2, norm="ortho")
        # split coefficients into n_bands groups and quantise with shaped step
        band_size = max(1, len(coeffs) // n_bands)
        q_coeffs = coeffs.copy()
        for b in range(n_bands):
            s = b * band_size
            e = min(len(coeffs), s + band_size)
            if e <= s:
                continue
            # higher bands get coarser quantisation (perceptual shaping)
            band_bits = max(1, bits - (b // max(1, n_bands // 4)))
            step = 2.0 / (2 ** band_bits)
            q_coeffs[s:e] = np.round(coeffs[s:e] / step) * step
        rec = idct(q_coeffs, type=2, norm="ortho")
        out[i:end] += rec[:L] * window[:L]
        norm[i:end] += window[:L] ** 2
        i += hop
    norm[norm == 0] = 1.0
    return out / norm


def _try_real_codec(audio, sr, codec, bitrate_kbps):
    """Attempt a real encode/decode round-trip if pydub+ffmpeg available."""
    try:
        from pydub import AudioSegment
    except Exception:
        return None
    try:
        import array
        pcm = np.clip(audio, -1.0, 1.0)
        pcm16 = (pcm * 32767).astype(np.int16)
        seg = AudioSegment(
            pcm16.tobytes(), frame_rate=sr,
            sample_width=2, channels=1)
        buf = io.BytesIO()
        fmt = {"mp3": "mp3", "aac": "adts", "opus": "opus"}.get(codec, "mp3")
        seg.export(buf, format=fmt, bitrate=f"{bitrate_kbps}k")
        buf.seek(0)
        decoded = AudioSegment.from_file(buf, format=fmt)
        y = np.array(array.array("h", decoded.raw_data))
        y = y.astype(np.float32) / 32768.0
        if decoded.frame_rate != sr:
            from math import gcd
            g = gcd(int(decoded.frame_rate), int(sr))
            y = resample_poly(y, int(sr) // g, int(decoded.frame_rate) // g)
        # match length
        if len(y) < len(audio):
            y = np.pad(y, (0, len(audio) - len(y)))
        return y[:len(audio)]
    except Exception:
        return None
