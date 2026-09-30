"""LFCC (Linear-frequency Cepstral Coefficients) extraction.

A self-contained numpy/scipy re-implementation that mirrors the ASVspoof 2021
baseline front-end (see ``2021-main/.../LFCC_pipeline.py`` and the torch
``LFCC`` class in ``sandbox/util_frontend.py``) without depending on ``spafe``
or ``librosa``.

Pipeline: pre-emphasis -> framing -> Hamming window -> |FFT|^2 -> linear
triangular filter-bank -> log -> DCT-II -> (optional) delta & delta-delta.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter
from scipy.fft import dct


def _pre_emphasis(sig: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    return lfilter([1.0, -coeff], [1.0], sig)


def _framing(sig: np.ndarray, fs: int, win_len: float = 0.03,
             win_hop: float = 0.015) -> tuple:
    frame_length = int(round(win_len * fs))
    frame_step = int(round(win_hop * fs))
    n = len(sig)
    if n <= frame_length:
        # pad to a single frame
        frames = np.pad(sig, (0, max(0, frame_length - n)))
        return frames.reshape(1, -1), frame_length
    n_frames = 1 + int(np.ceil((n - frame_length) / frame_step))
    pad_len = (n_frames - 1) * frame_step + frame_length
    padded = np.pad(sig, (0, pad_len - n))
    indices = (np.arange(frame_length)[None, :]
               + np.arange(n_frames)[:, None] * frame_step)
    frames = padded[indices]
    return frames, frame_length


def _linear_filterbanks(nfilts: int, nfft: int, fs: int,
                        low_freq: float, high_freq: float,
                        scale: str = "constant") -> np.ndarray:
    """Linear triangular filter-bank, shape (nfilts, nfft//2+1)."""
    f_min = max(0.0, low_freq)
    f_max = min(fs / 2.0, high_freq) if high_freq else fs / 2.0
    n_bins = nfft // 2 + 1
    bin_freqs = np.linspace(0.0, fs / 2.0, n_bins)
    band_edges = np.linspace(f_min, f_max, nfilts + 2)
    fb = np.zeros((nfilts, n_bins), dtype=np.float64)
    for i in range(nfilts):
        left, center, right = band_edges[i], band_edges[i + 1], band_edges[i + 2]
        # rising slope
        l_idx = (bin_freqs >= left) & (bin_freqs <= center)
        if center > left:
            fb[i, l_idx] = (bin_freqs[l_idx] - left) / (center - left)
        # falling slope
        r_idx = (bin_freqs >= center) & (bin_freqs <= right)
        if right > center:
            fb[i, r_idx] = (right - bin_freqs[r_idx]) / (right - center)
    if scale == "constant":
        # normalise each filter to unit area so all bands contribute equally
        energy = fb.sum(axis=1, keepdims=True)
        energy[energy == 0] = 1.0
        fb = fb / energy
    return fb


def _deltas(x: np.ndarray, width: int = 3) -> np.ndarray:
    """Delta features along the time axis (axis=0: frames)."""
    hlen = int(np.floor(width / 2))
    win = list(range(hlen, -hlen - 1, -1))
    # pad along time axis
    pad = np.pad(x, ((hlen, hlen), (0, 0)), mode="edge")
    d = lfilter(win, 1, pad, axis=0)
    return d[hlen * 2:]


def lfcc(sig: np.ndarray, fs: int = 16000, num_ceps: int = 20,
         pre_emph: bool = True, pre_emph_coeff: float = 0.97,
         win_len: float = 0.030, win_hop: float = 0.015,
         win_type: str = "hamming", nfilts: int = 70, nfft: int = 1024,
         low_freq: float = 0.0, high_freq: float = 0.0,
         scale: str = "constant", dct_type: int = 2,
         order_deltas: int = 2, with_energy: bool = False) -> np.ndarray:
    """Compute LFCC features.

    Returns
    -------
    np.ndarray, shape (num_frames, feat_dim)
        ``feat_dim = num_ceps * (1 + order_deltas)`` when deltas are used.
    """
    sig = np.asarray(sig, dtype=np.float64).reshape(-1)
    if sig.size == 0:
        return np.zeros((1, num_ceps * (1 + order_deltas)), dtype=np.float32)

    if pre_emph:
        sig = _pre_emphasis(sig, pre_emph_coeff)

    frames, frame_length = _framing(sig, fs, win_len, win_hop)
    if win_type == "hamming":
        win = np.hamming(frame_length)
    elif win_type == "hann":
        win = np.hanning(frame_length)
    else:
        win = np.ones(frame_length)
    frames = frames * win[None, :]

    # FFT power spectrum
    spec = np.abs(np.fft.rfft(frames, n=nfft)) ** 2

    # filter-bank
    fb = _linear_filterbanks(nfilts, nfft, fs,
                             low_freq if low_freq else 0.0,
                             high_freq if high_freq else fs / 2.0, scale)
    fbank = spec @ fb.T  # (frames, nfilts)
    fbank = np.log10(fbank + 2.220446049250313e-16)

    # DCT
    feats = dct(fbank, type=dct_type, norm="ortho", axis=1)[:, :num_ceps]

    if with_energy:
        power = spec.sum(axis=1) / nfft
        feats[:, 0] = np.log10(power + 2.220446049250313e-16)

    if order_deltas > 0:
        out = [feats]
        for _ in range(order_deltas):
            out.append(_deltas(out[-1]))
        feats = np.concatenate(out, axis=1)
    return feats.astype(np.float32, copy=False)


def extract_lfcc_array(sig: np.ndarray, fs: int = 16000,
                       num_ceps: int = 20, with_delta: bool = True) -> np.ndarray:
    """Convenience wrapper used by both GMM and LCNN defenses.

    Returns ``(num_frames, feat_dim)`` with ``feat_dim = num_ceps * 3`` when
    deltas are enabled (matches the LCNN baseline config).
    """
    return lfcc(sig, fs=fs, num_ceps=num_ceps, order_deltas=2 if with_delta else 0,
                nfilts=max(70, num_ceps), nfft=512,
                low_freq=0.0, high_freq=0.0, with_energy=False)
