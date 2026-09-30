"""Fidelity / speech-quality metrics (spec §6.2, §6.3).

Self-contained implementations of SI-SDR and STOI that only need numpy/scipy,
plus optional use of ``pystoi`` / ``pesq`` when they are installed.  Also
computes cheap signal-level descriptors: RMS, clipping ratio, duration ratio,
silence ratio.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly, stft, istft


# --------------------------------------------------------------------------- #
# Optional deps
# --------------------------------------------------------------------------- #
def _try_pystoi():
    try:
        from pystoi import stoi as _stoi  # noqa: WPS433
        return _stoi
    except Exception:
        return None


def _try_pesq():
    try:
        from pesq import pesq as _pesq  # noqa: WPS433
        return _pesq
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# SI-SDR  (Le Roux et al., 2019)
# --------------------------------------------------------------------------- #
def si_sdr(ref: np.ndarray, est: np.ndarray) -> float:
    """Scale-invariant SDR between reference and estimate (dB)."""
    ref = np.asarray(ref, dtype=np.float64).reshape(-1)
    est = np.asarray(est, dtype=np.float64).reshape(-1)
    n = min(len(ref), len(est))
    ref, est = ref[:n], est[:n]
    eps = 1e-8
    # zero-mean
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    alpha = float(np.dot(est, ref) / (np.dot(ref, ref) + eps))
    target = alpha * ref
    noise = est - target
    val = float(10.0 * np.log10(np.dot(target, target)
                                  / (np.dot(noise, noise) + eps) + eps))
    return val


# --------------------------------------------------------------------------- #
# STOI  (self-contained, narrow-band implementation)
# --------------------------------------------------------------------------- #
def stoi(ref: np.ndarray, est: np.ndarray, fs: int = 16000) -> float:
    """Short-Time Objective Intelligibility measure.

    Uses ``pystoi`` if available; otherwise falls back to a self-contained
    narrow-band correlation implementation following the STOI reference
    (Taal et al., 2011).  Returns a value in roughly [0, 1].
    """
    ref = np.asarray(ref, dtype=np.float64).reshape(-1)
    est = np.asarray(est, dtype=np.float64).reshape(-1)
    n = min(len(ref), len(est))
    ref, est = ref[:n], est[:n]
    pystoi_fn = _try_pystoi()
    if pystoi_fn is not None:
        try:
            return float(pystoi_fn(ref, est, fs, extended=False))
        except Exception:
            pass
    return _stoi_numpy(ref, est, fs)


def _stoi_numpy(ref: np.ndarray, est: np.ndarray, fs: int) -> float:
    """Narrow-band STOI approximation via short-time correlation."""
    # STOI parameters
    Nfft = 512
    frame_shift = int(0.01 * fs)        # 10 ms
    frame_len = int(0.03 * fs)          # 30 ms analysis (overlapped)
    n_bands = 15
    j_sample = int(0.26 * fs / frame_shift)  # ~26 ms correlation window in frames

    f, t, Zref = stft(ref, fs=fs, window="hann", nperseg=Nfft, noverlap=Nfft - frame_shift)
    _, _, Zest = stft(est, fs=fs, window="hann", nperseg=Nfft, noverlap=Nfft - frame_shift)
    # align frame count
    n_frames = min(Zref.shape[1], Zest.shape[1])
    Zref = Zref[:, :n_frames]
    Zest = Zest[:, :n_frames]

    # one-third octave band aggregation (log-spaced on [150, 3800] Hz)
    band_edges = _third_octave_edges(150.0, min(3800.0, fs / 2.0), n_bands)
    band_idx = []
    for k in range(len(band_edges) - 1):
        lo = np.searchsorted(f, band_edges[k])
        hi = np.searchsorted(f, band_edges[k + 1])
        if hi <= lo:
            hi = lo + 1
        band_idx.append((lo, min(hi, len(f))))
    band_idx = [(lo, hi) for lo, hi in band_idx if hi > lo]

    X_ref = np.zeros((len(band_idx), n_frames))
    X_est = np.zeros((len(band_idx), n_frames))
    for bi, (lo, hi) in enumerate(band_idx):
        X_ref[bi] = np.sqrt(np.sum(np.abs(Zref[lo:hi]) ** 2, axis=0))
        X_est[bi] = np.sqrt(np.sum(np.abs(Zest[lo:hi]) ** 2, axis=0))

    # normalise by 15-frame sliding window std (clipping)
    win = 15
    d = np.zeros((len(band_idx), max(0, n_frames - j_sample + 1)))
    for bi in range(len(band_idx)):
        for m in range(d.shape[1]):
            seg_ref = X_ref[bi, m:m + j_sample]
            seg_est = X_est[bi, m:m + j_sample]
            sigma = np.sqrt(np.mean(seg_ref ** 2)) + 1e-9
            seg_ref_n = seg_ref / sigma
            seg_est_n = np.clip(seg_est / sigma, -1e3, 1e3)
            seg_est_n = np.clip(seg_est_n, -np.sqrt(10.0), np.sqrt(10.0))
            num = np.sum(seg_ref_n * seg_est_n)
            den = (np.linalg.norm(seg_ref_n) * np.linalg.norm(seg_est_n) + 1e-12)
            d[bi, m] = num / den
    if d.size == 0:
        return float("nan")
    return float(np.clip(np.mean(d), -1.0, 1.0))


def _third_octave_edges(fmin: float, fmax: float, n: int) -> np.ndarray:
    ratio = 2.0 ** (1.0 / 3.0)
    edges = [fmin]
    f = fmin
    while f < fmax and len(edges) <= n + 1:
        f *= ratio
        edges.append(min(f, fmax))
    return np.array(edges[:n + 1])


# --------------------------------------------------------------------------- #
# PESQ (optional)
# --------------------------------------------------------------------------- #
def pesq_score(ref: np.ndarray, est: np.ndarray, fs: int = 16000):
    """Return PESQ score or ``None`` if ``pesq`` is unavailable / fails."""
    fn = _try_pesq()
    if fn is None:
        return None
    ref = np.asarray(ref, dtype=np.float64).reshape(-1)
    est = np.asarray(est, dtype=np.float64).reshape(-1)
    n = min(len(ref), len(est))
    ref, est = ref[:n], est[:n]
    try:
        if fs != 8000 and fs != 16000:
            # resample to 16k
            from math import gcd
            g = gcd(int(fs), 16000)
            est = resample_poly(est, 16000 // g, int(fs) // g)
            ref = resample_poly(ref, 16000 // g, int(fs) // g)
            fs = 16000
        return float(fn(fs, ref, est, "wb"))
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Cheap signal descriptors
# --------------------------------------------------------------------------- #
def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x ** 2) + 1e-12)) if x.size else 0.0


def clipping_ratio(x: np.ndarray, thr: float = 0.99) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    return float(np.mean(np.abs(x) >= thr))


def silence_ratio(x: np.ndarray, fs: int = 16000,
                  frame_len: float = 0.03, rms_thr: float = 0.01) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    fl = int(frame_len * fs)
    n_frames = max(1, len(x) // fl)
    frames = x[:n_frames * fl].reshape(n_frames, fl)
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
    return float(np.mean(frame_rms < rms_thr))


def duration_ratio(ref: np.ndarray, est: np.ndarray) -> float:
    """``T(x')/T(x)`` per spec §6.1."""
    r = len(np.asarray(ref))
    e = len(np.asarray(est))
    if r == 0:
        return 0.0
    return float(e / r)


# --------------------------------------------------------------------------- #
# Aggregate fidelity bundle
# --------------------------------------------------------------------------- #
def fidelity_bundle(ref: np.ndarray, est: np.ndarray, fs: int = 16000) -> dict:
    """Compute all fidelity metrics for one (original, attacked) pair."""
    ref = np.asarray(ref, dtype=np.float32).reshape(-1)
    est = np.asarray(est, dtype=np.float32).reshape(-1)
    sdr = si_sdr(ref, est)
    st = stoi(ref, est, fs)
    pq = pesq_score(ref, est, fs)
    out = {
        "SI_SDR": sdr,
        "STOI": st,
        "PESQ": pq if pq is not None else float("nan"),
        "PESQ_available": pq is not None,
        "RMS_ref": rms(ref),
        "RMS_est": rms(est),
        "RMS_ratio": float(rms(est) / (rms(ref) + 1e-12)),
        "clipping_ratio_ref": clipping_ratio(ref),
        "clipping_ratio_est": clipping_ratio(est),
        "silence_ratio_ref": silence_ratio(ref, fs),
        "silence_ratio_est": silence_ratio(est, fs),
        "duration_ratio": duration_ratio(ref, est),
    }
    return out
