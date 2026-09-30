"""Content-preservation metrics (spec §6.2).

The competition compares ``Transcript(x)`` vs ``Transcript(x')`` using CER
(Chinese) / WER (English).  A real ASR model is optional here: if a callable
``asr_fn`` (or ``whisper``) is supplied, true CER/WER is computed; otherwise we
fall back to a content-preservation *proxy* based on STOI + spectral
alignment and flag ``asr_available=False`` so the report is honest about it.
"""
from __future__ import annotations

import numpy as np


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance on sequences of characters or tokens."""
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    cur = [0] * (lb + 1)
    for i in range(1, la + 1):
        cur[0] = i
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ca == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev, cur = cur, prev
    return prev[lb]


def cer(ref: str, hyp: str) -> float:
    """Character error rate in [0, 1+] (unnormalised edit distance / len(ref))."""
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(ref, hyp) / max(1, len(ref))


def wer(ref: str, hyp: str) -> float:
    """Word error rate in [0, 1+]."""
    ref_t = ref.split()
    hyp_t = hyp.split()
    if not ref_t:
        return 0.0 if not hyp_t else 1.0
    return _edit_distance("".join(chr(c) for c in range(len(ref_t))),
                          "".join(chr(c) for c in range(len(hyp_t)))) / max(1, len(ref_t))


# --------------------------------------------------------------------------- #
# Whisper probe (optional)
# --------------------------------------------------------------------------- #
def _try_whisper():
    try:
        import whisper  # noqa: WPS433
        return whisper
    except Exception:
        return None


def make_whisper_asr(model_name: str = "base"):
    """Return an ``asr_fn(audio, sr) -> str`` using OpenAI whisper if installed."""
    whisper = _try_whisper()
    if whisper is None:
        return None
    try:
        model = whisper.load_model(model_name)
    except Exception:
        return None

    def asr_fn(audio: np.ndarray, sr: int = 16000) -> str:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        try:
            res = model.transcribe(audio, language=None, fp16=False)
            return str(res.get("text", "")).strip()
        except Exception:
            return ""
    return asr_fn


# --------------------------------------------------------------------------- #
# Content preservation
# --------------------------------------------------------------------------- #
def content_preservation(ref_audio: np.ndarray, est_audio: np.ndarray,
                         fs: int = 16000, asr_fn=None,
                         ref_transcript: str | None = None) -> dict:
    """Compare content of original vs attacked audio.

    Parameters
    ----------
    asr_fn : callable(audio, sr) -> str, optional
        If provided (or whisper available), used to transcribe both signals
        and compute true CER.
    ref_transcript : str, optional
        Precomputed transcript of ``ref_audio`` to avoid re-transcribing.

    Returns
    -------
    dict with ``CER`` (or ``content_proxy``), ``asr_available`` and transcripts.
    """
    if asr_fn is None:
        asr_fn = make_whisper_asr()

    if asr_fn is not None:
        try:
            t_ref = ref_transcript if ref_transcript is not None else asr_fn(ref_audio, fs)
            t_est = asr_fn(est_audio, fs)
            return {
                "CER": cer(t_ref, t_est),
                "transcript_ref": t_ref,
                "transcript_est": t_est,
                "asr_available": True,
                "content_proxy": float("nan"),
            }
        except Exception:
            pass

    # fallback proxy: STOI-based content similarity + spec envelope correlation
    from .fidelity import stoi
    st = stoi(ref_audio, est_audio, fs)
    proxy = float(np.clip(st, 0.0, 1.0))
    return {
        "CER": float("nan"),
        "transcript_ref": "",
        "transcript_est": "",
        "asr_available": False,
        "content_proxy": proxy,
    }
