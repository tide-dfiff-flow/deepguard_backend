"""Referee pre-check pipeline (spec §4) and attack legality judge (spec §6)."""
from __future__ import annotations

import numpy as np

from ..audio_io import normalize_audio, check_format


def precheck(audio: np.ndarray, sample_rate: int = 16000):
    """Spec §4 referee pipeline: format check, peak-clip protect, resample to
    16 kHz.  Returns ``(audio, sample_rate, format_info)``."""
    audio, sr = normalize_audio(audio, target_sr=sample_rate, peak_protect=True)
    info = check_format(audio, sr)
    return audio, sr, info


class FidelityJudge:
    """Decide ``valid_attack`` per spec §6.

    Combines hard format constraints (§6.1) with content/quality thresholds
    (§6.2, §6.3, §6.4).  All thresholds are configurable; defaults follow the
    spec's recommended ranges.
    """

    def __init__(self,
                 duration_min: float = 0.95, duration_max: float = 1.05,
                 stoi_min: float = 0.45, si_sdr_min: float = -5.0,
                 cer_max: float = 0.30, clipping_max: float = 0.10,
                 silence_max: float = 0.80,
                 require_asr: bool = False):
        self.duration_min = duration_min
        self.duration_max = duration_max
        self.stoi_min = stoi_min
        self.si_sdr_min = si_sdr_min
        self.cer_max = cer_max
        self.clipping_max = clipping_max
        self.silence_max = silence_max
        self.require_asr = require_asr

    def judge(self, ref_audio, est_audio, sample_rate=16000,
              fidelity: dict | None = None,
              content: dict | None = None) -> dict:
        """Return ``{valid, reasons, fidelity, content}`` for one sample."""
        if fidelity is None:
            from ..metrics.fidelity import fidelity_bundle
            fidelity = fidelity_bundle(ref_audio, est_audio, sample_rate)
        if content is None:
            from ..metrics.content import content_preservation
            content = content_preservation(ref_audio, est_audio, sample_rate)

        reasons = []
        ref_audio = np.asarray(ref_audio)
        est_audio = np.asarray(est_audio)

        # §6.1 format constraints
        if not np.isfinite(est_audio).all():
            reasons.append("nan_or_inf")
        if est_audio.size == 0:
            reasons.append("empty_output")
        dur = fidelity.get("duration_ratio", 1.0)
        if not (self.duration_min <= dur <= self.duration_max):
            reasons.append(f"duration_ratio_{dur:.2f}")
        if fidelity.get("clipping_ratio_est", 0.0) > self.clipping_max:
            reasons.append("severe_clipping")
        if fidelity.get("silence_ratio_est", 0.0) > self.silence_max:
            reasons.append("excessive_silence")

        # §6.3 signal quality
        stoi = fidelity.get("STOI", 1.0)
        if np.isfinite(stoi) and stoi < self.stoi_min:
            reasons.append(f"stoi_low_{stoi:.2f}")
        sdr = fidelity.get("SI_SDR", 100.0)
        if np.isfinite(sdr) and sdr < self.si_sdr_min:
            reasons.append(f"sdr_low_{sdr:.2f}")

        # §6.2 content preservation
        if content.get("asr_available", False):
            cer = content.get("CER", 0.0)
            if np.isfinite(cer) and cer > self.cer_max:
                reasons.append(f"cer_high_{cer:.2f}")
        elif self.require_asr:
            reasons.append("asr_unavailable")

        valid = len(reasons) == 0
        return {"valid": valid, "reasons": reasons,
                "fidelity": fidelity, "content": content}
