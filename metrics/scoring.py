"""Final scoreboard scoring (spec §11.4 attack, §十二 defense)."""
from __future__ import annotations

import numpy as np


def quality_coefficient(fidelity: dict, validity_thresholds: dict | None = None) -> float:
    """``Q(a) in [0, 1]`` from fidelity bundle (spec §11.3).

    Combines STOI, SI-SDR, content consistency, clipping and duration
    consistency into a single quality coefficient.  Illegal samples (failed
    validity) get Q=0 upstream.
    """
    thr = {
        "STOI_min": 0.45,
        "SI_SDR_min": -5.0,
        "duration_min": 0.95,
        "duration_max": 1.05,
        "clipping_max": 0.1,
    }
    if validity_thresholds:
        thr.update(validity_thresholds)

    stoi = fidelity.get("STOI", float("nan"))
    sdr = fidelity.get("SI_SDR", float("nan"))
    dur = fidelity.get("duration_ratio", 1.0)
    clip = fidelity.get("clipping_ratio_est", 0.0)
    cer = fidelity.get("CER", float("nan"))
    proxy = fidelity.get("content_proxy", float("nan"))

    # normalise each axis to [0,1]
    stoi_q = float(np.clip((stoi - thr["STOI_min"]) / (1.0 - thr["STOI_min"]), 0.0, 1.0))
    sdr_q = float(np.clip((sdr - thr["SI_SDR_min"]) / (20.0 - thr["SI_SDR_min"]), 0.0, 1.0))
    dur_q = 1.0 if thr["duration_min"] <= dur <= thr["duration_max"] else 0.0
    clip_q = float(np.clip(1.0 - clip / max(thr["clipping_max"], 1e-6), 0.0, 1.0))
    if np.isnan(cer):
        # no ASR -> use STOI proxy (fallback to stoi_q if proxy also missing)
        if np.isnan(proxy):
            content_q = stoi_q
        else:
            content_q = float(np.clip(proxy, 0.0, 1.0))
    else:
        content_q = float(np.clip(1.0 - cer, 0.0, 1.0))

    # geometric-ish blend so a catastrophic axis kills the score
    q = (0.35 * stoi_q + 0.25 * sdr_q + 0.25 * content_q
         + 0.10 * clip_q + 0.05 * dur_q)
    return float(np.clip(q, 0.0, 1.0))


def core_asr(asr_fixed_pool: float, asr_hidden: float,
             w_fixed: float = 0.7, w_hidden: float = 0.3) -> float:
    """``CoreASR(a) = 0.7 ASR(a,D0) + 0.3 ASR(a,Dq)`` (spec §11.2)."""
    if np.isnan(asr_fixed_pool) and np.isnan(asr_hidden):
        return float("nan")
    a = 0.0 if np.isnan(asr_fixed_pool) else asr_fixed_pool
    b = 0.0 if np.isnan(asr_hidden) else asr_hidden
    return float(w_fixed * a + w_hidden * b)


def attack_score(core_asr_val: float, q: float) -> float:
    """``AttackScore = 100 * CoreASR * Q`` (spec §11.4)."""
    if np.isnan(core_asr_val):
        return 0.0
    return float(100.0 * core_asr_val * float(np.clip(q, 0.0, 1.0)))


def defense_score(eer_robust: float, eer_clean: float,
                  auc_robust: float, efficiency: float = 1.0) -> float:
    """``DefenseScore = 100 * [0.5 RobustScore + 0.25 CleanScore
    + 0.15 RobustAUC + 0.10 Efficiency]`` (spec §十二)."""
    robust_score = 1.0 - eer_robust
    clean_score = 1.0 - eer_clean
    auc = float(np.clip(auc_robust if not np.isnan(auc_robust) else 0.5, 0.0, 1.0))
    eff = float(np.clip(efficiency, 0.0, 1.0))
    return float(100.0 * (0.50 * robust_score + 0.25 * clean_score
                          + 0.15 * auc + 0.10 * eff))


def efficiency(model_size_mb: float, avg_inference_ms: float,
               peak_mem_mb: float, ref_size: float = 100.0,
               ref_time: float = 200.0, ref_mem: float = 1024.0) -> float:
    """Heuristic efficiency in [0,1] from size/time/mem (lower is better)."""
    s = float(np.clip(ref_size / (model_size_mb + 1e-6), 0.0, 1.0))
    t = float(np.clip(ref_time / (avg_inference_ms + 1e-6), 0.0, 1.0))
    m = float(np.clip(ref_mem / (peak_mem_mb + 1e-6), 0.0, 1.0))
    return float(np.clip(0.4 * t + 0.4 * s + 0.2 * m, 0.0, 1.0))
