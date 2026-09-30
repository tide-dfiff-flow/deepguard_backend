"""Attack-side metrics (spec §8).

For a given attack ``A`` and defense ``D`` operating on a set of spoof
samples:

* ``ASR(A,D)`` = #(valid_attack & p_s(x') < τ) / N_spoof      (§8.1)
* ``CS(A,D)``  = mean_i max(0, p_s(x_i) - p_s(x_i'))           (§8.2)
* ``valid_attack_rate`` = fraction of outputs passing the fidelity judge

These functions take *already computed* per-sample probabilities and validity
flags so they are cheap and decoupled from the engine.
"""
from __future__ import annotations

import numpy as np


def attack_success_rate(p_spoof_before: np.ndarray,
                        p_spoof_after: np.ndarray,
                        valid: np.ndarray,
                        threshold: float = 0.5) -> float:
    """``ASR(A,D)`` per spec §8.1.

    A spoof sample is a *success* iff the attack was valid *and* the
    post-attack spoof probability dropped below ``threshold``.
    """
    p_before = np.asarray(p_spoof_before, dtype=np.float64)
    p_after = np.asarray(p_spoof_after, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    n = len(p_before)
    if n == 0:
        return float("nan")
    success = valid & (p_after < threshold)
    return float(np.mean(success))


def confidence_suppression(p_spoof_before: np.ndarray,
                           p_spoof_after: np.ndarray) -> float:
    """``CS(A,D)`` per spec §8.2 (mean drop in spoof confidence)."""
    p_before = np.asarray(p_spoof_before, dtype=np.float64)
    p_after = np.asarray(p_spoof_after, dtype=np.float64)
    if len(p_before) == 0:
        return float("nan")
    return float(np.mean(np.maximum(0.0, p_before - p_after)))


def valid_attack_rate(valid: np.ndarray) -> float:
    valid = np.asarray(valid, dtype=bool)
    if valid.size == 0:
        return float("nan")
    return float(np.mean(valid))


def attack_metric_bundle(p_spoof_before: np.ndarray,
                         p_spoof_after: np.ndarray,
                         valid: np.ndarray,
                         threshold: float = 0.5) -> dict:
    return {
        "ASR": attack_success_rate(p_spoof_before, p_spoof_after, valid, threshold),
        "CS": confidence_suppression(p_spoof_before, p_spoof_after),
        "valid_attack_rate": valid_attack_rate(valid),
        "threshold": threshold,
        "n_spoof": int(np.asarray(p_spoof_before).size),
        "mean_p_before": float(np.mean(p_spoof_before)) if len(p_spoof_before) else float("nan"),
        "mean_p_after": float(np.mean(p_spoof_after)) if len(p_spoof_after) else float("nan"),
    }
