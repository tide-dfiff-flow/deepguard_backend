"""Detection metrics: EER, AUC, min t-DCF, minDCF, FPR/FNR at fixed threshold.

Ported and lightly generalised from the official ASVspoof 2021 evaluation
package (``2021-main/eval-package/eval_metrics.py``).  The convention here:

* ``spoof_probability`` (the defense output, 0=bonafide .. 1=spoof) is the
  *higher means more spoof* score.
* The ASVspoof ``compute_det_curve``/``compute_eer`` use the opposite
  convention (higher = more bonafide).  We therefore expose two layers:

  - ``compute_det_curve_eer(bonafide_scores, spoof_scores)`` takes scores
    where **higher = more bonafide** (ASVspoof convention), used internally
    for t-DCF compatibility.
  - ``eer_from_spoof_prob`` / ``auc_from_spoof_prob`` / ``min_dcf`` take the
    defense's ``spoof_probability`` directly (higher = more spoof).
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# ASVspoof-convention primitives (higher score = more bonafide)
# --------------------------------------------------------------------------- #
def compute_det_curve(target_scores: np.ndarray,
                      nontarget_scores: np.ndarray):
    """DET curve.  ``target_scores`` = bonafide (positive class).

    Returns ``(frr, far, thresholds)`` where higher threshold => stricter
    bonafide decision.  Ported verbatim from the official implementation.
    """
    target_scores = np.asarray(target_scores, dtype=np.float64)
    nontarget_scores = np.asarray(nontarget_scores, dtype=np.float64)
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size),
                             np.zeros(nontarget_scores.size)))
    indices = np.argsort(all_scores, kind="mergesort")
    labels = labels[indices]
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = (nontarget_scores.size
                            - (np.arange(1, n_scores + 1) - tar_trial_sums))
    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate((np.atleast_1d(1),
                          nontarget_trial_sums / nontarget_scores.size))
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))
    return frr, far, thresholds


def compute_eer(bonafide_scores, spoof_scores):
    """EER using ASVspoof convention (higher = more bonafide).

    Returns ``(eer, threshold)`` where threshold is on the bonafide-score
    scale.  ``eer`` is a fraction in [0, 1].
    """
    frr, far, thresholds = compute_det_curve(bonafide_scores, spoof_scores)
    abs_diffs = np.abs(frr - far)
    min_index = int(np.argmin(abs_diffs))
    eer = float(np.mean((frr[min_index], far[min_index])))
    return eer, float(thresholds[min_index])


# --------------------------------------------------------------------------- #
# Defense-output convention (higher spoof_probability = more spoof)
# --------------------------------------------------------------------------- #
def eer_from_spoof_prob(bonafide_prob: np.ndarray, spoof_prob: np.ndarray):
    """EER for defense ``spoof_probability`` (higher = more spoof).

    Returns ``(eer, threshold_on_spoof_prob)``.
    """
    bonafide_prob = np.asarray(bonafide_prob, dtype=np.float64)
    spoof_prob = np.asarray(spoof_prob, dtype=np.float64)
    # convert to bonafide score = 1 - spoof_prob and reuse ASVspoof routine
    eer, thr_bona = compute_eer(1.0 - bonafide_prob, 1.0 - spoof_prob)
    return eer, float(1.0 - thr_bona)


def auc_from_spoof_prob(bonafide_prob: np.ndarray, spoof_prob: np.ndarray) -> float:
    """ROC-AUC where label 1 = spoof and higher score = more spoof.

    Implemented without sklearn to avoid a hard dependency, using the rank
    statistic (Mann-Whitney U).  Ties handled by average ranks.
    """
    bonafide_prob = np.asarray(bonafide_prob, dtype=np.float64)
    spoof_prob = np.asarray(spoof_prob, dtype=np.float64)
    scores = np.concatenate([spoof_prob, bonafide_prob])
    labels = np.concatenate([np.ones(len(spoof_prob)),
                             np.zeros(len(bonafide_prob))])
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # average ranks (1-based)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # handle ties: average rank per tied group
    sorted_scores = scores[order]
    # find tied groups
    ties_start = []
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ties_start.append((i, j))
        i = j + 1
    for (i, j) in ties_start:
        avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
    sum_ranks_pos = ranks[labels == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def fpr_fnr_at_threshold(bonafide_prob: np.ndarray, spoof_prob: np.ndarray,
                         threshold: float = 0.5):
    """False-reject (bonafide->spoof) and false-accept (spoof->bonafide) at τ.

    With spoof_probability convention:
    * decide "spoof" if prob >= threshold
    * FRR (bonafide->spoof) = fraction of bonafide with prob >= threshold
    * FAR (spoof->bonafide) = fraction of spoof with prob < threshold
    """
    bonafide_prob = np.asarray(bonafide_prob, dtype=np.float64)
    spoof_prob = np.asarray(spoof_prob, dtype=np.float64)
    if bonafide_prob.size == 0:
        frr = float("nan")
    else:
        frr = float(np.mean(bonafide_prob >= threshold))
    if spoof_prob.size == 0:
        far = float("nan")
    else:
        far = float(np.mean(spoof_prob < threshold))
    return {"FRR": frr, "FAR": far, "threshold": threshold}


# --------------------------------------------------------------------------- #
# min t-DCF / minDCF  (standalone-CM mode, ASVspoof 2021 cost model)
# --------------------------------------------------------------------------- #
# Default ASVspoof 2021 cost model
ASVSPOOF2021_COST_MODEL = {
    "Ptar": 0.99, "Pnon": 0.005, "Pspoof": 0.005,
    "Cmiss": 1.0, "Cfa": 10.0, "Cfa_spoof": 50.0,
}

# Default ASV error rates (worst-case / no ASV; for standalone CM scoring we
# assume a near-ideal ASV so that t-DCF is dominated by the CM).  These match
# the ASVspoof 2021 standalone evaluation assumption.
_DEFAULT_ASV = {"Pfa_asv": 0.01, "Pmiss_asv": 0.01,
                "Pfa_spoof_asv": 0.05, "Pmiss_spoof_asv": 0.95}


def compute_tdcf(bonafide_prob: np.ndarray, spoof_prob: np.ndarray,
                 cost_model: dict | None = None,
                 asv_errors: dict | None = None):
    """min t-DCF (normalised) for a standalone CM.

    ``bonafide_prob`` / ``spoof_prob`` are defense spoof_probabilities
    (higher = more spoof).  Internally converted to bonafide scores.

    Returns dict with ``mintDCF``, ``eer``, ``threshold``.
    """
    cost_model = {**ASVSPOOF2021_COST_MODEL, **(cost_model or {})}
    asv = {**_DEFAULT_ASV, **(asv_errors or {})}
    bona_score = 1.0 - np.asarray(bonafide_prob, dtype=np.float64)
    spoof_score = 1.0 - np.asarray(spoof_prob, dtype=np.float64)

    combined = np.concatenate((bona_score, spoof_score))
    if combined.size < 3 or not np.isfinite(combined).all():
        return {"mintDCF": float("nan"), "eer": float("nan"),
                "threshold": float("nan")}

    Pmiss_cm, Pfa_cm, _ = compute_det_curve(bona_score, spoof_score)

    C0 = (cost_model["Ptar"] * cost_model["Cmiss"] * asv["Pmiss_asv"]
          + cost_model["Pnon"] * cost_model["Cfa"] * asv["Pfa_asv"])
    C1 = (cost_model["Ptar"] * cost_model["Cmiss"]
          - (cost_model["Ptar"] * cost_model["Cmiss"] * asv["Pmiss_asv"]
             + cost_model["Pnon"] * cost_model["Cfa"] * asv["Pfa_asv"]))
    C2 = cost_model["Pspoof"] * cost_model["Cfa_spoof"] * asv["Pfa_spoof_asv"]
    if C0 < 0 or C1 < 0 or C2 < 0:
        return {"mintDCF": float("nan"), "eer": float("nan"),
                "threshold": float("nan")}
    tdcf = C0 + C1 * Pmiss_cm + C2 * Pfa_cm
    tdcf_default = C0 + np.minimum(C1, C2)
    tdcf_norm = tdcf / tdcf_default
    mintDCF = float(tdcf_norm[np.argmin(tdcf_norm)])

    abs_diffs = np.abs(Pmiss_cm - Pfa_cm)
    min_index = int(np.argmin(abs_diffs))
    eer = float(np.mean((Pmiss_cm[min_index], Pfa_cm[min_index])))
    return {"mintDCF": mintDCF, "eer": eer, "threshold": float("nan")}


def min_dcf(bonafide_prob: np.ndarray, spoof_prob: np.ndarray,
            p_target: float = 0.01, c_miss: float = 1.0,
            c_fa: float = 1.0) -> float:
    """A simplified minDCF for standalone DeepFake detection.

    With spoof as the positive class, DCF(thr) = p_target*Pmiss(thr)*Cmiss
    + (1-p_target)*Pfa(thr)*Cfa, normalised by min(p_target*Cmiss,
    (1-p_target)*Cfa).  Returns the minimum normalised DCF.
    """
    bona = np.asarray(bonafide_prob, dtype=np.float64)
    spoof = np.asarray(spoof_prob, dtype=np.float64)
    if bona.size == 0 or spoof.size == 0:
        return float("nan")
    scores = np.concatenate([bona, spoof])
    labels = np.concatenate([np.zeros(len(bona)), np.ones(len(spoof))])
    order = np.argsort(scores, kind="mergesort")
    scores_s = scores[order]
    labels_s = labels[order]
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    # iterate candidate thresholds (between consecutive distinct scores)
    cum_pos = np.cumsum(labels_s)        # positives with score <= current
    # for threshold placed just above scores_s[i]: positives below = cum_pos[i],
    # negatives below = (i+1) - cum_pos[i]
    idx = np.arange(len(scores_s))
    pos_below = cum_pos
    neg_below = (idx + 1) - cum_pos
    Pmiss = (n_pos - pos_below) / n_pos  # spoofs predicted bonafide
    Pfa = neg_below / n_neg              # bonafide predicted spoof
    dcf = p_target * c_miss * Pmiss + (1 - p_target) * c_fa * Pfa
    default = min(p_target * c_miss, (1 - p_target) * c_fa)
    if default <= 0:
        return float("nan")
    # also consider thr = +inf (all predicted bonafide): Pmiss=1, Pfa=0
    dcf = np.concatenate([dcf, [p_target * c_miss]])
    return float(np.min(dcf) / default)


# --------------------------------------------------------------------------- #
# Convenience aggregate
# --------------------------------------------------------------------------- #
def all_detection_metrics(bonafide_prob: np.ndarray, spoof_prob: np.ndarray,
                          threshold: float = 0.5) -> dict:
    """Compute the full defense metric bundle for one trial set."""
    eer, eer_thr = eer_from_spoof_prob(bonafide_prob, spoof_prob)
    auc = auc_from_spoof_prob(bonafide_prob, spoof_prob)
    fr_fn = fpr_fnr_at_threshold(bonafide_prob, spoof_prob, threshold)
    tdcf = compute_tdcf(bonafide_prob, spoof_prob)
    mdcf = min_dcf(bonafide_prob, spoof_prob)
    return {
        "EER": eer,
        "EER_threshold": eer_thr,
        "AUC": auc,
        "min_tDCF": tdcf["mintDCF"],
        "minDCF": mdcf,
        "FRR": fr_fn["FRR"],          # bonafide->spoof at τ
        "FAR": fr_fn["FAR"],          # spoof->bonafide at τ
        "threshold": threshold,
        "n_bonafide": int(np.asarray(bonafide_prob).size),
        "n_spoof": int(np.asarray(spoof_prob).size),
    }
