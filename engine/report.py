"""Report generation: CSV + JSON + attack/defense heatmaps (spec §17)."""
from __future__ import annotations

import os
import json
import csv
import numpy as np


def _safe(v):
    if isinstance(v, (np.floating, np.integer)):
        v = v.item()
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    return v


def write_pair_csv(results: list[dict], path: str):
    cols = ["attack", "defense",
            "ASR", "CS", "valid_attack_rate",
            "mean_p_before", "mean_p_after",
            "EER_clean", "EER_robust", "AUC_clean", "AUC_robust",
            "min_tDCF_clean", "min_tDCF_robust",
            "minDCF_clean", "minDCF_robust",
            "FRR_clean", "FAR_clean", "FRR_robust", "FAR_robust",
            "Q", "CoreASR", "AttackScore", "DefenseScore",
            "asr_available", "wall_time_s"]
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in results:
            w.writerow([_safe(r.get(c)) for c in cols])


def _agg_attack_report(results: list[dict]) -> list[dict]:
    by_attack = {}
    for r in results:
        by_attack.setdefault(r["attack"], []).append(r)
    rows = []
    for a, rs in by_attack.items():
        fids = [r["fidelity"] for r in rs if r.get("fidelity")]
        def _mean(k):
            vals = [f.get(k, float("nan")) for f in fids]
            vals = [v for v in vals if np.isfinite(v)]
            return float(np.mean(vals)) if vals else float("nan")
        rows.append({
            "attack": a,
            "mean_ASR": float(np.mean([r["ASR"] for r in rs])),
            "mean_CS": float(np.mean([r["CS"] for r in rs])),
            "mean_valid_rate": float(np.mean([r["valid_attack_rate"] for r in rs])),
            "per_defense_ASR": {r["defense"]: _safe(r["ASR"]) for r in rs},
            "STOI": _mean("STOI"),
            "SI_SDR": _mean("SI_SDR"),
            "duration_ratio": _mean("duration_ratio"),
            "clipping_ratio_est": _mean("clipping_ratio_est"),
            "mean_Q": float(np.mean([r["Q"] for r in rs])),
            "AttackScore": float(np.mean([r["AttackScore"] for r in rs])),
        })
    rows.sort(key=lambda x: -x["AttackScore"])
    return rows


def _agg_defense_report(results: list[dict]) -> list[dict]:
    by_def = {}
    for r in results:
        by_def.setdefault(r["defense"], []).append(r)
    rows = []
    for d, rs in by_def.items():
        # robust metrics = averaged over attacks; clean = same across attacks
        clean = rs[0]
        rows.append({
            "defense": d,
            "EER_clean": clean["EER_clean"],
            "AUC_clean": clean["AUC_clean"],
            "EER_robust_mean": float(np.mean([r["EER_robust"] for r in rs])),
            "AUC_robust_mean": float(np.mean([r["AUC_robust"] for r in rs])),
            "min_tDCF_robust_mean": float(np.mean([r["min_tDCF_robust"] for r in rs])),
            "per_attack_EER": {r["attack"]: _safe(r["EER_robust"]) for r in rs},
            "per_attack_ASR": {r["attack"]: _safe(r["ASR"]) for r in rs},
            "DefenseScore": float(np.mean([r["DefenseScore"] for r in rs])),
        })
    rows.sort(key=lambda x: x["EER_robust_mean"])
    return rows


def _heatmap(results: list[dict], attacks: list[str], defenses: list[str],
             metric: str) -> np.ndarray:
    M = np.full((len(attacks), len(defenses)), np.nan)
    aidx = {a: i for i, a in enumerate(attacks)}
    didx = {d: j for j, d in enumerate(defenses)}
    for r in results:
        if r["attack"] in aidx and r["defense"] in didx:
            M[aidx[r["attack"]], didx[r["defense"]]] = r.get(metric, np.nan)
    return M


def _write_matrix_csv(M: np.ndarray, rows: list[str], cols: list[str], path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["attack\\defense"] + cols)
        for i, r in enumerate(rows):
            w.writerow([r] + [("" if np.isnan(M[i, j]) else f"{M[i,j]:.4f}")
                              for j in range(len(cols))])


def _try_heatmap_png(asr_M, eer_M, attacks, defenses, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(max(8, 2 + 1.2 * len(defenses)),
                                            max(4, 0.6 + 0.8 * len(attacks))))
    for ax, M, title, cmap in [
        (axes[0], asr_M, "Attack Success Rate (higher = better attack)", "Reds"),
        (axes[1], eer_M, "Robust EER (lower = better defense)", "Blues"),
    ]:
        im = ax.imshow(M, cmap=cmap, aspect="auto", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(defenses)))
        ax.set_xticklabels(defenses, rotation=45, ha="right")
        ax.set_yticks(range(len(attacks)))
        ax.set_yticklabels(attacks)
        ax.set_title(title)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                ax.text(j, i, "" if np.isnan(v) else f"{v:.2f}",
                        ha="center", va="center", color="black", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def generate_reports(results: list[dict], out_dir: str) -> dict:
    """Write the full report bundle.  Returns a summary dict."""
    os.makedirs(out_dir, exist_ok=True)
    attacks = sorted({r["attack"] for r in results})
    defenses = sorted({r["defense"] for r in results})

    write_pair_csv(results, os.path.join(out_dir, "pair_metrics.csv"))

    attack_report = _agg_attack_report(results)
    defense_report = _agg_defense_report(results)

    with open(os.path.join(out_dir, "attack_report.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["attack", "mean_ASR", "mean_CS", "mean_valid_rate",
                    "STOI", "SI_SDR", "duration_ratio", "clipping_ratio_est",
                    "mean_Q", "AttackScore"])
        for r in attack_report:
            w.writerow([r["attack"], f"{r['mean_ASR']:.4f}", f"{r['mean_CS']:.4f}",
                        f"{r['mean_valid_rate']:.4f}",
                        f"{r['STOI']:.4f}", f"{r['SI_SDR']:.4f}",
                        f"{r['duration_ratio']:.4f}", f"{r['clipping_ratio_est']:.4f}",
                        f"{r['mean_Q']:.4f}", f"{r['AttackScore']:.4f}"])

    with open(os.path.join(out_dir, "defense_report.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["defense", "EER_clean", "AUC_clean", "EER_robust_mean",
                    "AUC_robust_mean", "min_tDCF_robust_mean", "DefenseScore"])
        for r in defense_report:
            w.writerow([r["defense"], f"{r['EER_clean']:.4f}", f"{r['AUC_clean']:.4f}",
                        f"{r['EER_robust_mean']:.4f}", f"{r['AUC_robust_mean']:.4f}",
                        f"{r['min_tDCF_robust_mean']:.4f}", f"{r['DefenseScore']:.4f}"])

    asr_M = _heatmap(results, attacks, defenses, "ASR")
    eer_M = _heatmap(results, attacks, defenses, "EER_robust")
    _write_matrix_csv(asr_M, attacks, defenses,
                      os.path.join(out_dir, "heatmap_asr.csv"))
    _write_matrix_csv(eer_M, attacks, defenses,
                      os.path.join(out_dir, "heatmap_eer.csv"))
    png_ok = _try_heatmap_png(asr_M, eer_M, attacks, defenses,
                              os.path.join(out_dir, "heatmap.png"))

    summary = {
        "n_attacks": len(attacks), "n_defenses": len(defenses),
        "n_pairs": len(results),
        "attacks": attacks, "defenses": defenses,
        "attack_leaderboard": attack_report,
        "defense_leaderboard": defense_report,
        "heatmap_png": png_ok,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary
