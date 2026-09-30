#!/usr/bin/env python
"""DeepGuard-Audio 攻防对抗后端主入口。

对任意组合的攻击方法 × 防御方法执行全对阵（spec §10.2），自动计算攻击
指标（ASR / 置信度压制 / 保真度）与防御指标（EER / AUC / minDCF / 每
攻击 EER），并输出报告 + 攻防热力图（spec §17）。

用法:
    python run_eval.py                         # 用 config.yaml 默认全对阵
    python run_eval.py --attacks A_codec,A_noise --defenses D_GMM,D_LCNN
    python run_eval.py --data /path/to/protocol.csv --audio-dir /path/to/wav
    python run_eval.py --quick                 # 小规模快速冒烟
"""
from __future__ import annotations

import os
import sys
import argparse

# allow running as a script
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)  # project root containing the package
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from deepguard_backend.attacks import build_attack, list_attacks
from deepguard_backend.defenses import build_defense, list_defenses
from deepguard_backend.engine import (MatchupEngine, FidelityJudge,
                                      load_protocol, generate_reports)
from deepguard_backend.data_gen import generate_dataset


def _load_config(path):
    try:
        import yaml
    except Exception:
        # minimal fallback parser for the subset of YAML we use
        return _fallback_yaml(path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _fallback_yaml(path):
    """Very small YAML reader for the flat key: value / nested dict layout
    used by config.yaml.  Only good enough when PyYAML is unavailable."""
    cfg = {}
    stack = [(0, cfg)]
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            indent = len(raw) - len(raw.lstrip())
            keyval = raw.strip()
            while stack and indent < stack[-1][0]:
                stack.pop()
            parent = stack[-1][1]
            if ":" in keyval and not keyval.startswith("- "):
                k, v = keyval.split(":", 1)
                k = k.strip()
                v = v.strip()
                if v == "" or v.startswith("#"):
                    new = {}
                    parent[k] = new
                    stack.append((indent + 2, new))
                else:
                    parent[k] = _parse_scalar(v)
    return cfg


def _parse_scalar(v):
    v = v.strip()
    if v in ("true", "True"):
        return True
    if v in ("false", "False"):
        return False
    if v.startswith('"') and v.endswith('"'):
        return v[1:-1]
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(x.strip()) for x in inner.split(",")]
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def _parse_list(spec):
    if spec in (None, "", "all"):
        return None
    return [s.strip() for s in spec.split(",") if s.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(description="DeepGuard-Audio adversarial backend")
    ap.add_argument("--config", default=os.path.join(_HERE, "config.yaml"))
    ap.add_argument("--attacks", default=None, help="comma list or 'all'")
    ap.add_argument("--defenses", default=None, help="comma list or 'all'")
    ap.add_argument("--data", default=None, help="protocol.csv path")
    ap.add_argument("--audio-dir", default=None)
    ap.add_argument("--out", default=None, help="report output dir")
    ap.add_argument("--quick", action="store_true", help="small smoke run")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    cfg = _load_config(args.config)
    data_cfg = cfg.get("data", {})
    eval_cfg = cfg.get("eval", {})
    run_cfg = cfg.get("run", {})

    # ---------------- data ----------------
    if args.data:
        protocol = args.data
        audio_dir = args.audio_dir or os.path.dirname(protocol)
    elif data_cfg.get("protocol"):
        protocol = data_cfg["protocol"]
        audio_dir = data_cfg.get("audio_dir", os.path.dirname(protocol))
    else:
        if args.quick:
            data_cfg = {**data_cfg, "n_bonafide": 8, "n_spoof": 8, "duration": 2.0}
        protocol = generate_dataset(
            data_cfg.get("synth_out", os.path.join(_HERE, "data", "synth")),
            n_bonafide=data_cfg.get("n_bonafide", 24),
            n_spoof=data_cfg.get("n_spoof", 24),
            duration=data_cfg.get("duration", 3.0),
            seed=data_cfg.get("seed", 1234))
        audio_dir = os.path.join(os.path.dirname(protocol), "wav")

    samples = load_protocol(protocol, audio_dir)
    bonafide = [s for s in samples if s.label == 0]
    spoof = [s for s in samples if s.label == 1]
    if not bonafide or not spoof:
        print(f"ERROR: need both bonafide ({len(bonafide)}) and spoof ({len(spoof)}) samples")
        sys.exit(1)
    print(f"Loaded {len(bonafide)} bonafide + {len(spoof)} spoof samples")

    # ---------------- attacks / defenses ----------------
    atk_overrides = cfg.get("attacks", {})
    def_overrides = cfg.get("defenses", {})

    atk_names = _parse_list(args.attacks) or _parse_list(run_cfg.get("attacks"))
    atk_names = atk_names or list_attacks()
    def_names = _parse_list(args.defenses) or _parse_list(run_cfg.get("defenses"))
    def_names = def_names or list_defenses()

    if args.quick:
        # trim to a fast subset unless user overrode
        atk_names = [a for a in ["A_codec", "A_noise"] if a in atk_names] or atk_names[:2]
        def_names = [d for d in ["D_GMM", "D_flat"] if d in def_names] or def_names[:2]

    attacks = [build_attack(n, **atk_overrides.get(n, {})) for n in atk_names]
    defenses = []
    for n in def_names:
        params = dict(def_overrides.get(n, {}))
        defenses.append(build_defense(n, **params))

    print(f"Attacks ({len(attacks)}): {atk_names}")
    print(f"Defenses ({len(defenses)}): {def_names}")

    # ---------------- engine ----------------
    judge = FidelityJudge(
        duration_min=eval_cfg.get("duration_min", 0.95),
        duration_max=eval_cfg.get("duration_max", 1.05),
        stoi_min=eval_cfg.get("stoi_min", 0.45),
        si_sdr_min=eval_cfg.get("si_sdr_min", -5.0),
        cer_max=eval_cfg.get("cer_max", 0.30),
        clipping_max=eval_cfg.get("clipping_max", 0.10),
        silence_max=eval_cfg.get("silence_max", 0.80),
        require_asr=eval_cfg.get("require_asr", False),
    )
    engine = MatchupEngine(bonafide, spoof, judge=judge,
                           threshold=eval_cfg.get("threshold", 0.5),
                           verbose=not args.quiet)
    results = engine.run_grid(attacks, defenses)

    # ---------------- report ----------------
    out_dir = args.out or run_cfg.get("out_dir", os.path.join(_HERE, "report"))
    summary = generate_reports(results, out_dir)
    print("\n========== SUMMARY ==========")
    print(f"Pairs evaluated: {summary['n_pairs']} "
          f"({summary['n_attacks']} attacks x {summary['n_defenses']} defenses)")
    print("\n-- Attack leaderboard (by AttackScore) --")
    for r in summary["attack_leaderboard"]:
        print(f"  {r['attack']:10s} AttackScore={r['AttackScore']:.2f} "
              f"mean_ASR={r['mean_ASR']:.3f} mean_CS={r['mean_CS']:.3f} "
              f"STOI={r['STOI']:.3f} SI_SDR={r['SI_SDR']:.2f}")
    print("\n-- Defense leaderboard (by Robust EER, lower=better) --")
    for r in summary["defense_leaderboard"]:
        print(f"  {r['defense']:10s} EER_clean={r['EER_clean']:.3f} "
              f"EER_robust={r['EER_robust_mean']:.3f} "
              f"AUC_robust={r['AUC_robust_mean']:.3f} "
              f"DefenseScore={r['DefenseScore']:.2f}")
    print(f"\nHeatmap PNG: {summary['heatmap_png']}")
    print(f"Reports written to: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
