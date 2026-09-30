"""Attack × Defense matchup engine.

Runs the full n×m grid (spec §10.2) with strict decoupling (§10.3):

1. Train each trainable defense once on the bonafide/spoof training split.
2. For every attack A, apply ``A.attack`` to all spoof samples *once* and
   cache the attacked audio + per-sample fidelity/validity (the attack never
   sees the defense).
3. For every (A, D) pair, run D on bonafide (original), spoof (original) and
   spoof-attacked-by-A, then compute attack metrics (ASR, CS) and defense
   metrics (clean EER/AUC, robust EER/AUC, per-attack EER).

The clean-defense metrics (no attack) are computed once per defense and
reused; the per-attack defense metrics reuse the cached attacked audio.
"""
from __future__ import annotations

import time
import numpy as np

from ..attacks.base import BaseAttack
from ..defenses.base import BaseDefense
from ..defenses.registry import TRAINABLE_DEFENSES
from ..metrics.detection import all_detection_metrics
from ..metrics.attack_metrics import attack_metric_bundle
from ..metrics.fidelity import fidelity_bundle
from ..metrics.content import content_preservation
from ..metrics.scoring import (quality_coefficient, core_asr, attack_score,
                                defense_score)
from .sample import Sample, LABEL_SPOOF, LABEL_BONAFIDE
from .fidelity_judge import FidelityJudge, precheck


class MatchupEngine:
    def __init__(self, bonafide: list[Sample], spoof: list[Sample],
                 judge: FidelityJudge | None = None,
                 threshold: float = 0.5,
                 asr_fn=None,
                 verbose: bool = True):
        self.bonafide = bonafide
        self.spoof = spoof
        self.judge = judge or FidelityJudge()
        self.threshold = threshold
        self.asr_fn = asr_fn
        self.verbose = verbose
        # caches
        self._attacked_cache: dict[str, dict] = {}   # attack_name -> per-sample
        self._clean_defense: dict[str, dict] = {}    # defense_name -> metrics

    # ------------------------------------------------------------------ #
    # logging
    # ------------------------------------------------------------------ #
    def _log(self, msg):
        if self.verbose:
            print(f"[engine] {msg}", flush=True)

    # ------------------------------------------------------------------ #
    # step 1: train defenses
    # ------------------------------------------------------------------ #
    def train_defenses(self, defenses: list[BaseDefense]):
        for d in defenses:
            if d.name in TRAINABLE_DEFENSES and not d.is_trained:
                t0 = time.time()
                d.train([s.audio for s in self.bonafide],
                        [s.audio for s in self.spoof],
                        sample_rate=self.bonafide[0].sample_rate if self.bonafide else 16000)
                self._log(f"trained {d.name} in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------ #
    # step 2: run each attack on all spoof samples (cached)
    # ------------------------------------------------------------------ #
    def run_attack(self, attack: BaseAttack):
        if attack.name in self._attacked_cache:
            return self._attacked_cache[attack.name]
        self._log(f"running attack {attack.name} on {len(self.spoof)} spoof samples")
        per_sample = []
        for s in self.spoof:
            out = attack.attack(s.as_attack_input())
            est, sr, _ = precheck(out["audio"], out.get("sample_rate", 16000))
            fid = fidelity_bundle(s.audio, est, sr)
            content = content_preservation(s.audio, est, sr, asr_fn=self.asr_fn)
            verdict = self.judge.judge(s.audio, est, sr, fidelity=fid, content=content)
            # store pre-attack spoof probability placeholder (filled later per defense)
            per_sample.append({
                "sample_id": s.sample_id,
                "orig_audio": s.audio,
                "attacked_audio": est,
                "sample_rate": sr,
                "valid": verdict["valid"],
                "reasons": verdict["reasons"],
                "fidelity": verdict["fidelity"],
                "content": verdict["content"],
            })
        self._attacked_cache[attack.name] = per_sample
        return per_sample

    # ------------------------------------------------------------------ #
    # defense scoring helpers
    # ------------------------------------------------------------------ #
    def _defend_many(self, defense: BaseDefense, samples: list[Sample]) -> np.ndarray:
        out = np.zeros(len(samples), dtype=np.float64)
        for i, s in enumerate(samples):
            out[i] = defense.defend(s.as_defense_input())["spoof_probability"]
        return out

    def _defend_audio_many(self, defense: BaseDefense, items: list[dict]) -> np.ndarray:
        out = np.zeros(len(items), dtype=np.float64)
        for i, it in enumerate(items):
            out[i] = defense.defend({"sample_id": it["sample_id"],
                                     "audio": it["attacked_audio"],
                                     "sample_rate": it["sample_rate"]})["spoof_probability"]
        return out

    def _clean_metrics(self, defense: BaseDefense) -> dict:
        if defense.name in self._clean_defense:
            return self._clean_defense[defense.name]
        bp = self._defend_many(defense, self.bonafide)
        sp = self._defend_many(defense, self.spoof)
        m = all_detection_metrics(bp, sp, self.threshold)
        m["bonafide_prob"] = bp
        m["spoof_prob"] = sp
        self._clean_defense[defense.name] = m
        return m

    # ------------------------------------------------------------------ #
    # step 3: one (attack, defense) pair
    # ------------------------------------------------------------------ #
    def evaluate_pair(self, attack: BaseAttack, defense: BaseDefense) -> dict:
        attacked = self.run_attack(attack)
        clean = self._clean_metrics(defense)

        # spoof probabilities: original (before attack) and after attack
        p_before = clean["spoof_prob"]                # on original spoof
        p_after = self._defend_audio_many(defense, attacked)
        valid = np.array([it["valid"] for it in attacked], dtype=bool)

        atk = attack_metric_bundle(p_before, p_after, valid, self.threshold)

        # robust defense metrics: bonafide(original) + attacked-spoof
        bp = clean["bonafide_prob"]
        robust = all_detection_metrics(bp, p_after, self.threshold)

        # per-attack fidelity aggregate (mean over valid samples, or all)
        fids = [it["fidelity"] for it in attacked]
        agg_fid = self._aggregate_fidelity(fids)
        contents = [it["content"] for it in attacked]
        asr_avail = any(c.get("asr_available") for c in contents)

        # scores
        q = quality_coefficient({**agg_fid,
                                 "CER": (np.mean([c["CER"] for c in contents
                                                  if np.isfinite(c["CER"])]) if asr_avail else float("nan")),
                                 "content_proxy": (np.mean([c["content_proxy"] for c in contents])
                                                   if not asr_avail else float("nan"))})
        casr = core_asr(atk["ASR"], atk["ASR"])   # hidden pool = fixed pool here
        a_score = attack_score(casr, q)
        d_score = defense_score(robust["EER"], clean["EER"], robust["AUC"], efficiency=0.8)

        return {
            "attack": attack.name,
            "defense": defense.name,
            # attack metrics
            "ASR": atk["ASR"],
            "CS": atk["CS"],
            "valid_attack_rate": atk["valid_attack_rate"],
            "mean_p_before": atk["mean_p_before"],
            "mean_p_after": atk["mean_p_after"],
            # defense metrics
            "EER_clean": clean["EER"],
            "EER_robust": robust["EER"],
            "AUC_clean": clean["AUC"],
            "AUC_robust": robust["AUC"],
            "min_tDCF_clean": clean["min_tDCF"],
            "min_tDCF_robust": robust["min_tDCF"],
            "minDCF_clean": clean["minDCF"],
            "minDCF_robust": robust["minDCF"],
            "FRR_clean": clean["FRR"],
            "FAR_clean": clean["FAR"],
            "FRR_robust": robust["FRR"],
            "FAR_robust": robust["FAR"],
            # fidelity aggregate
            "fidelity": agg_fid,
            "asr_available": asr_avail,
            # scores
            "Q": q,
            "CoreASR": casr,
            "AttackScore": a_score,
            "DefenseScore": d_score,
        }

    @staticmethod
    def _aggregate_fidelity(fids: list[dict]) -> dict:
        if not fids:
            return {}
        keys = ["SI_SDR", "STOI", "PESQ", "RMS_ratio", "duration_ratio",
                "clipping_ratio_est", "silence_ratio_est"]
        agg = {}
        for k in keys:
            vals = [f.get(k, float("nan")) for f in fids]
            vals = [v for v in vals if np.isfinite(v)]
            agg[k] = float(np.mean(vals)) if vals else float("nan")
        return agg

    # ------------------------------------------------------------------ #
    # full grid
    # ------------------------------------------------------------------ #
    def run_grid(self, attacks: list[BaseAttack],
                 defenses: list[BaseDefense]) -> list[dict]:
        self.train_defenses(defenses)
        # pre-run all attacks (caches attacked audio shared across defenses)
        for a in attacks:
            self.run_attack(a)
        results = []
        for d in defenses:
            for a in attacks:
                self._log(f"pair: {a.name} x {d.name}")
                t0 = time.time()
                res = self.evaluate_pair(a, d)
                res["wall_time_s"] = time.time() - t0
                results.append(res)
        return results
