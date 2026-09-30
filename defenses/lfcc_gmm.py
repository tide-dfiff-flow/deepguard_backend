"""D_GMM: LFCC + two-GMM countermeasure (spec §15, D0-1).

A faithful, dependency-light port of the ASVspoof 2021 LFCC-GMM baseline
(``2021-main/.../Baseline-LFCC-GMM/python/gmm.py``):

* extract LFCC frames (our numpy/scipy front-end),
* train two diagonal-covariance GMMs -- one on bonafide frames, one on spoof
  frames -- using ``sklearn.mixture.GaussianMixture``,
* score an utterance as ``LLR = score_bona(x) - score_spoof(x)`` and map to a
  spoof probability via a logistic calibrated on the training LLRs.

This is CPU-friendly, simple, and a real ASVspoof baseline.
"""
from __future__ import annotations

import pickle
import os
import numpy as np
from sklearn.mixture import GaussianMixture

from .base import BaseDefense
from ..features.lfcc import extract_lfcc_array


class LFCCGMMDefense(BaseDefense):
    name = "D_GMM"
    params = {
        "n_components": 32,
        "num_ceps": 20,
        "with_delta": True,
        "max_frames_per_utt": 600,
        "llr_scale": 1.0,
        "seed": 0,
    }

    def __init__(self, **params):
        super().__init__(**params)
        self._gmm_bona = None
        self._gmm_spoof = None
        self._calib = {"mean": 0.0, "scale": 1.0}  # LLR calibration

    # ------------------------------------------------------------------ #
    # feature extraction
    # ------------------------------------------------------------------ #
    def _featurize(self, audio: np.ndarray, sr: int) -> np.ndarray:
        feats = extract_lfcc_array(audio, sr, num_ceps=self.params["num_ceps"],
                                   with_delta=self.params["with_delta"])
        # cap number of frames for speed
        mf = self.params["max_frames_per_utt"]
        if feats.shape[0] > mf:
            idx = np.linspace(0, feats.shape[0] - 1, mf).astype(int)
            feats = feats[idx]
        return feats

    # ------------------------------------------------------------------ #
    # training
    # ------------------------------------------------------------------ #
    def train(self, bonafide_samples, spoof_samples, sample_rate=16000):
        bona_X = self._stack_frames(bonafide_samples, sample_rate)
        spoof_X = self._stack_frames(spoof_samples, sample_rate)
        if bona_X.shape[0] == 0 or spoof_X.shape[0] == 0:
            raise ValueError("GMM defense needs both bonafide and spoof frames")

        n = self.params["n_components"]
        seed = self.params["seed"]
        self._gmm_bona = GaussianMixture(
            n_components=min(n, bona_X.shape[0]),
            covariance_type="diag", max_iter=20, reg_covar=1e-3,
            random_state=seed).fit(bona_X)
        self._gmm_spoof = GaussianMixture(
            n_components=min(n, spoof_X.shape[0]),
            covariance_type="diag", max_iter=20, reg_covar=1e-3,
            random_state=seed).fit(spoof_X)

        # calibrate LLR distribution for probability mapping
        bona_llr = self._llr_frames(bonafide_samples, sample_rate)
        spoof_llr = self._llr_frames(spoof_samples, sample_rate)
        all_llr = np.concatenate([bona_llr, spoof_llr])
        self._calib["mean"] = float(np.mean(all_llr))
        self._calib["scale"] = float(np.std(all_llr) + 1e-6)
        self.is_trained = True

    def _stack_frames(self, samples, sr):
        feats = [self._featurize(s, sr) for s in samples]
        feats = [f for f in feats if f.shape[0] > 0]
        if not feats:
            return np.zeros((0, self.params["num_ceps"] * (3 if self.params["with_delta"] else 1)),
                            dtype=np.float32)
        return np.vstack(feats)

    def _llr_frames(self, samples, sr):
        llrs = []
        for s in samples:
            f = self._featurize(s, sr)
            if f.shape[0] == 0:
                continue
            llrs.append(self._gmm_bona.score(f) - self._gmm_spoof.score(f))
        return np.asarray(llrs, dtype=np.float64)

    # ------------------------------------------------------------------ #
    # scoring
    # ------------------------------------------------------------------ #
    def score(self, audio, sample_rate):
        if self._gmm_bona is None or self._gmm_spoof is None:
            return 0.5
        f = self._featurize(audio, sample_rate)
        if f.shape[0] == 0:
            return 0.5
        llr = float(self._gmm_bona.score(f) - self._gmm_spoof.score(f))
        return llr

    def _to_probability(self, raw_score, audio, sample_rate):
        # higher LLR => more bonafide => lower spoof_probability
        z = (raw_score - self._calib["mean"]) / self._calib["scale"]
        z *= self.params["llr_scale"]
        return 1.0 / (1.0 + np.exp(z))  # sigmoid: large LLR -> 0 (bonafide)

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "gmm_bona": self._gmm_bona,
                "gmm_spoof": self._gmm_spoof,
                "calib": self._calib,
                "params": self.params,
            }, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            d = pickle.load(f)
        self._gmm_bona = d["gmm_bona"]
        self._gmm_spoof = d["gmm_spoof"]
        self._calib = d["calib"]
        self.params = {**self.params, **d.get("params", {})}
        self.is_trained = True
