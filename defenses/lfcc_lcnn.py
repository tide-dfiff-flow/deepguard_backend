"""D_LCNN: LFCC + LCNN countermeasure (spec §十).

A practical, CPU-trainable port of the ASVspoof 2021 LFCC-LCNN baseline
(``2021-main/.../Baseline-LFCC-LCNN/project/baseline_DF/model.py``).  The
front-end is LFCC (numpy, shared with the GMM defense).  The back-end is an
LCNN with the signature *MaxFeatureMap* activation (elementwise max over
pairs of channels) and Conv2D blocks, then global average pooling and a
linear classifier.  This is trainable end-to-end on whatever bonafide/spoof
data is provided and produces a calibrated spoof probability.
"""
from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn

from .base import BaseDefense
from ..features.lfcc import extract_lfcc_array


class _MaxFeatureMap(nn.Module):
    """Maxout over channel pairs (the LCNN 'MaxFeatureMap2D' activation)."""
    def forward(self, x):
        # x: (N, C, ...) -> (N, C//2, ...)
        if x.shape[1] % 2 != 0:
            x = x[:, :-1]
        a, b = x[:, ::2], x[:, 1::2]
        return torch.max(a, b)


class _LCNN(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 1):
        super().__init__()
        self.front = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2), _MaxFeatureMap(),
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.Conv2d(32, 64, kernel_size=1), _MaxFeatureMap(),
            nn.BatchNorm2d(32), nn.Conv2d(32, 96, kernel_size=3, padding=1),
            _MaxFeatureMap(), nn.MaxPool2d((2, 2), (2, 2)), nn.BatchNorm2d(48),
            nn.Conv2d(48, 96, kernel_size=1), _MaxFeatureMap(),
            nn.BatchNorm2d(48), nn.Conv2d(48, 128, kernel_size=3, padding=1),
            _MaxFeatureMap(), nn.MaxPool2d((2, 2), (2, 2)),
            nn.Conv2d(64, 128, kernel_size=1), _MaxFeatureMap(),
            nn.BatchNorm2d(64), nn.Conv2d(64, 64, kernel_size=3, padding=1),
            _MaxFeatureMap(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=1), _MaxFeatureMap(),
            nn.BatchNorm2d(32), nn.Conv2d(32, 64, kernel_size=3, padding=1),
            _MaxFeatureMap(), nn.AdaptiveAvgPool2d(1), nn.Dropout(0.3),
        )
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        # x: (N, 1, frames, feat_dim)
        h = self.front(x)
        h = h.flatten(1)
        return self.fc(h).squeeze(-1)


class LFCCLCNNDefense(BaseDefense):
    name = "D_LCNN"
    params = {
        "num_ceps": 20,
        "with_delta": True,
        "max_frames": 160,
        "epochs": 8,
        "batch_size": 16,
        "lr": 1e-3,
        "seed": 0,
        "device": "cpu",
    }

    def __init__(self, **params):
        super().__init__(**params)
        self._device = torch.device(self.params["device"])
        feat_dim = self.params["num_ceps"] * (3 if self.params["with_delta"] else 1)
        self._model = _LCNN(in_dim=feat_dim).to(self._device)
        self._feat_dim = feat_dim
        self._calib = {"mean": 0.0, "scale": 1.0}

    # ------------------------------------------------------------------ #
    # featurise to a fixed-shape tensor (1, frames, feat_dim)
    # ------------------------------------------------------------------ #
    def _featurize(self, audio, sr):
        f = extract_lfcc_array(audio, sr, num_ceps=self.params["num_ceps"],
                               with_delta=self.params["with_delta"])
        mf = self.params["max_frames"]
        if f.shape[0] == 0:
            f = np.zeros((1, self._feat_dim), dtype=np.float32)
        if f.shape[0] < mf:
            f = np.pad(f, ((0, mf - f.shape[0]), (0, 0)))
        elif f.shape[0] > mf:
            # centre crop
            s = (f.shape[0] - mf) // 2
            f = f[s:s + mf]
        return f.astype(np.float32)

    def _tensor(self, audio, sr):
        f = self._featurize(audio, sr)
        return torch.from_numpy(f).unsqueeze(0).unsqueeze(0).to(self._device)

    # ------------------------------------------------------------------ #
    # training
    # ------------------------------------------------------------------ #
    def train(self, bonafide_samples, spoof_samples, sample_rate=16000):
        torch.manual_seed(self.params["seed"])
        X_b = [self._featurize(s, sample_rate) for s in bonafide_samples]
        X_s = [self._featurize(s, sample_rate) for s in spoof_samples]
        X = X_b + X_s
        Y = [0.0] * len(X_b) + [1.0] * len(X_s)
        if len(X) < 2:
            raise ValueError("LCNN defense needs at least 2 training samples")

        X_t = torch.from_numpy(np.stack(X)).unsqueeze(1).to(self._device)
        Y_t = torch.tensor(Y, dtype=torch.float32, device=self._device)
        opt = torch.optim.Adam(self._model.parameters(), lr=self.params["lr"])
        loss_fn = nn.BCEWithLogitsLoss()
        self._model.train()
        n = len(X)
        bs = self.params["batch_size"]
        for ep in range(self.params["epochs"]):
            perm = torch.randperm(n)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                logits = self._model(X_t[idx])
                loss = loss_fn(logits, Y_t[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        # calibrate logits
        self._model.eval()
        with torch.no_grad():
            logits = self._model(X_t).cpu().numpy()
        self._calib["mean"] = float(np.mean(logits))
        self._calib["scale"] = float(np.std(logits) + 1e-6)
        self.is_trained = True

    # ------------------------------------------------------------------ #
    # scoring
    # ------------------------------------------------------------------ #
    def score(self, audio, sample_rate):
        if not self.is_trained:
            return 0.0
        self._model.eval()
        with torch.no_grad():
            t = self._tensor(audio, sample_rate)
            logit = float(self._model(t).item())
        return logit

    def _to_probability(self, raw_score, audio, sample_rate):
        if not self.is_trained:
            return 0.5
        z = (raw_score - self._calib["mean"]) / self._calib["scale"]
        return 1.0 / (1.0 + np.exp(-z))  # higher logit => more spoof

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        torch.save({"state_dict": self._model.state_dict(),
                    "calib": self._calib, "params": self.params}, path)

    def load(self, path: str):
        d = torch.load(path, map_location=self._device)
        self._model.load_state_dict(d["state_dict"])
        self._calib = d["calib"]
        self.params = {**self.params, **d.get("params", {})}
        self.is_trained = True
