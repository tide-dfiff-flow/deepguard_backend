"""D_RawNet2: end-to-end raw-waveform countermeasure (spec §十).

A CPU-trainable port of the ASVspoof 2021 RawNet2 baseline
(``2021-main/.../Baseline-RawNet2/model.py``): SincConv front-end (learned
Mel-spaced band-pass filters) -> residual blocks with self-attention ->
GRU -> linear.  We use a slightly reduced channel count so it trains in
reasonable time on CPU, but the architecture and data flow are faithful.
"""
from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseDefense


def _to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


class _SincConv(nn.Module):
    """Mel-spaced band-pass sinc filters (RawNet2 front-end)."""
    def __init__(self, out_channels=20, kernel_size=251, sample_rate=16000):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        # mel-spaced band edges
        mel_h = _to_mel(sample_rate / 2.0)
        mel_edges = np.linspace(0, mel_h, out_channels + 1)
        self.register_buffer("mel", torch.tensor(_to_hz(mel_edges), dtype=torch.float32))
        self.register_buffer("hsupp", torch.arange(-(kernel_size - 1) / 2,
                                                    (kernel_size - 1) / 2 + 1,
                                                    dtype=torch.float32))

    def forward(self, x):
        # x: (N, 1, T)
        band_pass = torch.zeros(self.out_channels, self.kernel_size, device=x.device)
        for i in range(len(self.mel) - 1):
            fmin = float(self.mel[i])
            fmax = float(self.mel[i + 1])
            sr = self.sample_rate
            h_high = (2 * fmax / sr) * np.sinc(2 * fmax * self.hsupp.numpy() / sr)
            h_low = (2 * fmin / sr) * np.sinc(2 * fmin * self.hsupp.numpy() / sr)
            hideal = torch.tensor(h_high - h_low, device=x.device)
            band_pass[i] = torch.hamming_window(self.kernel_size, device=x.device) * hideal
        filters = band_pass.unsqueeze(1)
        return F.conv1d(x, filters, stride=1, padding=0)


class _ResidualBlock(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first
        if not self.first:
            self.bn1 = nn.BatchNorm1d(nb_filts[0])
        self.lrelu = nn.LeakyReLU(0.3)
        self.conv1 = nn.Conv1d(nb_filts[0], nb_filts[1], 3, padding=1)
        self.bn2 = nn.BatchNorm1d(nb_filts[1])
        self.conv2 = nn.Conv1d(nb_filts[1], nb_filts[1], 3, padding=1)
        self.downsample = (nb_filts[0] != nb_filts[1])
        if self.downsample:
            self.conv_down = nn.Conv1d(nb_filts[0], nb_filts[1], 1)
        self.mp = nn.MaxPool1d(3)

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.lrelu(self.bn1(x))
        else:
            out = x
        out = self.lrelu(self.bn2(self.conv1(out)))
        out = self.conv2(out)
        if self.downsample:
            identity = self.conv_down(identity)
        out = out + identity
        return self.mp(out)


class _RawNet2(nn.Module):
    def __init__(self, sinc_channels=20, filts=((20, 20), (20, 64), (64, 64)),
                 gru_node=64, nb_classes=1):
        super().__init__()
        self.sinc = _SincConv(out_channels=sinc_channels)
        self.first_bn = nn.BatchNorm1d(sinc_channels)
        self.selu = nn.SELU(inplace=True)
        self.block0 = _ResidualBlock(filts[0], first=True)   # -> filts[0][1]
        self.block1 = _ResidualBlock(filts[1])               # -> filts[1][1]
        self.block2 = _ResidualBlock(filts[2])               # -> filts[2][1]
        self.block3 = _ResidualBlock((filts[2][1], filts[2][1]))
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # attention: dims must match the output channels of each block
        self.attn1 = nn.Linear(filts[0][1], filts[0][1])
        self.attn2 = nn.Linear(filts[1][1], filts[1][1])
        self.attn3 = nn.Linear(filts[2][1], filts[2][1])
        self.bn_gru = nn.BatchNorm1d(filts[2][1])
        self.gru = nn.GRU(filts[2][1], gru_node, batch_first=True)
        self.fc1 = nn.Linear(gru_node, 32)
        self.fc2 = nn.Linear(32, nb_classes)
        self.sig = nn.Sigmoid()

    def _att(self, x, fc):
        y = self.avgpool(x).squeeze(-1)
        y = self.sig(fc(y)).unsqueeze(-1)
        return x * y + y

    def forward(self, x):
        # x: (N, T)
        x = x.unsqueeze(1)
        x = torch.abs(self.sinc(x))
        x = F.max_pool1d(x, 3)
        x = self.selu(self.first_bn(x))
        x = self._att(self.block0(x), self.attn1)
        x = self._att(self.block1(x), self.attn2)
        x = self._att(self.block2(x), self.attn3)
        x = self.block3(x)
        x = self.selu(self.bn_gru(x))
        x = x.permute(0, 2, 1)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        return self.fc2(x).squeeze(-1)


class RawNet2Defense(BaseDefense):
    name = "D_RawNet2"
    params = {
        "max_samples": 16000 * 4,   # ~4s context
        "epochs": 6,
        "batch_size": 8,
        "lr": 1e-3,
        "seed": 0,
        "device": "cpu",
    }

    def __init__(self, **params):
        super().__init__(**params)
        self._device = torch.device(self.params["device"])
        self._model = _RawNet2().to(self._device)
        self._calib = {"mean": 0.0, "scale": 1.0}

    def _tensor(self, audio, sr):
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        ms = self.params["max_samples"]
        if x.shape[0] < ms:
            x = np.pad(x, (0, ms - x.shape[0]))
        else:
            s = (x.shape[0] - ms) // 2
            x = x[s:s + ms]
        return torch.from_numpy(x).unsqueeze(0).to(self._device)

    def train(self, bonafide_samples, spoof_samples, sample_rate=16000):
        torch.manual_seed(self.params["seed"])
        X = ([self._tensor(s, sample_rate) for s in bonafide_samples]
             + [self._tensor(s, sample_rate) for s in spoof_samples])
        Y = ([0.0] * len(bonafide_samples) + [1.0] * len(spoof_samples))
        if len(X) < 2:
            raise ValueError("RawNet2 needs >=2 samples")
        X_t = torch.cat([t for t in X], 0)
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
        self._model.eval()
        with torch.no_grad():
            logits = self._model(X_t).cpu().numpy()
        self._calib["mean"] = float(np.mean(logits))
        self._calib["scale"] = float(np.std(logits) + 1e-6)
        self.is_trained = True

    def score(self, audio, sample_rate):
        if not self.is_trained:
            return 0.0
        self._model.eval()
        with torch.no_grad():
            t = self._tensor(audio, sample_rate)
            return float(self._model(t).item())

    def _to_probability(self, raw_score, audio, sample_rate):
        if not self.is_trained:
            return 0.5
        z = (raw_score - self._calib["mean"]) / self._calib["scale"]
        return 1.0 / (1.0 + np.exp(-z))

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
