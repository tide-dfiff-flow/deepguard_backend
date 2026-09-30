"""Base class and interface for attack methods (spec §5)."""
from __future__ import annotations

import numpy as np


class BaseAttack:
    """Attack interface.

    Subclasses set ``name`` and implement :meth:`process`.  The public
    :meth:`attack` enforces the spec contract: it takes a ``sample`` dict and
    returns ``{audio, sample_rate}``.  Attacks must NOT receive or query the
    defense (decoupling, spec §10.3).
    """

    name: str = "base"
    params: dict = {}

    def __init__(self, **params):
        self.params = {**self.params, **params}
        self.rng = np.random.default_rng(self.params.get("seed", None))

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply the attack to a float32 mono waveform.  Override me."""
        raise NotImplementedError

    def attack(self, sample: dict) -> dict:
        """Spec §5.4 entry point.

        ``sample = {sample_id, audio, sample_rate}`` ->
        ``{audio, sample_rate}``.
        """
        audio = np.asarray(sample["audio"], dtype=np.float32).reshape(-1)
        sr = int(sample.get("sample_rate", 16000))
        out = self.process(audio, sr)
        out = np.asarray(out, dtype=np.float64).reshape(-1)
        # safety: kill NaN/Inf, soft clip on float64 to avoid cast overflow
        if not np.isfinite(out).all():
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        out = np.tanh(out)  # gentle soft-clip instead of hard clipping
        return {"audio": out.astype(np.float32), "sample_rate": sr}

    def __repr__(self):
        return f"<{self.__class__.__name__} name={self.name} params={self.params}>"
