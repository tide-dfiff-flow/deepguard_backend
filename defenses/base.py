"""Base class and interface for defense / countermeasure methods (spec §7)."""
from __future__ import annotations

import numpy as np


class BaseDefense:
    """Defense (countermeasure) interface.

    Subclasses set ``name`` and implement :meth:`score` (and optionally
    :meth:`train`).  The public :meth:`defend` enforces the spec §7.3 contract:
    ``request = {sample_id, audio, sample_rate}`` ->
    ``{spoof_probability: 0..1}`` where 0 = bonafide, 1 = spoof.

    Defenses must be stateless w.r.t. the attack: they receive only the audio
    and never the label / generator / attack type (spec §3.4).
    """

    name: str = "base"
    params: dict = {}
    is_trained: bool = False

    def __init__(self, **params):
        self.params = {**self.params, **params}
        self.is_trained = False

    # ------------------------------------------------------------------ #
    # to override
    # ------------------------------------------------------------------ #
    def score(self, audio: np.ndarray, sample_rate: int) -> float:
        """Return a raw *spoof-ness* score (higher = more spoof).

        Subclasses can return an uncalibrated score; :meth:`defend` maps it
        to a probability in [0, 1].
        """
        raise NotImplementedError

    def train(self, bonafide_samples: list[np.ndarray],
              spoof_samples: list[np.ndarray], sample_rate: int = 16000) -> None:
        """Optional training hook.  Override if the detector is trainable."""
        self.is_trained = True

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def defend(self, request: dict) -> dict:
        audio = np.asarray(request["audio"], dtype=np.float32).reshape(-1)
        sr = int(request.get("sample_rate", 16000))
        try:
            raw = float(self.score(audio, sr))
        except Exception:
            raw = 0.5  # fail-safe neutral
        prob = self._to_probability(raw, audio, sr)
        prob = float(np.clip(prob, 0.0, 1.0))
        if not np.isfinite(prob):
            prob = 0.5
        return {"spoof_probability": prob}

    def _to_probability(self, raw_score: float, audio: np.ndarray,
                        sample_rate: int) -> float:
        """Map a raw score to ``spoof_probability`` in [0, 1].

        Default: identity (assume ``score`` already returns a probability).
        GMM-style detectors that return log-likelihood-rat should override
        this with a sigmoid centred on the LLR threshold.
        """
        return raw_score

    def __repr__(self):
        return f"<{self.__class__.__name__} name={self.name} trained={self.is_trained}>"
