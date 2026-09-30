"""A_hidden: combination attack (spec §十).

Applies a random/seeded chain of the base attacks (codec, noise, filter,
channel) to simulate unknown combined channel+post-processing.  This is the
hardest pool entry and the one used for the "hidden test C" analogue.
"""
from __future__ import annotations

import numpy as np

from .base import BaseAttack
from .codec_attack import CodecAttack
from .noise_attack import NoiseAttack
from .filter_attack import FilterAttack
from .channel_attack import ChannelAttack


class ComboAttack(BaseAttack):
    name = "A_hidden"
    params = {
        "stages": ("codec", "filter", "noise", "channel"),
        "shuffle": True,
        "codec": {"bitrate_kbps": 48, "bits": 7},
        "noise": {"snr_db": 22.0, "kind": "pink"},
        "filter": {"n_stages": 2, "gain_db": 2.5},
        "channel": {"rt60": 0.2, "resample_to": 8000},
        "seed": 0,
    }

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        x = np.asarray(audio, dtype=np.float64)
        stages = list(self.params["stages"])
        if self.params.get("shuffle", True):
            self.rng.shuffle(stages)
        for st in stages:
            sub_seed = int(self.rng.integers(0, 2 ** 31 - 1))
            if st == "codec":
                atk = CodecAttack(seed=sub_seed, **self.params.get("codec", {}))
            elif st == "noise":
                atk = NoiseAttack(seed=sub_seed, **self.params.get("noise", {}))
            elif st == "filter":
                atk = FilterAttack(seed=sub_seed, **self.params.get("filter", {}))
            elif st == "channel":
                atk = ChannelAttack(seed=sub_seed, **self.params.get("channel", {}))
            else:
                continue
            x = np.asarray(atk.process(x.astype(np.float32), sample_rate),
                           dtype=np.float64)
        return x.astype(np.float32)
