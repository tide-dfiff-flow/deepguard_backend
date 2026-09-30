"""Defense registry and factory."""
from __future__ import annotations

from .base import BaseDefense
from .lfcc_gmm import LFCCGMMDefense
from .lfcc_lcnn import LFCCLCNNDefense
from .rawnet2 import RawNet2Defense
from .heuristic_detectors import (
    SpectralFlatnessDefense, HighBandEnergyDefense, PhaseDiscontinuityDefense,
)


DEFENSE_REGISTRY = {
    "D_GMM": LFCCGMMDefense,
    "D_LCNN": LFCCLCNNDefense,
    "D_RawNet2": RawNet2Defense,
    # hidden / heuristic pool
    "D_flat": SpectralFlatnessDefense,
    "D_hbe": HighBandEnergyDefense,
    "D_phase": PhaseDiscontinuityDefense,
}

# Detectors that need training (heuristic ones calibrate their raw scores)
TRAINABLE_DEFENSES = {"D_GMM", "D_LCNN", "D_RawNet2", "D_flat", "D_hbe", "D_phase"}


def list_defenses() -> list[str]:
    return list(DEFENSE_REGISTRY.keys())


def build_defense(name: str, **params) -> BaseDefense:
    if name not in DEFENSE_REGISTRY:
        raise KeyError(f"Unknown defense '{name}'. Available: {list_defenses()}")
    return DEFENSE_REGISTRY[name](**params)


def build_all_defenses(overrides: dict | None = None) -> list[BaseDefense]:
    """Instantiate the full defense pool (spec §10.1 D0 + hidden)."""
    overrides = overrides or {}
    out = []
    for name, cls in DEFENSE_REGISTRY.items():
        out.append(cls(**overrides.get(name, {})))
    return out
