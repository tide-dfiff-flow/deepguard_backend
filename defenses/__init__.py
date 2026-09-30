"""Defense (countermeasure) methods for the DeepGuard-Audio backend."""
from .base import BaseDefense  # noqa: F401
from .lfcc_gmm import LFCCGMMDefense  # noqa: F401
from .lfcc_lcnn import LFCCLCNNDefense  # noqa: F401
from .rawnet2 import RawNet2Defense  # noqa: F401
from .heuristic_detectors import (  # noqa: F401
    SpectralFlatnessDefense, HighBandEnergyDefense, PhaseDiscontinuityDefense,
)
from .registry import (  # noqa: F401
    DEFENSE_REGISTRY, TRAINABLE_DEFENSES, list_defenses,
    build_defense, build_all_defenses,
)
