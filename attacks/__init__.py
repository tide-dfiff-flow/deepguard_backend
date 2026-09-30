"""Attack methods for the DeepGuard-Audio backend."""
from .base import BaseAttack  # noqa: F401
from .codec_attack import CodecAttack  # noqa: F401
from .noise_attack import NoiseAttack  # noqa: F401
from .filter_attack import FilterAttack  # noqa: F401
from .channel_attack import ChannelAttack  # noqa: F401
from .combo_attack import ComboAttack  # noqa: F401
from .registry import (  # noqa: F401
    ATTACK_REGISTRY, list_attacks, build_attack, build_all_attacks,
)
