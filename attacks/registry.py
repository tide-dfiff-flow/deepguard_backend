"""Attack registry and factory."""
from __future__ import annotations

from .base import BaseAttack
from .codec_attack import CodecAttack
from .noise_attack import NoiseAttack
from .filter_attack import FilterAttack
from .channel_attack import ChannelAttack
from .combo_attack import ComboAttack


ATTACK_REGISTRY = {
    "A_codec": CodecAttack,
    "A_noise": NoiseAttack,
    "A_filter": FilterAttack,
    "A_channel": ChannelAttack,
    "A_hidden": ComboAttack,
}


def list_attacks() -> list[str]:
    return list(ATTACK_REGISTRY.keys())


def build_attack(name: str, **params) -> BaseAttack:
    if name not in ATTACK_REGISTRY:
        raise KeyError(f"Unknown attack '{name}'. Available: {list_attacks()}")
    return ATTACK_REGISTRY[name](**params)


def build_all_attacks(overrides: dict | None = None) -> list[BaseAttack]:
    """Instantiate the full fixed attack pool (spec §10.1 A0)."""
    overrides = overrides or {}
    attacks = []
    for name, cls in ATTACK_REGISTRY.items():
        attacks.append(cls(**overrides.get(name, {})))
    return attacks
