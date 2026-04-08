from __future__ import annotations

from typing import Any, Callable

from pptrain.core.base import Mechanism

MechanismFactory = Callable[[dict[str, Any]], Mechanism]

_MECHANISMS: dict[str, MechanismFactory] = {}


def register_mechanism(name: str, factory: MechanismFactory) -> None:
    _MECHANISMS[name] = factory


def create_mechanism(name: str, config: dict[str, Any]) -> Mechanism:
    if not _MECHANISMS:
        import pptrain.mechanisms  # noqa: F401
    if name not in _MECHANISMS:
        raise KeyError(f"Unknown mechanism '{name}'. Registered: {sorted(_MECHANISMS)}")
    return _MECHANISMS[name](config)
