from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pptrain.core.base import Mechanism

MechanismFactory = Callable[[dict[str, Any]], Mechanism]


@dataclass(frozen=True, slots=True)
class RegisteredMechanism:
    name: str
    description: str


@dataclass(slots=True)
class _MechanismEntry:
    factory: MechanismFactory
    description: str


_MECHANISMS: dict[str, _MechanismEntry] = {}


def _ensure_mechanisms_loaded() -> None:
    if not _MECHANISMS:
        import pptrain.mechanisms  # noqa: F401


def register_mechanism(
    name: str,
    factory: MechanismFactory,
    *,
    description: str = "",
) -> None:
    _MECHANISMS[name] = _MechanismEntry(factory=factory, description=description)


def create_mechanism(name: str, config: dict[str, Any]) -> Mechanism:
    _ensure_mechanisms_loaded()
    if name not in _MECHANISMS:
        raise KeyError(f"Unknown mechanism '{name}'. Registered: {sorted(_MECHANISMS)}")
    return _MECHANISMS[name].factory(config)


def registered_mechanisms() -> tuple[RegisteredMechanism, ...]:
    _ensure_mechanisms_loaded()
    return tuple(
        RegisteredMechanism(name=name, description=entry.description)
        for name, entry in sorted(_MECHANISMS.items())
    )
