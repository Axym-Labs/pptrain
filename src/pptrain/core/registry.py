from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pptrain.core.base import Mechanism
from pptrain.core.presets import MechanismPreset, merge_preset_config

MechanismFactory = Callable[[dict[str, Any]], Mechanism]


@dataclass(frozen=True, slots=True)
class RegisteredMechanism:
    name: str
    description: str
    presets: tuple[MechanismPreset, ...] = ()


@dataclass(slots=True)
class _MechanismEntry:
    factory: MechanismFactory
    description: str
    presets: dict[str, MechanismPreset]


_MECHANISMS: dict[str, _MechanismEntry] = {}


def _ensure_mechanisms_loaded() -> None:
    if not _MECHANISMS:
        import pptrain.mechanisms  # noqa: F401


def register_mechanism(
    name: str,
    factory: MechanismFactory,
    *,
    description: str = "",
    presets: tuple[MechanismPreset, ...] = (),
) -> None:
    _MECHANISMS[name] = _MechanismEntry(
        factory=factory,
        description=description,
        presets={preset.name: preset for preset in presets},
    )


def create_mechanism(name: str, config: dict[str, Any]) -> Mechanism:
    _ensure_mechanisms_loaded()
    if name not in _MECHANISMS:
        raise KeyError(f"Unknown mechanism '{name}'. Registered: {sorted(_MECHANISMS)}")
    entry = _MECHANISMS[name]
    resolved_config = dict(config)
    preset_name = resolved_config.pop("preset", None)
    if preset_name is not None:
        if preset_name not in entry.presets:
            raise KeyError(
                f"Unknown preset '{preset_name}' for mechanism '{name}'. "
                f"Registered: {sorted(entry.presets)}"
            )
        resolved_config = merge_preset_config(entry.presets[preset_name], resolved_config)
    return entry.factory(resolved_config)


def registered_mechanisms() -> tuple[RegisteredMechanism, ...]:
    _ensure_mechanisms_loaded()
    return tuple(
        RegisteredMechanism(
            name=name,
            description=entry.description,
            presets=tuple(entry.presets[preset_name] for preset_name in sorted(entry.presets)),
        )
        for name, entry in sorted(_MECHANISMS.items())
    )


def registered_presets(name: str) -> tuple[MechanismPreset, ...]:
    _ensure_mechanisms_loaded()
    if name not in _MECHANISMS:
        raise KeyError(f"Unknown mechanism '{name}'. Registered: {sorted(_MECHANISMS)}")
    entry = _MECHANISMS[name]
    return tuple(entry.presets[preset_name] for preset_name in sorted(entry.presets))
