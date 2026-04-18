from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class TaskPreset:
    name: str
    description: str
    config: dict[str, Any]
    reference: str = ""


def sequence_preset(
    name: str,
    description: str,
    *,
    sequence_count: int,
    eval_sequence_count: int,
    reference: str = "",
    **config: Any,
) -> TaskPreset:
    payload = dict(config)
    payload["sequence_count"] = sequence_count
    payload["eval_sequence_count"] = eval_sequence_count
    return TaskPreset(
        name=name,
        description=description,
        config=payload,
        reference=reference,
    )


def merge_preset_config(
    preset: TaskPreset,
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    resolved = dict(preset.config)
    resolved.update(overrides)
    return resolved


# Backward-compatible alias during the terminology transition.
MechanismPreset = TaskPreset
