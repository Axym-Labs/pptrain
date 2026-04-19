from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pptrain.core.base import Task
from pptrain.core.presets import TaskPreset, merge_preset_config

TaskFactory = Callable[[dict[str, Any]], Task]


@dataclass(frozen=True, slots=True)
class RegisteredTask:
    name: str
    description: str
    presets: tuple[TaskPreset, ...] = ()


@dataclass(slots=True)
class _TaskEntry:
    factory: TaskFactory
    description: str
    presets: dict[str, TaskPreset]


_TASKS: dict[str, _TaskEntry] = {}


def _ensure_tasks_loaded() -> None:
    if not _TASKS:
        import pptrain.tasks  # noqa: F401


def register_task(
    name: str,
    factory: TaskFactory,
    *,
    description: str = "",
    presets: tuple[TaskPreset, ...] = (),
) -> None:
    _TASKS[name] = _TaskEntry(
        factory=factory,
        description=description,
        presets={preset.name: preset for preset in presets},
    )


def create_task(name: str, config: dict[str, Any]) -> Task:
    _ensure_tasks_loaded()
    if name not in _TASKS:
        raise KeyError(f"Unknown task '{name}'. Registered: {sorted(_TASKS)}")
    entry = _TASKS[name]
    resolved_config = dict(config)
    preset_name = resolved_config.pop("preset", None)
    if preset_name is not None:
        if preset_name not in entry.presets:
            raise KeyError(
                f"Unknown preset '{preset_name}' for task '{name}'. "
                f"Registered: {sorted(entry.presets)}"
            )
        resolved_config = merge_preset_config(entry.presets[preset_name], resolved_config)
    return entry.factory(resolved_config)


def registered_tasks() -> tuple[RegisteredTask, ...]:
    _ensure_tasks_loaded()
    return tuple(
        RegisteredTask(
            name=name,
            description=entry.description,
            presets=tuple(entry.presets[preset_name] for preset_name in sorted(entry.presets)),
        )
        for name, entry in sorted(_TASKS.items())
    )


def registered_presets(name: str) -> tuple[TaskPreset, ...]:
    _ensure_tasks_loaded()
    if name not in _TASKS:
        raise KeyError(f"Unknown task '{name}'. Registered: {sorted(_TASKS)}")
    entry = _TASKS[name]
    return tuple(entry.presets[preset_name] for preset_name in sorted(entry.presets))
