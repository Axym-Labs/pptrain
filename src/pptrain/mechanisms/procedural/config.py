from __future__ import annotations

from dataclasses import dataclass, field

from pptrain.core.presets import MechanismPreset, sequence_preset


@dataclass(slots=True)
class ProceduralConfig:
    tasks: tuple[str, ...] = field(default_factory=lambda: ("copy", "reverse", "sort", "addition"))
    min_symbol_length: int = 4
    max_symbol_length: int = 24
    alphabet: str = "abcdefghijklmnopqrstuvwxyz"
    max_number: int = 9999
    sequence_count: int = 512
    eval_sequence_count: int = 64
    max_length: int = 128


_PROCEDURAL_REFERENCE = "Jiang et al. 2026"
_PROCEDURAL_LENGTHS = (16, 32, 64)
_PROCEDURAL_TASKS = ("identity", "reverse", "sort", "set", "union", "delete")


def _paper_task_preset(task: str, *, max_length: int) -> MechanismPreset:
    return sequence_preset(
        f"paper_{task}_len{max_length}",
        f"Procedural-pretraining {task} preset at sequence length {max_length}.",
        sequence_count=160_064,
        eval_sequence_count=4_096,
        reference=_PROCEDURAL_REFERENCE,
        tasks=(task,),
        min_symbol_length=4,
        max_symbol_length=max(8, max_length // 2),
        max_length=max_length,
    )


def _paper_task_presets() -> tuple[MechanismPreset, ...]:
    return tuple(_paper_task_preset(task, max_length=max_length) for max_length in _PROCEDURAL_LENGTHS for task in _PROCEDURAL_TASKS)


PROCEDURAL_PRESETS: tuple[MechanismPreset, ...] = (
    sequence_preset(
        "smoke",
        "Tiny procedural smoke run.",
        sequence_count=128,
        eval_sequence_count=32,
        reference="pptrain",
        tasks=("copy", "reverse", "sort"),
        min_symbol_length=4,
        max_symbol_length=16,
        max_length=64,
    ),
    *_paper_task_presets(),
)
