from __future__ import annotations

from dataclasses import dataclass, field
import string

from pptrain.core.presets import TaskPreset, sequence_preset
from pptrain.tasks.simpler_tasks.tasks import SUPPORTED_SIMPLER_TASKS


@dataclass(slots=True)
class SimplerTasksConfig:
    tasks: tuple[str, ...] = field(
        default_factory=lambda: (
            "copy",
            "reverse",
            "set",
            "duplicate",
            "deduplicate",
            "length",
            "search",
            "union",
            "intersect",
            "set_1_minus_2",
            "set_2_minus_1",
        )
    )
    alphabet: str = string.ascii_letters
    min_symbols: int = 4
    max_symbols: int = 24
    min_query_symbols: int = 1
    max_query_symbols: int = 4
    positive_search_probability: float = 0.5
    sequence_count: int = 512
    eval_sequence_count: int = 64
    max_length: int = 128


def _single_task_preset(task: str, *, sequence_count: int, max_length: int, reference: str) -> TaskPreset:
    return sequence_preset(
        f"paper_{task}_{'1m' if sequence_count == 1_000_000 else '100k'}",
        f"Single-task {task} preset from the simpler synthetic pretraining benchmark.",
        sequence_count=sequence_count,
        eval_sequence_count=10_000,
        reference=reference,
        tasks=(task,),
        min_symbols=10,
        max_symbols=220,
        min_query_symbols=1,
        max_query_symbols=4,
        max_length=max_length,
    )


def _single_task_presets() -> tuple[TaskPreset, ...]:
    reference = "Wu et al. 2022"
    base_lengths = {
        "count": 256,
        "search": 320,
        "delete": 320,
        "filter": 320,
        "get_index": 320,
        "sort": 320,
        "replace": 320,
        "replace_many": 320,
        "union": 320,
        "intersect": 320,
        "set_1_minus_2": 320,
        "set_2_minus_1": 320,
    }
    result: list[TaskPreset] = []
    for task in SUPPORTED_SIMPLER_TASKS:
        if task in {"copy", "identity", "set"}:
            continue
        result.append(
            _single_task_preset(
                task,
                sequence_count=100_000,
                max_length=base_lengths.get(task, 256),
                reference=reference,
            )
        )
    return tuple(result)


SIMPLER_TASKS_PRESETS: tuple[TaskPreset, ...] = (
    sequence_preset(
        "smoke",
        "Tiny simpler-tasks smoke run.",
        sequence_count=128,
        eval_sequence_count=32,
        reference="pptrain",
        tasks=("copy", "reverse", "set", "length", "search"),
        min_symbols=4,
        max_symbols=24,
        max_length=96,
    ),
    sequence_preset(
        "paper_copy_1m",
        "Single-task copy preset from the simpler synthetic pretraining benchmark.",
        sequence_count=1_000_000,
        eval_sequence_count=10_000,
        reference="Wu et al. 2022",
        tasks=("copy",),
        min_symbols=10,
        max_symbols=220,
        max_length=256,
    ),
    sequence_preset(
        "paper_copy_100k",
        "Single-task copy preset from the simpler synthetic pretraining benchmark.",
        sequence_count=100_000,
        eval_sequence_count=10_000,
        reference="Wu et al. 2022",
        tasks=("copy",),
        min_symbols=10,
        max_symbols=220,
        max_length=256,
    ),
    sequence_preset(
        "paper_identity_1m",
        "Single-task identity preset from the simpler synthetic pretraining benchmark.",
        sequence_count=1_000_000,
        eval_sequence_count=10_000,
        reference="Wu et al. 2022",
        tasks=("identity",),
        min_symbols=10,
        max_symbols=220,
        max_length=256,
    ),
    sequence_preset(
        "paper_identity_100k",
        "Single-task identity preset from the simpler synthetic pretraining benchmark.",
        sequence_count=100_000,
        eval_sequence_count=10_000,
        reference="Wu et al. 2022",
        tasks=("identity",),
        min_symbols=10,
        max_symbols=220,
        max_length=256,
    ),
    sequence_preset(
        "paper_set_1m",
        "Single-task set preset from the simpler synthetic pretraining benchmark.",
        sequence_count=1_000_000,
        eval_sequence_count=10_000,
        reference="Wu et al. 2022",
        tasks=("set",),
        min_symbols=10,
        max_symbols=220,
        max_length=256,
    ),
    sequence_preset(
        "paper_set_100k",
        "Single-task set preset from the simpler synthetic pretraining benchmark.",
        sequence_count=100_000,
        eval_sequence_count=10_000,
        reference="Wu et al. 2022",
        tasks=("set",),
        min_symbols=10,
        max_symbols=220,
        max_length=256,
    ),
    sequence_preset(
        "paper_unary_core_100k",
        "Core unary-task benchmark preset at the 100k scale.",
        sequence_count=100_000,
        eval_sequence_count=10_000,
        reference="Wu et al. 2022",
        tasks=("copy", "reverse", "set", "first_token", "last_token", "duplicate", "deduplicate", "length"),
        min_symbols=10,
        max_symbols=220,
        max_length=256,
    ),
    sequence_preset(
        "paper_unary_core_1m",
        "Core unary-task benchmark preset at the 1M scale.",
        sequence_count=1_000_000,
        eval_sequence_count=10_000,
        reference="Wu et al. 2022",
        tasks=("copy", "reverse", "set", "first_token", "last_token", "duplicate", "deduplicate", "length"),
        min_symbols=10,
        max_symbols=220,
        max_length=256,
    ),
    sequence_preset(
        "paper_binary_1m",
        "Binary-task benchmark preset at the 1M scale.",
        sequence_count=1_000_000,
        eval_sequence_count=10_000,
        reference="Wu et al. 2022",
        tasks=(
            "count",
            "delete",
            "filter",
            "get_index",
            "search",
            "sort",
            "replace",
            "replace_many",
            "union",
            "intersect",
            "set_1_minus_2",
            "set_2_minus_1",
        ),
        min_symbols=10,
        max_symbols=220,
        min_query_symbols=1,
        max_query_symbols=4,
        max_length=320,
    ),
    sequence_preset(
        "paper_set_ops_1m",
        "Set-operation subset of the binary simpler-task benchmark at the 1M scale.",
        sequence_count=1_000_000,
        eval_sequence_count=10_000,
        reference="Wu et al. 2022",
        tasks=("search", "union", "intersect", "set_1_minus_2", "set_2_minus_1"),
        min_symbols=10,
        max_symbols=220,
        min_query_symbols=1,
        max_query_symbols=4,
        max_length=320,
    ),
    *_single_task_presets(),
)
