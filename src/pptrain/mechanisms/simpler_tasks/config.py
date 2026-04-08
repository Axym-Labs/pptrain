from __future__ import annotations

from dataclasses import dataclass, field
import string

from pptrain.core.presets import MechanismPreset, sequence_preset


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


SIMPLER_TASKS_PRESETS: tuple[MechanismPreset, ...] = (
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
)
