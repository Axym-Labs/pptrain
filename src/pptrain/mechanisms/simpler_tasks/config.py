from __future__ import annotations

from dataclasses import dataclass, field
import string


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
