from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DyckConfig:
    num_bracket_types: int = 4
    min_pairs: int = 8
    max_pairs: int = 64
    max_depth: int = 12
    close_probability: float = 0.45
    sequence_count: int = 512
    eval_sequence_count: int = 64
    max_length: int = 128

