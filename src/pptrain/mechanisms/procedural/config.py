from __future__ import annotations

from dataclasses import dataclass, field


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

