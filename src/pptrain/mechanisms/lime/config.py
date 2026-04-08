from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class LIMEConfig:
    modes: tuple[str, ...] = field(default_factory=lambda: ("induct", "deduct", "abduct"))
    upper_symbols: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lower_symbols: str = "abcdefghijklmnopqrstuvwxyz"
    math_symbols: str = "+-=*/()"
    min_variables: int = 3
    max_variables: int = 5
    min_pattern_length: int = 5
    max_pattern_length: int = 18
    min_substitution_length: int = 2
    max_substitution_length: int = 6
    sequence_count: int = 512
    eval_sequence_count: int = 64
    max_length: int = 192
