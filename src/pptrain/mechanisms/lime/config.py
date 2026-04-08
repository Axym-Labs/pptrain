from __future__ import annotations

from dataclasses import dataclass, field

from pptrain.core.presets import MechanismPreset, sequence_preset


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


LIME_PRESETS: tuple[MechanismPreset, ...] = (
    sequence_preset(
        "smoke",
        "Tiny LIME smoke run.",
        sequence_count=128,
        eval_sequence_count=32,
        reference="pptrain",
        modes=("induct", "deduct", "abduct"),
        max_length=128,
    ),
    sequence_preset(
        "paper_benchmark_100k",
        "LIME benchmark preset at the 100k scale used in the simpler synthetic-task comparison.",
        sequence_count=100_000,
        eval_sequence_count=10_000,
        reference="Wu et al. 2022",
        modes=("induct", "deduct", "abduct"),
        max_length=256,
    ),
    sequence_preset(
        "paper_benchmark_1m",
        "LIME benchmark preset at the 1M scale used in the simpler synthetic-task comparison.",
        sequence_count=1_000_000,
        eval_sequence_count=10_000,
        reference="Wu et al. 2022",
        modes=("induct", "deduct", "abduct"),
        max_length=256,
    ),
    sequence_preset(
        "paper_mixed_5m",
        "Original LIME mixed-task preset with a 5M-example budget.",
        sequence_count=5_000_000,
        eval_sequence_count=1_000,
        reference="Wu et al. 2021",
        modes=("induct", "deduct", "abduct"),
        max_length=256,
    ),
)
