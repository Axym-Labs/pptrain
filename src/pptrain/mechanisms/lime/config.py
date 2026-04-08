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


def _benchmark_preset(
    name: str,
    description: str,
    *,
    sequence_count: int,
    modes: tuple[str, ...],
    reference: str,
) -> MechanismPreset:
    return sequence_preset(
        name,
        description,
        sequence_count=sequence_count,
        eval_sequence_count=10_000 if sequence_count <= 1_000_000 else 1_000,
        reference=reference,
        modes=modes,
        max_length=256,
    )


def _single_mode_presets() -> tuple[MechanismPreset, ...]:
    scales = (
        ("paper_individual_100k", 100_000),
        ("paper_individual_1m", 1_000_000),
    )
    descriptions = {
        "induct": "Induct-only LIME preset",
        "deduct": "Deduct-only LIME preset",
        "abduct": "Abduct-only LIME preset",
    }
    result: list[MechanismPreset] = []
    for prefix, sequence_count in scales:
        for mode in ("induct", "deduct", "abduct"):
            result.append(
                _benchmark_preset(
                    f"{prefix}_{mode}",
                    f"{descriptions[mode]} at the {sequence_count:,}-example scale.",
                    sequence_count=sequence_count,
                    modes=(mode,),
                    reference="Wu et al. 2021 / Wu et al. 2022",
                )
            )
    return tuple(result)


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
    _benchmark_preset(
        "paper_benchmark_100k",
        "LIME benchmark preset at the 100k scale used in the simpler synthetic-task comparison.",
        sequence_count=100_000,
        modes=("induct", "deduct", "abduct"),
        reference="Wu et al. 2022",
    ),
    _benchmark_preset(
        "paper_benchmark_1m",
        "LIME benchmark preset at the 1M scale used in the simpler synthetic-task comparison.",
        sequence_count=1_000_000,
        modes=("induct", "deduct", "abduct"),
        reference="Wu et al. 2022",
    ),
    _benchmark_preset(
        "paper_mixed_5m",
        "Original LIME mixed-task preset with a 5M-example budget.",
        sequence_count=5_000_000,
        modes=("induct", "deduct", "abduct"),
        reference="Wu et al. 2021",
    ),
    *_single_mode_presets(),
)
