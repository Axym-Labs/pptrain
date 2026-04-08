from __future__ import annotations

from dataclasses import dataclass

from pptrain.core.presets import MechanismPreset, sequence_preset


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


def _paper_dyck_preset(bracket_types: int) -> MechanismPreset:
    return sequence_preset(
        f"paper_k{bracket_types}",
        f"Procedural-pretraining Dyck preset with k={bracket_types}.",
        sequence_count=16_032,
        eval_sequence_count=1_024,
        reference="Jiang et al. 2026",
        num_bracket_types=bracket_types,
        min_pairs=max(16, bracket_types * 2),
        max_pairs=max(128, bracket_types * 12),
        max_depth=max(16, bracket_types * 4),
        close_probability=0.45,
        max_length=2048,
    )


DYCK_PRESETS: tuple[MechanismPreset, ...] = (
    sequence_preset(
        "smoke",
        "Tiny local Dyck smoke run.",
        sequence_count=128,
        eval_sequence_count=32,
        reference="pptrain",
        num_bracket_types=8,
        min_pairs=8,
        max_pairs=48,
        max_depth=12,
        max_length=192,
    ),
    _paper_dyck_preset(8),
    _paper_dyck_preset(16),
    _paper_dyck_preset(32),
    _paper_dyck_preset(64),
)
