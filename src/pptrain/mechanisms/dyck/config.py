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
    sequence_preset(
        "paper_k64",
        "Procedural-pretraining Dyck preset with k=64 and long context.",
        sequence_count=16_032,
        eval_sequence_count=1_024,
        reference="Jiang et al. 2026",
        num_bracket_types=64,
        min_pairs=128,
        max_pairs=768,
        max_depth=256,
        close_probability=0.45,
        max_length=2048,
    ),
)
