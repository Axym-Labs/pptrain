from __future__ import annotations

from dataclasses import dataclass, field

from pptrain.core.presets import MechanismPreset, sequence_preset


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


PROCEDURAL_PRESETS: tuple[MechanismPreset, ...] = (
    sequence_preset(
        "smoke",
        "Tiny procedural smoke run.",
        sequence_count=128,
        eval_sequence_count=32,
        reference="pptrain",
        tasks=("copy", "reverse", "sort"),
        min_symbol_length=4,
        max_symbol_length=16,
        max_length=64,
    ),
    sequence_preset(
        "paper_identity_len64",
        "Procedural-pretraining identity preset at sequence length 64.",
        sequence_count=160_064,
        eval_sequence_count=4_096,
        reference="Jiang et al. 2026",
        tasks=("identity",),
        min_symbol_length=4,
        max_symbol_length=24,
        max_length=64,
    ),
    sequence_preset(
        "paper_reverse_len64",
        "Procedural-pretraining reverse preset at sequence length 64.",
        sequence_count=160_064,
        eval_sequence_count=4_096,
        reference="Jiang et al. 2026",
        tasks=("reverse",),
        min_symbol_length=4,
        max_symbol_length=24,
        max_length=64,
    ),
    sequence_preset(
        "paper_sort_len64",
        "Procedural-pretraining sort preset at sequence length 64.",
        sequence_count=160_064,
        eval_sequence_count=4_096,
        reference="Jiang et al. 2026",
        tasks=("sort",),
        min_symbol_length=4,
        max_symbol_length=24,
        max_length=64,
    ),
    sequence_preset(
        "paper_set_len64",
        "Procedural-pretraining set preset at sequence length 64.",
        sequence_count=160_064,
        eval_sequence_count=4_096,
        reference="Jiang et al. 2026",
        tasks=("set",),
        min_symbol_length=4,
        max_symbol_length=24,
        max_length=64,
    ),
    sequence_preset(
        "paper_union_len64",
        "Procedural-pretraining union preset at sequence length 64.",
        sequence_count=160_064,
        eval_sequence_count=4_096,
        reference="Jiang et al. 2026",
        tasks=("union",),
        min_symbol_length=4,
        max_symbol_length=24,
        max_length=64,
    ),
    sequence_preset(
        "paper_delete_len64",
        "Procedural-pretraining delete preset at sequence length 64.",
        sequence_count=160_064,
        eval_sequence_count=4_096,
        reference="Jiang et al. 2026",
        tasks=("delete",),
        min_symbol_length=4,
        max_symbol_length=24,
        max_length=64,
    ),
)
