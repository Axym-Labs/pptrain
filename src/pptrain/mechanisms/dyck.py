from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pptrain.core.base import ExecutedSymbolicTask, SymbolicTask, SymbolicTaskMechanism, TokenizerSpec
from pptrain.core.presets import MechanismPreset, sequence_preset
from pptrain.core.registry import register_mechanism


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


@dataclass(slots=True)
class DyckProgram:
    actions: list[tuple[str, int]]
    peak_depth: int


class DyckMechanism(SymbolicTaskMechanism):
    name = "dyck"
    description = "Balanced-bracket language generation with controllable depth and nesting."
    task_group_metadata_key = None

    def __init__(self, config: DyckConfig) -> None:
        super().__init__(config)
        if self.config.num_bracket_types < 1:
            raise ValueError("num_bracket_types must be positive.")
        if self.config.min_pairs < 1 or self.config.max_pairs < self.config.min_pairs:
            raise ValueError("min_pairs and max_pairs must define a valid positive range.")
        if self.config.max_depth < 1:
            raise ValueError("max_depth must be positive.")
        if not 0.0 < self.config.close_probability < 1.0:
            raise ValueError("close_probability must be in (0, 1).")
        if self.config.min_pairs > max(1, (self.config.max_length - 1) // 2):
            raise ValueError("max_length is too small for the configured min_pairs.")

    def tokenizer_spec(self) -> TokenizerSpec:
        vocab_offset = 2 * self.config.num_bracket_types
        return TokenizerSpec(
            vocab_size=vocab_offset + 3,
            pad_token_id=vocab_offset + 2,
            bos_token_id=vocab_offset,
            eos_token_id=vocab_offset + 1,
        )

    def sample_task(self, rng: np.random.Generator) -> SymbolicTask:
        max_pairs_bound = min(self.config.max_pairs, max(1, (self.config.max_length - 1) // 2))
        pair_count = int(rng.integers(self.config.min_pairs, max_pairs_bound + 1))
        return SymbolicTask(name="dyck", payload=self._sample_program(rng, pair_count))

    def execute_task(self, task: SymbolicTask) -> ExecutedSymbolicTask:
        program = task.payload
        sequence: list[int] = []
        for action, bracket_type in program.actions:
            if action == "open":
                sequence.append(bracket_type)
            else:
                sequence.append(self.config.num_bracket_types + bracket_type)
        return ExecutedSymbolicTask(
            name=task.name,
            payload=sequence,
            metadata={"peak_depth": program.peak_depth},
        )

    def serialize_task(self, executed: ExecutedSymbolicTask, spec: TokenizerSpec) -> list[int]:
        return [spec.bos_token_id or 0, *executed.payload, spec.eos_token_id or 1]

    def numeric_metadata_fields(self) -> tuple[str, ...]:
        return ("peak_depth",)

    def _sample_program(self, rng: np.random.Generator, pair_count: int) -> DyckProgram:
        actions: list[tuple[str, int]] = []
        stack: list[int] = []
        opens_used = 0
        peak_depth = 0
        while opens_used < pair_count or stack:
            can_open = opens_used < pair_count and len(stack) < self.config.max_depth
            should_close = bool(stack) and (not can_open or rng.random() < self.config.close_probability)
            if should_close:
                bracket_type = stack.pop()
                actions.append(("close", bracket_type))
            else:
                bracket_type = int(rng.integers(0, self.config.num_bracket_types))
                stack.append(bracket_type)
                opens_used += 1
                peak_depth = max(peak_depth, len(stack))
                actions.append(("open", bracket_type))
        return DyckProgram(actions=actions, peak_depth=peak_depth)


register_mechanism(
    "dyck",
    lambda config: DyckMechanism(DyckConfig(**config)),
    description=DyckMechanism.description,
    presets=DYCK_PRESETS,
)
