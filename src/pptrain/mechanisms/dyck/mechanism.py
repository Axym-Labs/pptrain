from __future__ import annotations

import numpy as np

from pptrain.core.base import TokenSequenceMechanism, TokenizerSpec
from pptrain.core.registry import register_mechanism
from pptrain.mechanisms.dyck.config import DYCK_PRESETS, DyckConfig


class DyckMechanism(TokenSequenceMechanism):
    name = "dyck"
    description = "Balanced-bracket language generation with controllable depth and nesting."

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

    def sample_example(
        self,
        rng: np.random.Generator,
        spec: TokenizerSpec,
    ) -> tuple[list[int], dict[str, int]]:
        max_pairs_bound = min(self.config.max_pairs, max(1, (self.config.max_length - 1) // 2))
        pair_count = int(rng.integers(self.config.min_pairs, max_pairs_bound + 1))
        sequence, peak_depth = self._sample_sequence(rng, pair_count, spec)
        return sequence, {"peak_depth": peak_depth}

    def _split_metadata(self, split: str, items: list[dict[str, int]]) -> dict[str, float | None]:
        depths = [item["peak_depth"] for item in items]
        return {f"{split}_avg_depth": float(np.mean(depths)) if depths else None}

    def _sample_sequence(
        self,
        rng: np.random.Generator,
        pair_count: int,
        spec: TokenizerSpec,
    ) -> tuple[list[int], int]:
        sequence = [spec.bos_token_id or 0]
        stack: list[int] = []
        opens_used = 0
        peak_depth = 0
        while opens_used < pair_count or stack:
            can_open = opens_used < pair_count and len(stack) < self.config.max_depth
            should_close = bool(stack) and (not can_open or rng.random() < self.config.close_probability)
            if should_close:
                bracket_type = stack.pop()
                sequence.append(self.config.num_bracket_types + bracket_type)
            else:
                bracket_type = int(rng.integers(0, self.config.num_bracket_types))
                stack.append(bracket_type)
                opens_used += 1
                peak_depth = max(peak_depth, len(stack))
                sequence.append(bracket_type)
        sequence.append(spec.eos_token_id or 1)
        return sequence, peak_depth


register_mechanism(
    "dyck",
    lambda config: DyckMechanism(DyckConfig(**config)),
    description=DyckMechanism.description,
    presets=DYCK_PRESETS,
)
