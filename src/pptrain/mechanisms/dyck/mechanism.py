from __future__ import annotations

from dataclasses import asdict

import numpy as np

from pptrain.core.base import DatasetBundle, Mechanism, TokenizerSpec
from pptrain.core.collator import CausalLMCollator
from pptrain.core.datasets import ListSequenceDataset
from pptrain.core.registry import register_mechanism
from pptrain.mechanisms.dyck.config import DyckConfig


class DyckMechanism(Mechanism):
    name = "dyck"

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

    def build_datasets(self, seed: int | None = None) -> DatasetBundle:
        rng = np.random.default_rng(seed)
        spec = self.tokenizer_spec()
        train_inputs, train_labels, train_depths = self._generate_examples(
            rng,
            spec,
            self.config.sequence_count,
        )
        eval_inputs, eval_labels, eval_depths = self._generate_examples(
            rng,
            spec,
            self.config.eval_sequence_count,
        )
        metadata = {
            "train_sequence_count": len(train_inputs),
            "eval_sequence_count": len(eval_inputs),
            "train_avg_depth": float(np.mean(train_depths)) if train_depths else None,
            "eval_avg_depth": float(np.mean(eval_depths)) if eval_depths else None,
            "config": asdict(self.config),
        }
        return DatasetBundle(
            train_dataset=ListSequenceDataset(train_inputs, labels=train_labels),
            eval_dataset=ListSequenceDataset(eval_inputs, labels=eval_labels),
            data_collator=CausalLMCollator(pad_token_id=spec.pad_token_id),
            metadata=metadata,
        )

    def _generate_examples(
        self,
        rng: np.random.Generator,
        spec: TokenizerSpec,
        count: int,
    ) -> tuple[list[list[int]], list[list[int]], list[int]]:
        inputs: list[list[int]] = []
        labels: list[list[int]] = []
        depths: list[int] = []
        max_pairs_bound = min(self.config.max_pairs, max(1, (self.config.max_length - 1) // 2))
        for _ in range(count):
            pair_count = int(rng.integers(self.config.min_pairs, max_pairs_bound + 1))
            sequence, peak_depth = self._sample_sequence(rng, pair_count, spec)
            inputs.append(sequence[:-1][: self.config.max_length])
            labels.append(sequence[1:][: self.config.max_length])
            depths.append(peak_depth)
        return inputs, labels, depths

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


register_mechanism("dyck", lambda config: DyckMechanism(DyckConfig(**config)))
