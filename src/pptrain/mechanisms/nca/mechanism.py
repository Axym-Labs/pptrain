from __future__ import annotations

from dataclasses import asdict

import numpy as np

from pptrain.core.base import DatasetBundle, Mechanism, TokenizerSpec
from pptrain.core.collator import CausalLMCollator
from pptrain.core.datasets import ListSequenceDataset
from pptrain.core.registry import register_mechanism
from pptrain.mechanisms.nca.config import NCAConfig
from pptrain.mechanisms.nca.generator import (
    compression_ratio,
    patchify_trajectory,
    rollout_rule,
    sample_rule,
)


class NCAMechanism(Mechanism):
    name = "nca"

    def __init__(self, config: NCAConfig) -> None:
        super().__init__(config)
        if self.config.grid_size % self.config.patch_size != 0:
            raise ValueError("grid_size must be divisible by patch_size.")
        if not 0.0 <= self.config.complexity_min <= self.config.complexity_max:
            raise ValueError("complexity_min must be <= complexity_max and both must be non-negative.")

    def tokenizer_spec(self) -> TokenizerSpec:
        patch_vocab_size = self.config.num_states ** (self.config.patch_size ** 2)
        return TokenizerSpec(
            vocab_size=patch_vocab_size + 4,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            extra_token_ids={"time": 3},
        )

    def build_datasets(self, seed: int | None = None) -> DatasetBundle:
        rng = np.random.default_rng(seed)
        spec = self.tokenizer_spec()
        train_sequences, train_ratios = self._generate_sequences(rng, self.config.sequence_count, spec)
        eval_sequences, eval_ratios = self._generate_sequences(
            rng,
            self.config.eval_sequence_count,
            spec,
        )
        metadata = {
            "train_sequence_count": len(train_sequences),
            "eval_sequence_count": len(eval_sequences),
            "train_avg_compression_ratio": float(np.mean(train_ratios)) if train_ratios else None,
            "eval_avg_compression_ratio": float(np.mean(eval_ratios)) if eval_ratios else None,
            "config": asdict(self.config),
        }
        return DatasetBundle(
            train_dataset=ListSequenceDataset(train_sequences),
            eval_dataset=ListSequenceDataset(eval_sequences),
            data_collator=CausalLMCollator(pad_token_id=spec.pad_token_id),
            metadata=metadata,
        )

    def _generate_sequences(
        self,
        rng: np.random.Generator,
        count: int,
        spec: TokenizerSpec,
    ) -> tuple[list[list[int]], list[float]]:
        sequences: list[list[int]] = []
        ratios: list[float] = []
        attempts = 0
        max_attempts = max(count * self.config.max_rule_attempts_per_sequence, count)
        while len(sequences) < count and attempts < max_attempts:
            attempts += 1
            rule = sample_rule(rng, self.config.num_states, self.config.hidden_dim)
            trajectory = rollout_rule(
                rng=rng,
                rule=rule,
                grid_size=self.config.grid_size,
                num_states=self.config.num_states,
                rollout_steps=self.config.rollout_steps,
                stochasticity=self.config.stochasticity,
            )
            ratio = compression_ratio(trajectory)
            if not self.config.complexity_min <= ratio <= self.config.complexity_max:
                continue
            sequence = patchify_trajectory(
                trajectory=trajectory,
                num_states=self.config.num_states,
                patch_size=self.config.patch_size,
                bos_token_id=spec.bos_token_id or 1,
                eos_token_id=spec.eos_token_id or 2,
                time_token_id=spec.extra_token_ids["time"],
                offset=4,
                max_length=self.config.max_length,
            )
            sequences.append(sequence)
            ratios.append(ratio)
        if len(sequences) < count:
            raise RuntimeError(
                "Failed to generate enough NCA sequences in the requested complexity band. "
                "Relax the band or increase max_rule_attempts_per_sequence."
            )
        return sequences, ratios


register_mechanism("nca", lambda config: NCAMechanism(NCAConfig(**config)))
