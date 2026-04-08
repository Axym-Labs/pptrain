from __future__ import annotations

from dataclasses import asdict

import numpy as np

from pptrain.core.base import DatasetBundle, Mechanism, TokenizerSpec
from pptrain.core.collator import CausalLMCollator
from pptrain.core.datasets import ListSequenceDataset
from pptrain.core.registry import register_mechanism
from pptrain.mechanisms.nca.config import NCAConfig, NCA_PRESETS
from pptrain.mechanisms.nca.generator import (
    create_training_example,
    generate_rule_pool,
    patchify_trajectory,
    required_frame_count,
    rollout_rule,
)


class NCAMechanism(Mechanism):
    name = "nca"
    description = "Neural cellular automata trajectories serialized into patch-token sequences."

    def __init__(self, config: NCAConfig) -> None:
        super().__init__(config)
        if self.config.grid_size % self.config.patch_size != 0:
            raise ValueError("grid_size must be divisible by patch_size.")
        if not 0.0 <= self.config.complexity_min <= self.config.complexity_max:
            raise ValueError("complexity_min must be <= complexity_max and both must be non-negative.")
        if self.config.min_frames < 0:
            raise ValueError("min_frames must be non-negative.")

    def tokenizer_spec(self) -> TokenizerSpec:
        patch_vocab_size = self.config.num_states ** (self.config.patch_size ** 2)
        return TokenizerSpec(
            vocab_size=patch_vocab_size + 3,
            pad_token_id=patch_vocab_size + 2,
            bos_token_id=patch_vocab_size,
            eos_token_id=patch_vocab_size + 1,
        )

    def build_datasets(self, seed: int | None = None) -> DatasetBundle:
        rng = np.random.default_rng(seed)
        spec = self.tokenizer_spec()
        frame_count = required_frame_count(
            max_length=self.config.max_length,
            grid_size=self.config.grid_size,
            patch_size=self.config.patch_size,
        )
        frame_token_length = (self.config.grid_size // self.config.patch_size) ** 2 + 2
        train_inputs, train_labels, train_ratios = self._generate_examples(
            rng,
            spec,
            sequence_count=self.config.sequence_count,
            rule_count=self.config.rule_count or self.config.sequence_count,
            frame_count=frame_count,
            frame_token_length=frame_token_length,
        )
        eval_inputs, eval_labels, eval_ratios = self._generate_examples(
            rng,
            spec,
            sequence_count=self.config.eval_sequence_count,
            rule_count=self.config.eval_rule_count or self.config.eval_sequence_count,
            frame_count=frame_count,
            frame_token_length=frame_token_length,
        )
        metadata = {
            "train_sequence_count": len(train_inputs),
            "eval_sequence_count": len(eval_inputs),
            "train_rule_count": self.config.rule_count or self.config.sequence_count,
            "eval_rule_count": self.config.eval_rule_count or self.config.eval_sequence_count,
            "frame_count": frame_count,
            "frame_token_length": frame_token_length,
            "train_avg_compression_ratio": float(np.mean(train_ratios)) if train_ratios else None,
            "eval_avg_compression_ratio": float(np.mean(eval_ratios)) if eval_ratios else None,
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
        *,
        sequence_count: int,
        rule_count: int,
        frame_count: int,
        frame_token_length: int,
    ) -> tuple[list[list[int]], list[list[int]], list[float]]:
        rules, ratios = generate_rule_pool(
            rng=rng,
            rule_count=rule_count,
            sequence_count=sequence_count,
            num_states=self.config.num_states,
            hidden_dim=self.config.hidden_dim,
            perception_dim=self.config.perception_dim,
            grid_size=self.config.grid_size,
            patch_size=self.config.patch_size,
            complexity_probe_frames=self.config.complexity_probe_frames,
            rollout_stride=self.config.rollout_stride,
            init_rollout_steps=self.config.init_rollout_steps,
            complexity_min=self.config.complexity_min,
            complexity_max=self.config.complexity_max,
            max_rule_attempts_per_sequence=self.config.max_rule_attempts_per_sequence,
            identity_bias=self.config.identity_bias,
            temperature=self.config.temperature,
        )
        selected_rules = self._repeat_rules(rules, sequence_count)
        input_sequences: list[list[int]] = []
        label_sequences: list[list[int]] = []
        for rule in selected_rules:
            trajectory = rollout_rule(
                rng=rng,
                rule=rule,
                grid_size=self.config.grid_size,
                num_states=self.config.num_states,
                frame_count=frame_count,
                rollout_stride=self.config.rollout_stride,
                init_rollout_steps=self.config.init_rollout_steps,
                identity_bias=self.config.identity_bias,
                temperature=self.config.temperature,
            )
            tokens = patchify_trajectory(
                trajectory=trajectory,
                num_states=self.config.num_states,
                patch_size=self.config.patch_size,
                start_token_id=spec.bos_token_id or 0,
                end_token_id=spec.eos_token_id or 1,
            )
            inputs, labels = create_training_example(
                tokens,
                max_length=self.config.max_length,
                frame_token_length=frame_token_length,
                min_frames=self.config.min_frames,
            )
            input_sequences.append(inputs)
            label_sequences.append(labels)
        return input_sequences, label_sequences, ratios[:sequence_count]

    @staticmethod
    def _repeat_rules(rules: list[object], sequence_count: int) -> list[object]:
        if not rules:
            return []
        repeated = [rules[idx % len(rules)] for idx in range(sequence_count)]
        return repeated


register_mechanism(
    "nca",
    lambda config: NCAMechanism(NCAConfig(**config)),
    description=NCAMechanism.description,
    presets=NCA_PRESETS,
)
