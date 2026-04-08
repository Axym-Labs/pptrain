from __future__ import annotations

from collections import Counter
from dataclasses import asdict

import numpy as np

from pptrain.core.base import DatasetBundle, Mechanism, TokenizerSpec
from pptrain.core.collator import CausalLMCollator
from pptrain.core.datasets import ListSequenceDataset
from pptrain.core.registry import register_mechanism
from pptrain.mechanisms.procedural.config import ProceduralConfig

BASE_CHARSET = "abcdefghijklmnopqrstuvwxyz0123456789:+=>,;|-_ "


class ProceduralMechanism(Mechanism):
    name = "procedural"

    def __init__(self, config: ProceduralConfig) -> None:
        super().__init__(config)
        if self.config.min_symbol_length < 1 or self.config.max_symbol_length < self.config.min_symbol_length:
            raise ValueError("min_symbol_length and max_symbol_length must define a valid positive range.")
        if self.config.max_number < 1:
            raise ValueError("max_number must be positive.")
        if not self.config.tasks:
            raise ValueError("At least one task must be configured.")
        unknown_tasks = sorted(set(self.config.tasks) - {"copy", "reverse", "sort", "addition"})
        if unknown_tasks:
            raise ValueError(f"Unsupported procedural tasks: {unknown_tasks}")
        if not set(self.config.alphabet).issubset(set(BASE_CHARSET)):
            raise ValueError("alphabet contains unsupported characters.")

    def tokenizer_spec(self) -> TokenizerSpec:
        vocab_size = len(BASE_CHARSET) + 3
        return TokenizerSpec(
            vocab_size=vocab_size,
            pad_token_id=len(BASE_CHARSET) + 2,
            bos_token_id=len(BASE_CHARSET),
            eos_token_id=len(BASE_CHARSET) + 1,
        )

    def build_datasets(self, seed: int | None = None) -> DatasetBundle:
        rng = np.random.default_rng(seed)
        spec = self.tokenizer_spec()
        train_inputs, train_labels, train_counts = self._generate_examples(rng, spec, self.config.sequence_count)
        eval_inputs, eval_labels, eval_counts = self._generate_examples(rng, spec, self.config.eval_sequence_count)
        metadata = {
            "train_sequence_count": len(train_inputs),
            "eval_sequence_count": len(eval_inputs),
            "train_task_counts": dict(train_counts),
            "eval_task_counts": dict(eval_counts),
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
    ) -> tuple[list[list[int]], list[list[int]], Counter[str]]:
        inputs: list[list[int]] = []
        labels: list[list[int]] = []
        task_counts: Counter[str] = Counter()
        for _ in range(count):
            task = self.config.tasks[int(rng.integers(0, len(self.config.tasks)))]
            text = self._sample_task_text(rng, task)
            encoded = self._encode_text(text, spec)
            inputs.append(encoded[:-1][: self.config.max_length])
            labels.append(encoded[1:][: self.config.max_length])
            task_counts[task] += 1
        return inputs, labels, task_counts

    def _sample_task_text(self, rng: np.random.Generator, task: str) -> str:
        if task == "copy":
            symbol = self._sample_symbol_string(rng)
            return f"copy:{symbol}=>{symbol}"
        if task == "reverse":
            symbol = self._sample_symbol_string(rng)
            return f"reverse:{symbol}=>{symbol[::-1]}"
        if task == "sort":
            symbol = self._sample_symbol_string(rng)
            return f"sort:{symbol}=>{''.join(sorted(symbol))}"
        if task == "addition":
            left = int(rng.integers(0, self.config.max_number + 1))
            right = int(rng.integers(0, self.config.max_number + 1))
            return f"addition:{left}+{right}=>{left + right}"
        raise AssertionError(f"Unhandled task '{task}'")

    def _sample_symbol_string(self, rng: np.random.Generator) -> str:
        length = int(rng.integers(self.config.min_symbol_length, self.config.max_symbol_length + 1))
        chars = rng.choice(list(self.config.alphabet), size=length, replace=True)
        return "".join(chars.tolist())

    @staticmethod
    def _encode_text(text: str, spec: TokenizerSpec) -> list[int]:
        char_to_id = {char: idx for idx, char in enumerate(BASE_CHARSET)}
        return [spec.bos_token_id or 0] + [char_to_id[char] for char in text] + [spec.eos_token_id or 1]


register_mechanism("procedural", lambda config: ProceduralMechanism(ProceduralConfig(**config)))
