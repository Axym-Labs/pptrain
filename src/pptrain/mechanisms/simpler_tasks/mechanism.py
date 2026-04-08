from __future__ import annotations

from collections import Counter

import numpy as np

from pptrain.core.base import TokenSequenceMechanism, TokenizerSpec
from pptrain.core.registry import register_mechanism
from pptrain.mechanisms._shared import (
    TokenVocabulary,
    TokenVocabularyBuilder,
    require_non_empty,
    require_positive_range,
    require_probability,
    require_supported,
    require_unique_characters,
)
from pptrain.mechanisms.simpler_tasks.config import SimplerTasksConfig
from pptrain.mechanisms.simpler_tasks.tasks import (
    BINARY_QUERY_TASKS,
    BINARY_SET_TASKS,
    SUPPORTED_SIMPLER_TASKS,
    UNARY_TASKS,
    apply_binary_task,
    apply_unary_task,
    sample_search_query,
    sample_symbols,
)


class SimplerTasksMechanism(TokenSequenceMechanism):
    name = "simpler_tasks"
    description = "Synthetic set/copy/query tasks derived from the simpler synthetic pretraining line."
    max_sampling_attempts = 32

    def __init__(self, config: SimplerTasksConfig) -> None:
        super().__init__(config)
        require_non_empty("tasks", self.config.tasks)
        require_supported("simpler tasks", self.config.tasks, SUPPORTED_SIMPLER_TASKS)
        require_positive_range("min_symbols", self.config.min_symbols, "max_symbols", self.config.max_symbols)
        require_positive_range(
            "min_query_symbols",
            self.config.min_query_symbols,
            "max_query_symbols",
            self.config.max_query_symbols,
        )
        require_probability("positive_search_probability", self.config.positive_search_probability)
        require_non_empty("alphabet", self.config.alphabet)
        require_unique_characters("alphabet", self.config.alphabet, allow_whitespace=False)
        self._vocabulary = self._build_vocabulary()

    def tokenizer_spec(self) -> TokenizerSpec:
        extra_tokens = self._vocabulary.token_ids(
            ["yes", "no", "input_sep", "output_sep", *self._task_token_names()]
        )
        return self._vocabulary.tokenizer_spec(extra_token_ids=extra_tokens)

    def sample_example(
        self,
        rng: np.random.Generator,
        spec: TokenizerSpec,
    ) -> tuple[list[int], dict[str, int | str]]:
        task = self.config.tasks[int(rng.integers(0, len(self.config.tasks)))]
        tokens, target_length, input_length = self._sample_task_tokens(rng, task, spec)
        return tokens, {
            "task": task,
            "input_symbols": input_length,
            "target_symbols": target_length,
        }

    def _split_metadata(self, split: str, items: list[dict[str, int | str]]) -> dict[str, object]:
        task_counts = Counter(str(item["task"]) for item in items)
        input_lengths = [int(item["input_symbols"]) for item in items]
        target_lengths = [int(item["target_symbols"]) for item in items]
        return {
            f"{split}_task_counts": dict(task_counts),
            f"{split}_avg_input_symbols": float(np.mean(input_lengths)) if input_lengths else None,
            f"{split}_avg_target_symbols": float(np.mean(target_lengths)) if target_lengths else None,
        }

    def _sample_task_tokens(
        self,
        rng: np.random.Generator,
        task: str,
        spec: TokenizerSpec,
    ) -> tuple[list[int], int, int]:
        if task in UNARY_TASKS:
            sequence = self._sample_sequence(rng)
            target = apply_unary_task(task, sequence)
            tokens = [
                spec.bos_token_id or 0,
                self._vocabulary.token(self._task_token_name(task)),
                *self._encode_symbols(sequence),
                self._vocabulary.token("output_sep"),
                *self._encode_payload(target),
                spec.eos_token_id or 1,
            ]
            return tokens, len(target), len(sequence)

        if task in BINARY_QUERY_TASKS:
            sequence = self._sample_sequence(rng)
            query = sample_search_query(
                rng,
                sequence=sequence,
                alphabet=self.config.alphabet,
                min_query_symbols=self.config.min_query_symbols,
                max_query_symbols=self.config.max_query_symbols,
                positive_probability=self.config.positive_search_probability,
            )
            target = apply_binary_task(task, sequence, query)
            tokens = [
                spec.bos_token_id or 0,
                self._vocabulary.token(self._task_token_name(task)),
                *self._encode_symbols(sequence),
                self._vocabulary.token("input_sep"),
                *self._encode_symbols(query),
                self._vocabulary.token("output_sep"),
                *self._encode_payload(target),
                spec.eos_token_id or 1,
            ]
            return tokens, len(target), len(sequence) + len(query)

        if task in BINARY_SET_TASKS:
            left = self._sample_sequence(rng)
            right = self._sample_sequence(rng)
            target = apply_binary_task(task, left, right)
            tokens = [
                spec.bos_token_id or 0,
                self._vocabulary.token(self._task_token_name(task)),
                *self._encode_symbols(left),
                self._vocabulary.token("input_sep"),
                *self._encode_symbols(right),
                self._vocabulary.token("output_sep"),
                *self._encode_payload(target),
                spec.eos_token_id or 1,
            ]
            return tokens, len(target), len(left) + len(right)

        raise KeyError(f"Unsupported task '{task}'")

    def _sample_sequence(self, rng: np.random.Generator) -> list[str]:
        length = int(rng.integers(self.config.min_symbols, self.config.max_symbols + 1))
        return sample_symbols(rng, self.config.alphabet, length)

    def _encode_symbols(self, items: list[str]) -> list[int]:
        return self._vocabulary.encode_group("symbol", items)

    def _encode_payload(self, items: list[str]) -> list[int]:
        encoded: list[int] = []
        for item in items:
            if item == "yes" or item == "no":
                encoded.append(self._vocabulary.token(item))
            else:
                encoded.extend(self._vocabulary.encode_any([item], groups=("symbol", "digit")))
        return encoded

    def _build_vocabulary(self) -> TokenVocabulary:
        builder = TokenVocabularyBuilder()
        builder.add_group("symbol", list(self.config.alphabet))
        builder.add_group("digit", list("0123456789"))
        builder.add_tokens(*self._task_token_names(), "yes", "no", "input_sep", "output_sep", "bos", "eos", "pad")
        return builder.build()

    def _task_token_names(self) -> list[str]:
        return [self._task_token_name(task) for task in self.config.tasks]

    @staticmethod
    def _task_token_name(task: str) -> str:
        return f"task:{task}"


register_mechanism(
    "simpler_tasks",
    lambda config: SimplerTasksMechanism(SimplerTasksConfig(**config)),
    description=SimplerTasksMechanism.description,
)
