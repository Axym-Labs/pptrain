from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from pptrain.core.base import TokenSequenceMechanism, TokenizerSpec
from pptrain.core.registry import register_mechanism
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


@dataclass(slots=True)
class _Vocabulary:
    symbol_to_id: dict[str, int]
    digit_to_id: dict[str, int]
    task_to_id: dict[str, int]
    yes_token_id: int
    no_token_id: int
    input_sep_token_id: int
    output_sep_token_id: int
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int


class SimplerTasksMechanism(TokenSequenceMechanism):
    name = "simpler_tasks"
    description = "Synthetic set/copy/query tasks derived from the simpler synthetic pretraining line."

    def __init__(self, config: SimplerTasksConfig) -> None:
        super().__init__(config)
        if not self.config.tasks:
            raise ValueError("At least one task must be configured.")
        unknown_tasks = sorted(set(self.config.tasks) - set(SUPPORTED_SIMPLER_TASKS))
        if unknown_tasks:
            raise ValueError(f"Unsupported simpler tasks: {unknown_tasks}")
        if self.config.min_symbols < 1 or self.config.max_symbols < self.config.min_symbols:
            raise ValueError("min_symbols and max_symbols must define a valid positive range.")
        if self.config.min_query_symbols < 1 or self.config.max_query_symbols < self.config.min_query_symbols:
            raise ValueError("min_query_symbols and max_query_symbols must define a valid positive range.")
        if not 0.0 <= self.config.positive_search_probability <= 1.0:
            raise ValueError("positive_search_probability must be in [0, 1].")
        if not self.config.alphabet:
            raise ValueError("alphabet must not be empty.")
        if any(char.isspace() for char in self.config.alphabet):
            raise ValueError("alphabet must not contain whitespace.")
        if len(set(self.config.alphabet)) != len(self.config.alphabet):
            raise ValueError("alphabet must contain unique symbols.")
        self._vocabulary = self._build_vocabulary()

    def tokenizer_spec(self) -> TokenizerSpec:
        vocab = self._vocabulary
        extra_tokens = {
            "yes": vocab.yes_token_id,
            "no": vocab.no_token_id,
            "input_sep": vocab.input_sep_token_id,
            "output_sep": vocab.output_sep_token_id,
        }
        extra_tokens.update({f"task:{task}": token_id for task, token_id in vocab.task_to_id.items()})
        return TokenizerSpec(
            vocab_size=vocab.pad_token_id + 1,
            pad_token_id=vocab.pad_token_id,
            bos_token_id=vocab.bos_token_id,
            eos_token_id=vocab.eos_token_id,
            extra_token_ids=extra_tokens,
        )

    def sample_tokens(
        self,
        rng: np.random.Generator,
        spec: TokenizerSpec,
    ) -> tuple[list[int], dict[str, int | str]]:
        for _ in range(32):
            task = self.config.tasks[int(rng.integers(0, len(self.config.tasks)))]
            tokens, target_length, input_length = self._sample_task_tokens(rng, task, spec)
            if len(tokens) <= self.config.max_length + 1:
                return tokens, {
                    "task": task,
                    "input_symbols": input_length,
                    "target_symbols": target_length,
                }
        raise RuntimeError("Failed to sample a simpler_tasks example within max_length.")

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
        vocab = self._vocabulary
        if task in UNARY_TASKS:
            sequence = self._sample_sequence(rng)
            target = apply_unary_task(task, sequence)
            tokens = [
                spec.bos_token_id or 0,
                vocab.task_to_id[task],
                *self._encode_symbols(sequence),
                vocab.output_sep_token_id,
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
                vocab.task_to_id[task],
                *self._encode_symbols(sequence),
                vocab.input_sep_token_id,
                *self._encode_symbols(query),
                vocab.output_sep_token_id,
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
                vocab.task_to_id[task],
                *self._encode_symbols(left),
                vocab.input_sep_token_id,
                *self._encode_symbols(right),
                vocab.output_sep_token_id,
                *self._encode_payload(target),
                spec.eos_token_id or 1,
            ]
            return tokens, len(target), len(left) + len(right)

        raise KeyError(f"Unsupported task '{task}'")

    def _sample_sequence(self, rng: np.random.Generator) -> list[str]:
        length = int(rng.integers(self.config.min_symbols, self.config.max_symbols + 1))
        return sample_symbols(rng, self.config.alphabet, length)

    def _encode_symbols(self, items: list[str]) -> list[int]:
        return [self._vocabulary.symbol_to_id[item] for item in items]

    def _encode_payload(self, items: list[str]) -> list[int]:
        encoded: list[int] = []
        for item in items:
            if item in self._vocabulary.symbol_to_id:
                encoded.append(self._vocabulary.symbol_to_id[item])
            elif item in self._vocabulary.digit_to_id:
                encoded.append(self._vocabulary.digit_to_id[item])
            elif item == "yes":
                encoded.append(self._vocabulary.yes_token_id)
            elif item == "no":
                encoded.append(self._vocabulary.no_token_id)
            else:
                raise KeyError(f"Unknown payload token '{item}'")
        return encoded

    def _build_vocabulary(self) -> _Vocabulary:
        next_token_id = 0
        symbol_to_id = {symbol: idx for idx, symbol in enumerate(self.config.alphabet)}
        next_token_id = len(symbol_to_id)
        digit_to_id = {digit: next_token_id + idx for idx, digit in enumerate("0123456789")}
        next_token_id += len(digit_to_id)
        task_to_id = {task: next_token_id + idx for idx, task in enumerate(self.config.tasks)}
        next_token_id += len(task_to_id)
        yes_token_id = next_token_id
        no_token_id = next_token_id + 1
        input_sep_token_id = next_token_id + 2
        output_sep_token_id = next_token_id + 3
        bos_token_id = next_token_id + 4
        eos_token_id = next_token_id + 5
        pad_token_id = next_token_id + 6
        return _Vocabulary(
            symbol_to_id=symbol_to_id,
            digit_to_id=digit_to_id,
            task_to_id=task_to_id,
            yes_token_id=yes_token_id,
            no_token_id=no_token_id,
            input_sep_token_id=input_sep_token_id,
            output_sep_token_id=output_sep_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )


register_mechanism(
    "simpler_tasks",
    lambda config: SimplerTasksMechanism(SimplerTasksConfig(**config)),
    description=SimplerTasksMechanism.description,
)
