from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pptrain.core.base import ExecutedSymbolicTask, SymbolicTask, SymbolicTaskMechanism, TokenizerSpec
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
from pptrain.mechanisms.simpler_tasks.config import SIMPLER_TASKS_PRESETS, SimplerTasksConfig
from pptrain.mechanisms.simpler_tasks.tasks import (
    BINARY_SET_TASKS,
    ORDER_QUERY_TASKS,
    REPLACEMENT_TASKS,
    SINGLE_SYMBOL_QUERY_TASKS,
    SUBSEQUENCE_QUERY_TASKS,
    SUPPORTED_SIMPLER_TASKS,
    UNARY_TASKS,
    apply_binary_task,
    apply_unary_task,
    sample_count_query,
    sample_replacement_query,
    sample_search_query,
    sample_sort_order,
    sample_symbols,
)


@dataclass(slots=True)
class SimplerTaskProgram:
    left: list[str]
    right: list[str] | None = None


class SimplerTasksMechanism(SymbolicTaskMechanism):
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

    def sample_task(self, rng: np.random.Generator) -> SymbolicTask:
        task = self.config.tasks[int(rng.integers(0, len(self.config.tasks)))]
        return SymbolicTask(name=task, payload=self._sample_program(rng, task))

    def execute_task(self, task: SymbolicTask) -> ExecutedSymbolicTask:
        program = task.payload
        if task.name in UNARY_TASKS:
            target = apply_unary_task(task.name, program.left)
            return ExecutedSymbolicTask(
                name=task.name,
                payload=(program.left, None, target),
                metadata={"input_symbols": len(program.left), "target_symbols": len(target)},
            )

        target = apply_binary_task(task.name, program.left, program.right or [])
        return ExecutedSymbolicTask(
            name=task.name,
            payload=(program.left, program.right, target),
            metadata={
                "input_symbols": len(program.left) + len(program.right or []),
                "target_symbols": len(target),
            },
        )

    def serialize_task(self, executed: ExecutedSymbolicTask, spec: TokenizerSpec) -> list[int]:
        left, right, target = executed.payload
        tokens = [
            spec.bos_token_id or 0,
            self._vocabulary.token(self._task_token_name(executed.name)),
            *self._encode_symbols(left),
        ]
        if right is not None:
            tokens.extend([self._vocabulary.token("input_sep"), *self._encode_symbols(right)])
        tokens.extend(
            [
                self._vocabulary.token("output_sep"),
                *self._encode_payload(target),
                spec.eos_token_id or 1,
            ]
        )
        return tokens

    def numeric_metadata_fields(self) -> tuple[str, ...]:
        return ("input_symbols", "target_symbols")

    def _sample_program(self, rng: np.random.Generator, task: str) -> SimplerTaskProgram:
        if task in UNARY_TASKS:
            return SimplerTaskProgram(left=self._sample_sequence(rng))

        sequence = self._sample_sequence(rng)
        if task in SINGLE_SYMBOL_QUERY_TASKS:
            query = sample_count_query(
                rng,
                sequence,
                self.config.alphabet,
            )
            return SimplerTaskProgram(left=sequence, right=query)

        if task in SUBSEQUENCE_QUERY_TASKS:
            query = sample_search_query(
                rng,
                sequence=sequence,
                alphabet=self.config.alphabet,
                min_query_symbols=self.config.min_query_symbols,
                max_query_symbols=self.config.max_query_symbols,
                positive_probability=self.config.positive_search_probability,
            )
            return SimplerTaskProgram(left=sequence, right=query)

        if task in ORDER_QUERY_TASKS:
            return SimplerTaskProgram(left=sequence, right=sample_sort_order(rng, sequence))

        if task in REPLACEMENT_TASKS:
            return SimplerTaskProgram(
                left=sequence,
                right=sample_replacement_query(
                    rng,
                    sequence,
                    self.config.alphabet,
                    many=task == "replace_many",
                ),
            )

        if task in BINARY_SET_TASKS:
            return SimplerTaskProgram(left=sequence, right=self._sample_sequence(rng))

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
        builder.add_group("digit", list("-0123456789"))
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
    presets=SIMPLER_TASKS_PRESETS,
)
