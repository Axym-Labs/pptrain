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
    require_subset,
    require_supported,
)
from pptrain.mechanisms.procedural.config import PROCEDURAL_PRESETS, ProceduralConfig

BASE_CHARSET = "abcdefghijklmnopqrstuvwxyz0123456789:+=>,;|-_ "
SUPPORTED_TASKS = {"copy", "identity", "reverse", "sort", "addition", "set", "union", "delete"}


@dataclass(slots=True)
class ProceduralProgram:
    left: str | int
    right: str | int | None = None


class ProceduralMechanism(SymbolicTaskMechanism):
    name = "procedural"
    description = "Short procedural text tasks such as copy, reverse, sort, and addition."

    def __init__(self, config: ProceduralConfig) -> None:
        super().__init__(config)
        require_positive_range(
            "min_symbol_length",
            self.config.min_symbol_length,
            "max_symbol_length",
            self.config.max_symbol_length,
        )
        if self.config.max_number < 1:
            raise ValueError("max_number must be positive.")
        require_non_empty("tasks", self.config.tasks)
        require_supported("procedural tasks", self.config.tasks, SUPPORTED_TASKS)
        require_non_empty("alphabet", self.config.alphabet)
        require_subset("alphabet", self.config.alphabet, BASE_CHARSET)
        self._vocabulary = self._build_vocabulary()

    def tokenizer_spec(self) -> TokenizerSpec:
        return self._vocabulary.tokenizer_spec()

    def sample_task(self, rng: np.random.Generator) -> SymbolicTask:
        task = self.config.tasks[int(rng.integers(0, len(self.config.tasks)))]
        return SymbolicTask(name=task, payload=self._sample_program(rng, task))

    def execute_task(self, task: SymbolicTask) -> ExecutedSymbolicTask:
        program = task.payload
        text = self._execute_program(task.name, program)
        return ExecutedSymbolicTask(name=task.name, payload=text)

    def serialize_task(self, executed: ExecutedSymbolicTask, spec: TokenizerSpec) -> list[int]:
        return [spec.bos_token_id or 0, *self._vocabulary.encode_group("char", executed.payload), spec.eos_token_id or 1]

    def _sample_program(self, rng: np.random.Generator, task: str) -> ProceduralProgram:
        if task == "addition":
            left = int(rng.integers(0, self.config.max_number + 1))
            right = int(rng.integers(0, self.config.max_number + 1))
            return ProceduralProgram(left=left, right=right)
        if task == "union" or task == "delete":
            return ProceduralProgram(
                left=self._sample_symbol_string(rng),
                right=self._sample_symbol_string(rng),
            )
        return ProceduralProgram(left=self._sample_symbol_string(rng))

    def _execute_program(self, task: str, program: ProceduralProgram) -> str:
        if task == "copy" or task == "identity":
            label = "identity" if task == "identity" else "copy"
            return f"{label}:{program.left}=>{program.left}"
        if task == "reverse":
            symbol = str(program.left)
            return f"reverse:{symbol}=>{symbol[::-1]}"
        if task == "sort":
            symbol = str(program.left)
            return f"sort:{symbol}=>{''.join(sorted(symbol))}"
        if task == "set":
            symbol = str(program.left)
            return f"set:{symbol}=>{_stable_unique_string(symbol)}"
        if task == "union":
            left = str(program.left)
            right = str(program.right)
            return f"union:{left}|{right}=>{_stable_unique_string(left + right)}"
        if task == "delete":
            source = str(program.left)
            query = str(program.right)
            filtered = "".join(char for char in source if char not in set(query))
            return f"delete:{source}|{query}=>{filtered}"
        if task == "addition":
            left = int(program.left)
            right = int(program.right or 0)
            return f"addition:{left}+{right}=>{left + right}"
        raise AssertionError(f"Unhandled task '{task}'")

    def _sample_symbol_string(self, rng: np.random.Generator) -> str:
        length = int(rng.integers(self.config.min_symbol_length, self.config.max_symbol_length + 1))
        chars = rng.choice(list(self.config.alphabet), size=length, replace=True)
        return "".join(chars.tolist())

    @staticmethod
    def _build_vocabulary() -> TokenVocabulary:
        builder = TokenVocabularyBuilder()
        builder.add_group("char", list(BASE_CHARSET))
        builder.add_tokens("bos", "eos", "pad")
        return builder.build()


register_mechanism(
    "procedural",
    lambda config: ProceduralMechanism(ProceduralConfig(**config)),
    description=ProceduralMechanism.description,
    presets=PROCEDURAL_PRESETS,
)


def _stable_unique_string(text: str) -> str:
    return "".join(dict.fromkeys(text))
