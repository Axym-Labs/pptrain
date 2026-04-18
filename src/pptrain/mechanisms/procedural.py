from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pptrain.core.base import ExecutedSymbolicTask, SymbolicTask, SymbolicTaskFamily, TokenizerSpec
from pptrain.core.presets import TaskPreset, sequence_preset
from pptrain.core.registry import register_task
from pptrain.mechanisms._shared import (
    TokenVocabulary,
    TokenVocabularyBuilder,
    require_non_empty,
    require_positive_range,
    require_subset,
    require_supported,
)

BASE_CHARSET = "abcdefghijklmnopqrstuvwxyz0123456789:+=>,;|-_ "
SUPPORTED_PROCEDURAL_TASKS = {"copy", "identity", "reverse", "sort", "addition", "set", "union", "delete"}


@dataclass(slots=True)
class ProceduralConfig:
    tasks: tuple[str, ...] = field(default_factory=lambda: ("copy", "reverse", "sort", "addition"))
    min_symbol_length: int = 4
    max_symbol_length: int = 24
    alphabet: str = "abcdefghijklmnopqrstuvwxyz"
    max_number: int = 9999
    sequence_count: int = 512
    eval_sequence_count: int = 64
    max_length: int = 128


_PROCEDURAL_REFERENCE = "Jiang et al. 2026"
_PROCEDURAL_LENGTHS = (16, 32, 64)
_PROCEDURAL_TASKS = ("identity", "reverse", "sort", "set", "union", "delete")


def _paper_task_preset(task: str, *, max_length: int) -> TaskPreset:
    return sequence_preset(
        f"paper_{task}_len{max_length}",
        f"Procedural-pretraining {task} preset at sequence length {max_length}.",
        sequence_count=160_064,
        eval_sequence_count=4_096,
        reference=_PROCEDURAL_REFERENCE,
        tasks=(task,),
        min_symbol_length=4,
        max_symbol_length=max(8, max_length // 2),
        max_length=max_length,
    )


def _paper_task_presets() -> tuple[TaskPreset, ...]:
    return tuple(
        _paper_task_preset(task, max_length=max_length)
        for max_length in _PROCEDURAL_LENGTHS
        for task in _PROCEDURAL_TASKS
    )


PROCEDURAL_PRESETS: tuple[TaskPreset, ...] = (
    sequence_preset(
        "smoke",
        "Tiny procedural smoke run.",
        sequence_count=128,
        eval_sequence_count=32,
        reference="pptrain",
        tasks=("copy", "reverse", "sort"),
        min_symbol_length=4,
        max_symbol_length=16,
        max_length=64,
    ),
    *_paper_task_presets(),
)


@dataclass(slots=True)
class ProceduralProgram:
    left: str | int
    right: str | int | None = None


class ProceduralTaskFamily(SymbolicTaskFamily):
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
        require_supported("procedural tasks", self.config.tasks, SUPPORTED_PROCEDURAL_TASKS)
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
        return [
            spec.bos_token_id or 0,
            *self._vocabulary.encode_group("char", executed.payload),
            spec.eos_token_id or 1,
        ]

    def _sample_program(self, rng: np.random.Generator, task: str) -> ProceduralProgram:
        if task == "addition":
            left = int(rng.integers(0, self.config.max_number + 1))
            right = int(rng.integers(0, self.config.max_number + 1))
            return ProceduralProgram(left=left, right=right)
        if task == "union":
            return ProceduralProgram(
                left=self._sample_symbol_string(rng),
                right=self._sample_symbol_string(rng),
            )
        if task == "delete":
            return ProceduralProgram(
                left=self._sample_symbol_string(rng),
                right=str(rng.choice(list(self.config.alphabet))),
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
            filtered = _remove_first_occurrence(source, query)
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


def _stable_unique_string(text: str) -> str:
    return "".join(dict.fromkeys(text))


def _remove_first_occurrence(source: str, query: str) -> str:
    if not query:
        return source
    index = source.find(query[0])
    if index < 0:
        return source
    return source[:index] + source[index + 1 :]


register_task(
    "procedural",
    lambda config: ProceduralTaskFamily(ProceduralConfig(**config)),
    description=ProceduralTaskFamily.description,
    presets=PROCEDURAL_PRESETS,
)


ProceduralMechanism = ProceduralTaskFamily
