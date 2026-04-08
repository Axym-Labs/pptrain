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
    require_subset,
    require_supported,
)
from pptrain.mechanisms.procedural.config import ProceduralConfig

BASE_CHARSET = "abcdefghijklmnopqrstuvwxyz0123456789:+=>,;|-_ "
SUPPORTED_TASKS = {"copy", "reverse", "sort", "addition"}


class ProceduralMechanism(TokenSequenceMechanism):
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

    def sample_example(
        self,
        rng: np.random.Generator,
        spec: TokenizerSpec,
    ) -> tuple[list[int], dict[str, str]]:
        task = self.config.tasks[int(rng.integers(0, len(self.config.tasks)))]
        text = self._sample_task_text(rng, task)
        encoded = self._encode_text(text, spec)
        return encoded, {"task": task}

    def _split_metadata(self, split: str, items: list[dict[str, str]]) -> dict[str, dict[str, int]]:
        task_counts = Counter(item["task"] for item in items)
        return {f"{split}_task_counts": dict(task_counts)}

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

    def _encode_text(self, text: str, spec: TokenizerSpec) -> list[int]:
        return [spec.bos_token_id or 0, *self._vocabulary.encode_group("char", text), spec.eos_token_id or 1]

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
)
