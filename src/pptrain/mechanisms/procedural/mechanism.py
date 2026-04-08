from __future__ import annotations

from collections import Counter

import numpy as np

from pptrain.core.base import TokenSequenceMechanism, TokenizerSpec
from pptrain.core.registry import register_mechanism
from pptrain.mechanisms.procedural.config import ProceduralConfig

BASE_CHARSET = "abcdefghijklmnopqrstuvwxyz0123456789:+=>,;|-_ "


class ProceduralMechanism(TokenSequenceMechanism):
    name = "procedural"
    description = "Short procedural text tasks such as copy, reverse, sort, and addition."

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

    def sample_tokens(
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

    @staticmethod
    def _encode_text(text: str, spec: TokenizerSpec) -> list[int]:
        char_to_id = {char: idx for idx, char in enumerate(BASE_CHARSET)}
        return [spec.bos_token_id or 0] + [char_to_id[char] for char in text] + [spec.eos_token_id or 1]


register_mechanism(
    "procedural",
    lambda config: ProceduralMechanism(ProceduralConfig(**config)),
    description=ProceduralMechanism.description,
)
