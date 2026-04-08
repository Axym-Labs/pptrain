from __future__ import annotations

import numpy as np

from pptrain.core.base import ExecutedSymbolicTask, SymbolicTask, SymbolicTaskMechanism, TokenizerSpec
from pptrain.core.registry import register_mechanism
from pptrain.mechanisms._shared import (
    TokenVocabulary,
    TokenVocabularyBuilder,
    require_non_empty,
    require_positive_range,
    require_supported,
    require_unique_characters,
)
from pptrain.mechanisms.lime.config import LIME_PRESETS, LIMEConfig
from pptrain.mechanisms.lime.generator import LIMEExample, sample_lime_example


class LIMEMechanism(SymbolicTaskMechanism):
    name = "lime"
    description = "LIME-style induction, deduction, and abduction substitution tasks."
    max_sampling_attempts = 32
    task_group_metadata_key = "mode"

    def __init__(self, config: LIMEConfig) -> None:
        super().__init__(config)
        require_non_empty("modes", self.config.modes)
        require_supported("LIME modes", self.config.modes, {"induct", "deduct", "abduct"})
        require_positive_range("min_variables", self.config.min_variables, "max_variables", self.config.max_variables)
        require_positive_range(
            "min_pattern_length",
            self.config.min_pattern_length,
            "max_pattern_length",
            self.config.max_pattern_length,
        )
        require_positive_range(
            "min_substitution_length",
            self.config.min_substitution_length,
            "max_substitution_length",
            self.config.max_substitution_length,
        )
        require_unique_characters("upper_symbols", self.config.upper_symbols, allow_whitespace=False)
        require_unique_characters("lower_symbols", self.config.lower_symbols, allow_whitespace=False)
        require_unique_characters("math_symbols", self.config.math_symbols, allow_whitespace=False)
        self._vocabulary = self._build_vocabulary()

    def tokenizer_spec(self) -> TokenizerSpec:
        extra_token_ids = self._vocabulary.token_ids(
            [
                *self._mode_token_names(),
                "declare_upper",
                "declare_lower",
                "declare_math",
                "section_sep",
                "output_sep",
                "maps_to",
                "pair_sep",
            ]
        )
        return self._vocabulary.tokenizer_spec(extra_token_ids=extra_token_ids)

    def sample_task(self, rng: np.random.Generator) -> SymbolicTask:
        mode = self.config.modes[int(rng.integers(0, len(self.config.modes)))]
        example = sample_lime_example(
            rng,
            upper_pool=self.config.upper_symbols,
            lower_pool=self.config.lower_symbols,
            math_pool=self.config.math_symbols,
            min_variables=self.config.min_variables,
            max_variables=self.config.max_variables,
            min_pattern_length=self.config.min_pattern_length,
            max_pattern_length=self.config.max_pattern_length,
            min_substitution_length=self.config.min_substitution_length,
            max_substitution_length=self.config.max_substitution_length,
        )
        return SymbolicTask(name=mode, payload=example)

    def execute_task(self, task: SymbolicTask) -> ExecutedSymbolicTask:
        example = task.payload
        return ExecutedSymbolicTask(
            name=task.name,
            payload=example,
            metadata={
            "variable_count": len(example.upper_vocab),
            "pattern_length": len(example.pattern),
            "result_length": len(example.result),
            },
        )

    def serialize_task(self, executed: ExecutedSymbolicTask, spec: TokenizerSpec) -> list[int]:
        return self._encode_example(executed.name, executed.payload, spec)

    def numeric_metadata_fields(self) -> tuple[str, ...]:
        return ("variable_count", "pattern_length", "result_length")

    def _encode_example(
        self,
        mode: str,
        example: LIMEExample,
        spec: TokenizerSpec,
    ) -> list[int]:
        prompt = [
            self._vocabulary.token(self._mode_token_name(mode)),
            self._vocabulary.token("declare_upper"),
            *self._encode_symbols(example.upper_vocab),
            self._vocabulary.token("section_sep"),
            self._vocabulary.token("declare_lower"),
            *self._encode_symbols(example.lower_vocab),
            self._vocabulary.token("section_sep"),
            self._vocabulary.token("declare_math"),
            *self._encode_symbols(example.math_vocab),
            self._vocabulary.token("section_sep"),
        ]
        if mode == "induct":
            input_body = [
                *self._encode_symbols(example.result),
                self._vocabulary.token("section_sep"),
                *self._encode_substitution_pairs(example.substitution_pairs),
            ]
            target_body = self._encode_symbols(example.pattern)
        elif mode == "deduct":
            input_body = [
                *self._encode_symbols(example.pattern),
                self._vocabulary.token("section_sep"),
                *self._encode_substitution_pairs(example.substitution_pairs),
            ]
            target_body = self._encode_symbols(example.result)
        elif mode == "abduct":
            input_body = [
                *self._encode_symbols(example.pattern),
                self._vocabulary.token("section_sep"),
                *self._encode_symbols(example.result),
            ]
            target_body = self._encode_substitution_pairs(example.substitution_pairs)
        else:
            raise AssertionError(f"Unhandled mode '{mode}'")

        return [
            spec.bos_token_id or 0,
            *prompt,
            *input_body,
            self._vocabulary.token("output_sep"),
            *target_body,
            spec.eos_token_id or 1,
        ]

    def _encode_substitution_pairs(self, pairs: list[tuple[str, list[str]]]) -> list[int]:
        encoded: list[int] = []
        for index, (variable, substitution) in enumerate(pairs):
            if index > 0:
                encoded.append(self._vocabulary.token("pair_sep"))
            encoded.append(self._vocabulary.group("upper")[variable])
            encoded.append(self._vocabulary.token("maps_to"))
            encoded.extend(self._encode_symbols(substitution))
        return encoded

    def _encode_symbols(self, symbols: list[str]) -> list[int]:
        return self._vocabulary.encode_any(symbols, groups=("upper", "lower", "math"))

    def _build_vocabulary(self) -> TokenVocabulary:
        builder = TokenVocabularyBuilder()
        builder.add_group("upper", list(self.config.upper_symbols))
        builder.add_group("lower", list(self.config.lower_symbols))
        builder.add_group("math", list(self.config.math_symbols))
        builder.add_tokens(
            *self._mode_token_names(),
            "declare_upper",
            "declare_lower",
            "declare_math",
            "section_sep",
            "output_sep",
            "maps_to",
            "pair_sep",
            "bos",
            "eos",
            "pad",
        )
        return builder.build()

    def _mode_token_names(self) -> list[str]:
        return [self._mode_token_name(mode) for mode in self.config.modes]

    @staticmethod
    def _mode_token_name(mode: str) -> str:
        return f"mode:{mode}"


register_mechanism(
    "lime",
    lambda config: LIMEMechanism(LIMEConfig(**config)),
    description=LIMEMechanism.description,
    presets=LIME_PRESETS,
)
