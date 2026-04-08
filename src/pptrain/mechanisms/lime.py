from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pptrain.core.base import ExecutedSymbolicTask, SymbolicTask, SymbolicTaskMechanism, TokenizerSpec
from pptrain.core.presets import MechanismPreset, sequence_preset
from pptrain.core.registry import register_mechanism
from pptrain.mechanisms._shared import (
    TokenVocabulary,
    TokenVocabularyBuilder,
    require_non_empty,
    require_positive_range,
    require_supported,
    require_unique_characters,
)


@dataclass(slots=True)
class LIMEConfig:
    modes: tuple[str, ...] = field(default_factory=lambda: ("induct", "deduct", "abduct"))
    upper_symbols: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lower_symbols: str = "abcdefghijklmnopqrstuvwxyz"
    math_symbols: str = "+-=*/()"
    min_variables: int = 3
    max_variables: int = 5
    min_pattern_length: int = 5
    max_pattern_length: int = 18
    min_substitution_length: int = 2
    max_substitution_length: int = 6
    sequence_count: int = 512
    eval_sequence_count: int = 64
    max_length: int = 192


def _benchmark_preset(
    name: str,
    description: str,
    *,
    sequence_count: int,
    modes: tuple[str, ...],
    reference: str,
) -> MechanismPreset:
    return sequence_preset(
        name,
        description,
        sequence_count=sequence_count,
        eval_sequence_count=10_000 if sequence_count <= 1_000_000 else 1_000,
        reference=reference,
        modes=modes,
        max_length=256,
    )


def _single_mode_presets() -> tuple[MechanismPreset, ...]:
    scales = (
        ("paper_individual_100k", 100_000),
        ("paper_individual_1m", 1_000_000),
    )
    descriptions = {
        "induct": "Induct-only LIME preset",
        "deduct": "Deduct-only LIME preset",
        "abduct": "Abduct-only LIME preset",
    }
    result: list[MechanismPreset] = []
    for prefix, sequence_count in scales:
        for mode in ("induct", "deduct", "abduct"):
            result.append(
                _benchmark_preset(
                    f"{prefix}_{mode}",
                    f"{descriptions[mode]} at the {sequence_count:,}-example scale.",
                    sequence_count=sequence_count,
                    modes=(mode,),
                    reference="Wu et al. 2021 / Wu et al. 2022",
                )
            )
    return tuple(result)


LIME_PRESETS: tuple[MechanismPreset, ...] = (
    sequence_preset(
        "smoke",
        "Tiny LIME smoke run.",
        sequence_count=128,
        eval_sequence_count=32,
        reference="pptrain",
        modes=("induct", "deduct", "abduct"),
        max_length=128,
    ),
    _benchmark_preset(
        "paper_benchmark_100k",
        "LIME benchmark preset at the 100k scale used in the simpler synthetic-task comparison.",
        sequence_count=100_000,
        modes=("induct", "deduct", "abduct"),
        reference="Wu et al. 2022",
    ),
    _benchmark_preset(
        "paper_benchmark_1m",
        "LIME benchmark preset at the 1M scale used in the simpler synthetic-task comparison.",
        sequence_count=1_000_000,
        modes=("induct", "deduct", "abduct"),
        reference="Wu et al. 2022",
    ),
    _benchmark_preset(
        "paper_mixed_5m",
        "Original LIME mixed-task preset with a 5M-example budget.",
        sequence_count=5_000_000,
        modes=("induct", "deduct", "abduct"),
        reference="Wu et al. 2021",
    ),
    *_single_mode_presets(),
)


@dataclass(slots=True)
class LIMEExample:
    upper_vocab: list[str]
    lower_vocab: list[str]
    math_vocab: list[str]
    pattern: list[str]
    result: list[str]
    substitution_pairs: list[tuple[str, list[str]]]


def sample_lime_example(
    rng: np.random.Generator,
    *,
    upper_pool: str,
    lower_pool: str,
    math_pool: str,
    min_variables: int,
    max_variables: int,
    min_pattern_length: int,
    max_pattern_length: int,
    min_substitution_length: int,
    max_substitution_length: int,
) -> LIMEExample:
    variable_count = int(rng.integers(min_variables, max_variables + 1))
    upper_vocab = _sample_unique(rng, upper_pool, variable_count)

    min_lower_count = max(variable_count + 2, min(8, len(lower_pool)))
    min_math_count = max(3, min(5, len(math_pool)))
    lower_count = int(rng.integers(min_lower_count, len(lower_pool) + 1))
    math_count = int(rng.integers(min_math_count, len(math_pool) + 1))
    lower_vocab = _sample_unique(rng, lower_pool, lower_count)
    math_vocab = _sample_unique(rng, math_pool, math_count)

    pattern_length = int(rng.integers(min_pattern_length, max_pattern_length + 1))
    pattern = _sample_pattern(rng, upper_vocab, math_vocab, pattern_length)
    substitutions = _sample_substitutions(
        rng,
        variables=upper_vocab,
        lower_vocab=lower_vocab,
        math_vocab=math_vocab,
        min_length=min_substitution_length,
        max_length=max_substitution_length,
    )
    result = apply_substitutions(pattern, substitutions)
    return LIMEExample(
        upper_vocab=upper_vocab,
        lower_vocab=lower_vocab,
        math_vocab=math_vocab,
        pattern=pattern,
        result=result,
        substitution_pairs=list(substitutions.items()),
    )


def apply_substitutions(
    pattern: list[str],
    substitutions: dict[str, list[str]],
) -> list[str]:
    result: list[str] = []
    for symbol in pattern:
        replacement = substitutions.get(symbol)
        if replacement is None:
            result.append(symbol)
        else:
            result.extend(replacement)
    return result


def _sample_pattern(
    rng: np.random.Generator,
    upper_vocab: list[str],
    math_vocab: list[str],
    length: int,
) -> list[str]:
    variable_tokens = rng.choice(upper_vocab, size=max(1, len(upper_vocab) * 2), replace=True).tolist()
    math_tokens = rng.choice(math_vocab, size=length, replace=True).tolist()
    pattern = [*variable_tokens, *math_tokens]
    rng.shuffle(pattern)
    return pattern[:length]


def _sample_substitutions(
    rng: np.random.Generator,
    *,
    variables: list[str],
    lower_vocab: list[str],
    math_vocab: list[str],
    min_length: int,
    max_length: int,
) -> dict[str, list[str]]:
    combined = [*lower_vocab, *math_vocab]
    substitutions: dict[str, list[str]] = {}
    for variable in variables:
        subst_length = int(rng.integers(min_length, max_length + 1))
        substitutions[variable] = rng.choice(combined, size=subst_length, replace=True).tolist()
    return substitutions


def _sample_unique(rng: np.random.Generator, pool: str, count: int) -> list[str]:
    if count > len(pool):
        raise ValueError("Requested more unique symbols than available in the pool.")
    return rng.choice(list(pool), size=count, replace=False).tolist()


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

    def _encode_example(self, mode: str, example: LIMEExample, spec: TokenizerSpec) -> list[int]:
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
