from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from pptrain.core.base import TokenSequenceMechanism, TokenizerSpec
from pptrain.core.registry import register_mechanism
from pptrain.mechanisms.lime.config import LIMEConfig
from pptrain.mechanisms.lime.generator import LIMEExample, sample_lime_example


@dataclass(slots=True)
class _Vocabulary:
    upper_to_id: dict[str, int]
    lower_to_id: dict[str, int]
    math_to_id: dict[str, int]
    token_to_id: dict[str, int]
    mode_to_id: dict[str, int]
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int


class LIMEMechanism(TokenSequenceMechanism):
    name = "lime"
    description = "LIME-style induction, deduction, and abduction substitution tasks."

    def __init__(self, config: LIMEConfig) -> None:
        super().__init__(config)
        if not self.config.modes:
            raise ValueError("At least one LIME mode must be configured.")
        unknown_modes = sorted(set(self.config.modes) - {"induct", "deduct", "abduct"})
        if unknown_modes:
            raise ValueError(f"Unsupported LIME modes: {unknown_modes}")
        if self.config.min_variables < 1 or self.config.max_variables < self.config.min_variables:
            raise ValueError("min_variables and max_variables must define a valid positive range.")
        if self.config.min_pattern_length < 1 or self.config.max_pattern_length < self.config.min_pattern_length:
            raise ValueError("min_pattern_length and max_pattern_length must define a valid positive range.")
        if (
            self.config.min_substitution_length < 1
            or self.config.max_substitution_length < self.config.min_substitution_length
        ):
            raise ValueError(
                "min_substitution_length and max_substitution_length must define a valid positive range."
            )
        if len(set(self.config.upper_symbols)) != len(self.config.upper_symbols):
            raise ValueError("upper_symbols must contain unique tokens.")
        if len(set(self.config.lower_symbols)) != len(self.config.lower_symbols):
            raise ValueError("lower_symbols must contain unique tokens.")
        if len(set(self.config.math_symbols)) != len(self.config.math_symbols):
            raise ValueError("math_symbols must contain unique tokens.")
        self._vocabulary = self._build_vocabulary()

    def tokenizer_spec(self) -> TokenizerSpec:
        vocab = self._vocabulary
        extra_token_ids = {
            token: token_id
            for token, token_id in vocab.token_to_id.items()
        }
        extra_token_ids.update({f"mode:{mode}": token_id for mode, token_id in vocab.mode_to_id.items()})
        return TokenizerSpec(
            vocab_size=vocab.pad_token_id + 1,
            pad_token_id=vocab.pad_token_id,
            bos_token_id=vocab.bos_token_id,
            eos_token_id=vocab.eos_token_id,
            extra_token_ids=extra_token_ids,
        )

    def sample_tokens(
        self,
        rng: np.random.Generator,
        spec: TokenizerSpec,
    ) -> tuple[list[int], dict[str, int | str]]:
        for _ in range(32):
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
            tokens = self._encode_example(mode, example, spec)
            if len(tokens) <= self.config.max_length + 1:
                return tokens, {
                    "mode": mode,
                    "variable_count": len(example.upper_vocab),
                    "pattern_length": len(example.pattern),
                    "result_length": len(example.result),
                }
        raise RuntimeError("Failed to sample a LIME example within max_length.")

    def _split_metadata(self, split: str, items: list[dict[str, int | str]]) -> dict[str, object]:
        mode_counts = Counter(str(item["mode"]) for item in items)
        pattern_lengths = [int(item["pattern_length"]) for item in items]
        result_lengths = [int(item["result_length"]) for item in items]
        return {
            f"{split}_mode_counts": dict(mode_counts),
            f"{split}_avg_pattern_length": float(np.mean(pattern_lengths)) if pattern_lengths else None,
            f"{split}_avg_result_length": float(np.mean(result_lengths)) if result_lengths else None,
        }

    def _encode_example(
        self,
        mode: str,
        example: LIMEExample,
        spec: TokenizerSpec,
    ) -> list[int]:
        vocab = self._vocabulary
        prompt = [
            vocab.mode_to_id[mode],
            vocab.token_to_id["declare_upper"],
            *self._encode_symbols(example.upper_vocab),
            vocab.token_to_id["section_sep"],
            vocab.token_to_id["declare_lower"],
            *self._encode_symbols(example.lower_vocab),
            vocab.token_to_id["section_sep"],
            vocab.token_to_id["declare_math"],
            *self._encode_symbols(example.math_vocab),
            vocab.token_to_id["section_sep"],
        ]
        if mode == "induct":
            input_body = [
                *self._encode_symbols(example.result),
                vocab.token_to_id["section_sep"],
                *self._encode_substitution_pairs(example.substitution_pairs),
            ]
            target_body = self._encode_symbols(example.pattern)
        elif mode == "deduct":
            input_body = [
                *self._encode_symbols(example.pattern),
                vocab.token_to_id["section_sep"],
                *self._encode_substitution_pairs(example.substitution_pairs),
            ]
            target_body = self._encode_symbols(example.result)
        elif mode == "abduct":
            input_body = [
                *self._encode_symbols(example.pattern),
                vocab.token_to_id["section_sep"],
                *self._encode_symbols(example.result),
            ]
            target_body = self._encode_substitution_pairs(example.substitution_pairs)
        else:
            raise AssertionError(f"Unhandled mode '{mode}'")

        return [
            spec.bos_token_id or 0,
            *prompt,
            *input_body,
            vocab.token_to_id["output_sep"],
            *target_body,
            spec.eos_token_id or 1,
        ]

    def _encode_substitution_pairs(self, pairs: list[tuple[str, list[str]]]) -> list[int]:
        vocab = self._vocabulary
        encoded: list[int] = []
        for index, (variable, substitution) in enumerate(pairs):
            if index > 0:
                encoded.append(vocab.token_to_id["pair_sep"])
            encoded.append(vocab.upper_to_id[variable])
            encoded.append(vocab.token_to_id["maps_to"])
            encoded.extend(self._encode_symbols(substitution))
        return encoded

    def _encode_symbols(self, symbols: list[str]) -> list[int]:
        encoded: list[int] = []
        for symbol in symbols:
            if symbol in self._vocabulary.upper_to_id:
                encoded.append(self._vocabulary.upper_to_id[symbol])
            elif symbol in self._vocabulary.lower_to_id:
                encoded.append(self._vocabulary.lower_to_id[symbol])
            elif symbol in self._vocabulary.math_to_id:
                encoded.append(self._vocabulary.math_to_id[symbol])
            else:
                raise KeyError(f"Unknown LIME symbol '{symbol}'")
        return encoded

    def _build_vocabulary(self) -> _Vocabulary:
        next_token_id = 0
        upper_to_id = {symbol: next_token_id + idx for idx, symbol in enumerate(self.config.upper_symbols)}
        next_token_id += len(upper_to_id)
        lower_to_id = {symbol: next_token_id + idx for idx, symbol in enumerate(self.config.lower_symbols)}
        next_token_id += len(lower_to_id)
        math_to_id = {symbol: next_token_id + idx for idx, symbol in enumerate(self.config.math_symbols)}
        next_token_id += len(math_to_id)
        mode_to_id = {mode: next_token_id + idx for idx, mode in enumerate(self.config.modes)}
        next_token_id += len(mode_to_id)
        token_names = (
            "declare_upper",
            "declare_lower",
            "declare_math",
            "section_sep",
            "output_sep",
            "maps_to",
            "pair_sep",
        )
        token_to_id = {name: next_token_id + idx for idx, name in enumerate(token_names)}
        next_token_id += len(token_to_id)
        bos_token_id = next_token_id
        eos_token_id = next_token_id + 1
        pad_token_id = next_token_id + 2
        return _Vocabulary(
            upper_to_id=upper_to_id,
            lower_to_id=lower_to_id,
            math_to_id=math_to_id,
            token_to_id=token_to_id,
            mode_to_id=mode_to_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )


register_mechanism(
    "lime",
    lambda config: LIMEMechanism(LIMEConfig(**config)),
    description=LIMEMechanism.description,
)
