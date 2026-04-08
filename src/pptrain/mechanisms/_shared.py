from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from pptrain.core.base import TokenizerSpec


@dataclass(frozen=True, slots=True)
class TokenVocabulary:
    groups: dict[str, dict[str, int]]
    tokens: dict[str, int]
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int

    @property
    def vocab_size(self) -> int:
        return self.pad_token_id + 1

    def group(self, name: str) -> dict[str, int]:
        return self.groups[name]

    def token(self, name: str) -> int:
        return self.tokens[name]

    def token_ids(self, names: Iterable[str]) -> dict[str, int]:
        return {name: self.tokens[name] for name in names}

    def encode_group(self, name: str, items: Iterable[str]) -> list[int]:
        mapping = self.group(name)
        return [mapping[item] for item in items]

    def encode_any(self, items: Iterable[str], *, groups: Sequence[str]) -> list[int]:
        mappings = [self.group(name) for name in groups]
        encoded: list[int] = []
        for item in items:
            for mapping in mappings:
                token_id = mapping.get(item)
                if token_id is not None:
                    encoded.append(token_id)
                    break
            else:
                raise KeyError(f"Unknown token '{item}'")
        return encoded

    def tokenizer_spec(self, *, extra_token_ids: Mapping[str, int] | None = None) -> TokenizerSpec:
        return TokenizerSpec(
            vocab_size=self.vocab_size,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            extra_token_ids=dict(extra_token_ids or {}),
        )


class TokenVocabularyBuilder:
    def __init__(self, start_id: int = 0) -> None:
        self._next_id = start_id
        self._groups: dict[str, dict[str, int]] = {}
        self._tokens: dict[str, int] = {}

    def add_group(self, name: str, symbols: Sequence[str]) -> dict[str, int]:
        if len(set(symbols)) != len(symbols):
            raise ValueError(f"Group '{name}' contains duplicate symbols.")
        mapping = {symbol: self._next_id + idx for idx, symbol in enumerate(symbols)}
        self._next_id += len(mapping)
        self._groups[name] = mapping
        return mapping

    def add_tokens(self, *names: str) -> dict[str, int]:
        mapping: dict[str, int] = {}
        for name in names:
            if name in self._tokens:
                raise ValueError(f"Token '{name}' was already registered.")
            mapping[name] = self._next_id
            self._tokens[name] = self._next_id
            self._next_id += 1
        return mapping

    def build(
        self,
        *,
        bos_token: str = "bos",
        eos_token: str = "eos",
        pad_token: str = "pad",
    ) -> TokenVocabulary:
        for name in (bos_token, eos_token, pad_token):
            if name not in self._tokens:
                raise ValueError(f"Special token '{name}' was not registered.")
        return TokenVocabulary(
            groups=dict(self._groups),
            tokens=dict(self._tokens),
            bos_token_id=self._tokens[bos_token],
            eos_token_id=self._tokens[eos_token],
            pad_token_id=self._tokens[pad_token],
        )


def require_non_empty(name: str, value: Sequence[object] | str) -> None:
    if not value:
        raise ValueError(f"{name} must not be empty.")


def require_supported(name: str, selected: Iterable[str], supported: Iterable[str]) -> None:
    unknown = sorted(set(selected) - set(supported))
    if unknown:
        raise ValueError(f"Unsupported {name}: {unknown}")


def require_positive_range(lower_name: str, lower: int, upper_name: str, upper: int) -> None:
    if lower < 1 or upper < lower:
        raise ValueError(f"{lower_name} and {upper_name} must define a valid positive range.")


def require_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1].")


def require_unique_characters(name: str, value: str, *, allow_whitespace: bool) -> None:
    if len(set(value)) != len(value):
        raise ValueError(f"{name} must contain unique characters.")
    if not allow_whitespace and any(char.isspace() for char in value):
        raise ValueError(f"{name} must not contain whitespace.")


def require_subset(name: str, value: str, allowed: str) -> None:
    if not set(value).issubset(set(allowed)):
        raise ValueError(f"{name} contains unsupported characters.")
