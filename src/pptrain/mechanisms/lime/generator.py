from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
    # Mirror the reference setup: the pattern mixes variables and math symbols.
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
