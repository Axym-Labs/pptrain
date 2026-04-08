from __future__ import annotations

from typing import Sequence

import numpy as np

SUPPORTED_SIMPLER_TASKS = (
    "copy",
    "reverse",
    "set",
    "duplicate",
    "deduplicate",
    "length",
    "search",
    "union",
    "intersect",
    "set_1_minus_2",
    "set_2_minus_1",
)

UNARY_TASKS = {
    "copy",
    "reverse",
    "set",
    "duplicate",
    "deduplicate",
    "length",
}
BINARY_SET_TASKS = {"union", "intersect", "set_1_minus_2", "set_2_minus_1"}
BINARY_QUERY_TASKS = {"search"}


def sample_symbols(
    rng: np.random.Generator,
    alphabet: Sequence[str],
    length: int,
) -> list[str]:
    return rng.choice(list(alphabet), size=length, replace=True).tolist()


def stable_unique(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def deduplicate_consecutive(items: Sequence[str]) -> list[str]:
    result: list[str] = []
    last: str | None = None
    for item in items:
        if item == last:
            continue
        result.append(item)
        last = item
    return result


def contains_subsequence(sequence: Sequence[str], query: Sequence[str]) -> bool:
    if not query:
        return True
    query_length = len(query)
    for start in range(len(sequence) - query_length + 1):
        if list(sequence[start : start + query_length]) == list(query):
            return True
    return False


def apply_unary_task(task: str, sequence: Sequence[str]) -> list[str]:
    if task == "copy":
        return list(sequence)
    if task == "reverse":
        return list(reversed(sequence))
    if task == "set":
        return stable_unique(sequence)
    if task == "duplicate":
        return [item for symbol in sequence for item in (symbol, symbol)]
    if task == "deduplicate":
        return deduplicate_consecutive(sequence)
    if task == "length":
        return list(str(len(sequence)))
    raise KeyError(f"Unsupported unary task '{task}'")


def apply_binary_task(task: str, left: Sequence[str], right: Sequence[str]) -> list[str]:
    if task == "search":
        return ["yes"] if contains_subsequence(left, right) else ["no"]

    left_unique = stable_unique(left)
    right_unique = stable_unique(right)
    right_set = set(right)
    left_set = set(left)

    if task == "union":
        return stable_unique([*left, *right])
    if task == "intersect":
        return [item for item in left_unique if item in right_set]
    if task == "set_1_minus_2":
        return [item for item in left_unique if item not in right_set]
    if task == "set_2_minus_1":
        return [item for item in right_unique if item not in left_set]
    raise KeyError(f"Unsupported binary task '{task}'")


def sample_search_query(
    rng: np.random.Generator,
    sequence: Sequence[str],
    alphabet: Sequence[str],
    min_query_symbols: int,
    max_query_symbols: int,
    positive_probability: float,
) -> list[str]:
    max_query = min(max_query_symbols, len(sequence))
    query_length = int(rng.integers(min_query_symbols, max_query + 1))
    if rng.random() < positive_probability:
        start = int(rng.integers(0, len(sequence) - query_length + 1))
        return list(sequence[start : start + query_length])

    for _ in range(16):
        query = sample_symbols(rng, alphabet, query_length)
        if not contains_subsequence(sequence, query):
            return query
    return sample_symbols(rng, alphabet, query_length)
