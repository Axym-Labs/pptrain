from __future__ import annotations

from typing import Sequence

import numpy as np

SUPPORTED_SIMPLER_TASKS = (
    "copy",
    "identity",
    "reverse",
    "set",
    "first_token",
    "last_token",
    "duplicate",
    "deduplicate",
    "length",
    "count",
    "search",
    "delete",
    "filter",
    "get_index",
    "sort",
    "replace",
    "replace_many",
    "union",
    "intersect",
    "set_1_minus_2",
    "set_2_minus_1",
)

UNARY_TASKS = {
    "copy",
    "identity",
    "reverse",
    "set",
    "first_token",
    "last_token",
    "duplicate",
    "deduplicate",
    "length",
}
BINARY_SET_TASKS = {"union", "intersect", "set_1_minus_2", "set_2_minus_1"}
SINGLE_SYMBOL_QUERY_TASKS = {"count"}
SUBSEQUENCE_QUERY_TASKS = {"search", "delete", "filter", "get_index"}
ORDER_QUERY_TASKS = {"sort"}
REPLACEMENT_TASKS = {"replace", "replace_many"}


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
    if task == "copy" or task == "identity":
        return list(sequence)
    if task == "reverse":
        return list(reversed(sequence))
    if task == "set":
        return stable_unique(sequence)
    if task == "first_token":
        return [sequence[0]]
    if task == "last_token":
        return [sequence[-1]]
    if task == "duplicate":
        return [item for symbol in sequence for item in (symbol, symbol)]
    if task == "deduplicate":
        return deduplicate_consecutive(sequence)
    if task == "length":
        return list(str(len(sequence)))
    raise KeyError(f"Unsupported unary task '{task}'")


def apply_binary_task(task: str, left: Sequence[str], right: Sequence[str]) -> list[str]:
    if task == "count":
        if not right:
            return ["0"]
        count = sum(1 for item in left if item == right[0])
        return list(str(count))
    if task == "search":
        return ["yes"] if contains_subsequence(left, right) else ["no"]
    if task == "delete":
        return delete_first_occurrence(left, right)
    if task == "filter":
        return delete_all_occurrences(left, right)
    if task == "get_index":
        return list(str(index_of_subsequence(left, right)))
    if task == "sort":
        order = {item: position for position, item in enumerate(right)}
        return sorted(left, key=lambda item: order[item])
    if task == "replace" or task == "replace_many":
        replacements = replacement_map(right)
        return [replacements.get(item, item) for item in left]

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


def index_of_subsequence(sequence: Sequence[str], query: Sequence[str]) -> int:
    if not query:
        return 0
    query_length = len(query)
    for start in range(len(sequence) - query_length + 1):
        if list(sequence[start : start + query_length]) == list(query):
            return start
    return -1


def delete_first_occurrence(sequence: Sequence[str], query: Sequence[str]) -> list[str]:
    index = index_of_subsequence(sequence, query)
    if index < 0:
        return list(sequence)
    return list(sequence[:index]) + list(sequence[index + len(query) :])


def delete_all_occurrences(sequence: Sequence[str], query: Sequence[str]) -> list[str]:
    result = list(sequence)
    while True:
        updated = delete_first_occurrence(result, query)
        if updated == result:
            return result
        result = updated


def replacement_map(query: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for index in range(0, len(query), 2):
        if index + 1 >= len(query):
            break
        result[str(query[index])] = str(query[index + 1])
    return result


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


def sample_count_query(
    rng: np.random.Generator,
    sequence: Sequence[str],
    alphabet: Sequence[str],
) -> list[str]:
    if rng.random() < 0.8:
        return [sequence[int(rng.integers(0, len(sequence)))]]
    return [str(rng.choice(list(alphabet)))]


def sample_sort_order(
    rng: np.random.Generator,
    sequence: Sequence[str],
) -> list[str]:
    order = stable_unique(sequence)
    rng.shuffle(order)
    return order


def sample_replacement_query(
    rng: np.random.Generator,
    sequence: Sequence[str],
    alphabet: Sequence[str],
    *,
    many: bool,
) -> list[str]:
    unique_items = stable_unique(sequence)
    max_pairs = min(len(unique_items), 4 if many else 1)
    pair_count = int(rng.integers(1, max_pairs + 1))
    sources = rng.choice(unique_items, size=pair_count, replace=False).tolist()
    replacements = rng.choice(list(alphabet), size=pair_count, replace=True).tolist()
    query: list[str] = []
    for source, replacement in zip(sources, replacements):
        query.extend([source, replacement])
    return query
