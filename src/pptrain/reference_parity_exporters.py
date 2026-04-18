"""Reference-parity adapters for external paper repositories.

The integration boundary is deliberately narrow:
- this library does not vendor paper repositories
- callers pass explicit file paths or repo roots to adapter helpers
- adapters normalize external artifacts into pptrain's canonical fixtures

That keeps the core library lean while still allowing parity checks against
paper code and datasets when those repos are available locally.
"""

from __future__ import annotations

import importlib.util
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from pptrain.core.base import ExecutedSymbolicTask, SymbolicTask, SymbolicTaskFamily, Task
from pptrain.mechanisms.lime import LIMEExample
from pptrain.mechanisms.summarization.generator import DocumentExample
from pptrain.reference_nca_repo import load_nca_reference_export_from_repo


@dataclass(frozen=True, slots=True)
class TaskReferenceAdapter:
    task_name: str
    comparison_target: str
    normalize_task_examples: Callable[[Task, int | None], tuple[list[dict[str, Any]], list[dict[str, Any]]]] | None = None
    notes: str = ""


def build_normalized_task_examples(
    task: Task,
    *,
    seed: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    adapter = TASK_REFERENCE_ADAPTERS.get(task.name)
    if adapter is None or adapter.normalize_task_examples is None:
        raise KeyError(f"Task '{task.name}' does not expose normalized reference examples.")
    return adapter.normalize_task_examples(task, seed)


def build_lime_reference_fixture_from_files(
    task: Task,
    *,
    train_src_path: str | Path,
    train_tgt_path: str | Path,
    eval_src_path: str | Path | None = None,
    eval_tgt_path: str | Path | None = None,
    preset_name: str | None = None,
    seed: int | None = None,
    metadata: dict[str, Any] | None = None,
    source: Any | None = None,
    ) -> Any:
    from pptrain.reference_parity import ReferenceFixture, ReferenceSplit

    train_examples = _parse_lime_example_pairs(
        _read_reference_lines(train_src_path),
        _read_reference_lines(train_tgt_path),
    )
    eval_examples: list[dict[str, Any]] | None = None
    if eval_src_path is not None or eval_tgt_path is not None:
        if eval_src_path is None or eval_tgt_path is None:
            raise ValueError("Both eval_src_path and eval_tgt_path must be provided together.")
        eval_examples = _parse_lime_example_pairs(
            _read_reference_lines(eval_src_path),
            _read_reference_lines(eval_tgt_path),
        )
    return ReferenceFixture(
        task_name=task.name,
        comparison_target="normalized_examples",
        tokenizer_spec=task.tokenizer_spec(),
        train=ReferenceSplit(examples=train_examples),
        eval=ReferenceSplit(examples=eval_examples) if eval_examples is not None else None,
        preset_name=preset_name,
        seed=seed,
        metadata={
            "train_sequence_count": len(train_examples),
            "eval_sequence_count": len(eval_examples) if eval_examples is not None else 0,
            **(metadata or {}),
        },
        source=source,
    )


def build_procedural_reference_fixture_from_rows(
    task: Task,
    *,
    reference_task_name: str,
    train_input_rows: Sequence[Sequence[int]],
    eval_input_rows: Sequence[Sequence[int]] | None = None,
    separator_token_id: int,
    pad_token_id: int,
    preset_name: str | None = None,
    seed: int | None = None,
    metadata: dict[str, Any] | None = None,
    source: Any | None = None,
) -> Any:
    from pptrain.reference_parity import ReferenceFixture, ReferenceSplit

    train_examples = [
        parse_procedural_reference_example(
            task_name=reference_task_name,
            input_ids=row,
            separator_token_id=separator_token_id,
            pad_token_id=pad_token_id,
        )
        for row in train_input_rows
    ]
    eval_examples = (
        [
            parse_procedural_reference_example(
                task_name=reference_task_name,
                input_ids=row,
                separator_token_id=separator_token_id,
                pad_token_id=pad_token_id,
            )
            for row in eval_input_rows
        ]
        if eval_input_rows is not None
        else None
    )
    return ReferenceFixture(
        task_name=task.name,
        comparison_target="normalized_examples",
        tokenizer_spec=task.tokenizer_spec(),
        train=ReferenceSplit(examples=train_examples),
        eval=ReferenceSplit(examples=eval_examples) if eval_examples is not None else None,
        preset_name=preset_name,
        seed=seed,
        metadata={
            "reference_task_name": reference_task_name,
            "train_sequence_count": len(train_examples),
            "eval_sequence_count": len(eval_examples) if eval_examples is not None else 0,
            **(metadata or {}),
        },
        source=source,
    )


def build_procedural_reference_fixture_from_repo(
    task: Task,
    *,
    repo_root: str | Path,
    reference_task_name: str,
    seq_len: int,
    vocab_size: int,
    train_sequence_count: int,
    eval_sequence_count: int,
    seed: int = 0,
    preset_name: str | None = None,
    metadata: dict[str, Any] | None = None,
    source: Any | None = None,
) -> Any:
    train_rows = _load_procedural_reference_rows(
        repo_root=repo_root,
        reference_task_name=reference_task_name,
        seq_len=seq_len,
        vocab_size=vocab_size,
        sequence_count=train_sequence_count,
        seed=seed,
    )
    eval_rows = _load_procedural_reference_rows(
        repo_root=repo_root,
        reference_task_name=reference_task_name,
        seq_len=seq_len,
        vocab_size=vocab_size,
        sequence_count=eval_sequence_count,
        seed=seed + 10_000,
    )
    return build_procedural_reference_fixture_from_rows(
        task,
        reference_task_name=reference_task_name,
        train_input_rows=train_rows,
        eval_input_rows=eval_rows,
        separator_token_id=vocab_size,
        pad_token_id=vocab_size + 1,
        preset_name=preset_name,
        seed=seed,
        metadata=metadata,
        source=source,
    )


def build_summarization_reference_fixture_from_jsonl(
    task: Task,
    *,
    reference_task_name: str,
    train_jsonl_path: str | Path,
    eval_jsonl_path: str | Path | None = None,
    preset_name: str | None = None,
    seed: int | None = None,
    metadata: dict[str, Any] | None = None,
    source: Any | None = None,
) -> Any:
    from pptrain.reference_parity import ReferenceFixture, ReferenceSplit

    train_examples = [
        parse_summarization_reference_example(
            task_name=reference_task_name,
            article_lines=record["article_lines"],
            summary_lines=record["summary_lines"],
        )
        for record in _read_jsonl_records(train_jsonl_path)
    ]
    eval_examples = (
        [
            parse_summarization_reference_example(
                task_name=reference_task_name,
                article_lines=record["article_lines"],
                summary_lines=record["summary_lines"],
            )
            for record in _read_jsonl_records(eval_jsonl_path)
        ]
        if eval_jsonl_path is not None
        else None
    )
    return ReferenceFixture(
        task_name=task.name,
        comparison_target="normalized_examples",
        tokenizer_spec=task.tokenizer_spec(),
        train=ReferenceSplit(examples=train_examples),
        eval=ReferenceSplit(examples=eval_examples) if eval_examples is not None else None,
        preset_name=preset_name,
        seed=seed,
        metadata={
            "reference_task_name": reference_task_name,
            "train_sequence_count": len(train_examples),
            "eval_sequence_count": len(eval_examples) if eval_examples is not None else 0,
            **(metadata or {}),
        },
        source=source,
    )


def build_summarization_reference_fixture_from_repo(
    task: Task,
    *,
    repo_root: str | Path,
    dataset_name: str,
    reference_task_name: str,
    train_split_name: str = "train",
    eval_split_name: str = "val",
    preset_name: str | None = None,
    seed: int | None = None,
    metadata: dict[str, Any] | None = None,
    source: Any | None = None,
) -> Any:
    dataset_dir = _resolve_reference_dataset_dir(
        repo_root,
        dataset_name,
        candidate_roots=(
            ("dataset_root", "pretraining_datasets"),
            ("pretraining_datasets",),
        ),
    )
    return build_summarization_reference_fixture_from_jsonl(
        task,
        reference_task_name=reference_task_name,
        train_jsonl_path=dataset_dir / f"{train_split_name}.jsonl",
        eval_jsonl_path=dataset_dir / f"{eval_split_name}.jsonl",
        preset_name=preset_name,
        seed=seed,
        metadata={"dataset_name": dataset_name, **(metadata or {})},
        source=source,
    )


def build_nca_reference_fixture_from_rows(
    task: Task,
    *,
    train_sequences: Sequence[Sequence[int]],
    train_labels: Sequence[Sequence[int]],
    eval_sequences: Sequence[Sequence[int]] | None = None,
    eval_labels: Sequence[Sequence[int]] | None = None,
    preset_name: str | None = None,
    seed: int | None = None,
    metadata: dict[str, Any] | None = None,
    source: Any | None = None,
) -> Any:
    from pptrain.reference_parity import ReferenceFixture, ReferenceSplit

    if len(train_sequences) != len(train_labels):
        raise ValueError("NCA train sequences and labels must have the same number of rows.")
    if (eval_sequences is None) != (eval_labels is None):
        raise ValueError("Both eval_sequences and eval_labels must be provided together.")
    if eval_sequences is not None and eval_labels is not None and len(eval_sequences) != len(eval_labels):
        raise ValueError("NCA eval sequences and labels must have the same number of rows.")
    return ReferenceFixture(
        task_name=task.name,
        comparison_target="dataset_bundle",
        tokenizer_spec=task.tokenizer_spec(),
        train=ReferenceSplit(
            sequences=[[int(token) for token in row] for row in train_sequences],
            labels=[[int(token) for token in row] for row in train_labels],
        ),
        eval=(
            ReferenceSplit(
                sequences=[[int(token) for token in row] for row in eval_sequences],
                labels=[[int(token) for token in row] for row in eval_labels],
            )
            if eval_sequences is not None and eval_labels is not None
            else None
        ),
        preset_name=preset_name,
        seed=seed,
        metadata={
            "train_sequence_count": len(train_sequences),
            "eval_sequence_count": len(eval_sequences) if eval_sequences is not None else 0,
            **(metadata or {}),
        },
        source=source,
    )


def build_nca_reference_fixture_from_export_json(
    task: Task,
    *,
    export_json_path: str | Path,
    preset_name: str | None = None,
    seed: int | None = None,
    metadata: dict[str, Any] | None = None,
    source: Any | None = None,
) -> Any:
    payload = json.loads(Path(export_json_path).read_text(encoding="utf-8"))
    return build_nca_reference_fixture_from_export_payload(
        task,
        payload=payload,
        preset_name=preset_name,
        seed=seed,
        metadata=metadata,
        source=source,
    )


def build_nca_reference_fixture_from_export_payload(
    task: Task,
    *,
    payload: dict[str, Any],
    preset_name: str | None = None,
    seed: int | None = None,
    metadata: dict[str, Any] | None = None,
    source: Any | None = None,
) -> Any:
    train = payload.get("train")
    if not isinstance(train, dict):
        raise ValueError("NCA export JSON must define a 'train' object.")
    eval_payload = payload.get("eval")
    payload_metadata = payload.get("metadata", {})
    return build_nca_reference_fixture_from_rows(
        task,
        train_sequences=train["sequences"],
        train_labels=train["labels"],
        eval_sequences=eval_payload["sequences"] if isinstance(eval_payload, dict) else None,
        eval_labels=eval_payload["labels"] if isinstance(eval_payload, dict) else None,
        preset_name=preset_name,
        seed=seed,
        metadata={**payload_metadata, **(metadata or {})},
        source=source,
    )


def build_nca_reference_fixture_from_repo(
    task: Task,
    *,
    repo_root: str | Path,
    preset_name: str | None = None,
    seed: int | None = None,
    metadata: dict[str, Any] | None = None,
    source: Any | None = None,
    python_executable: str | None = None,
    export_json_path: str | Path | None = None,
) -> Any:
    payload = load_nca_reference_export_from_repo(
        repo_root,
        config={
            "seed": int(0 if seed is None else seed),
            "grid_size": int(getattr(task.config, "grid_size")),
            "num_states": int(getattr(task.config, "num_states")),
            "patch_size": int(getattr(task.config, "patch_size")),
            "temperature": float(getattr(task.config, "temperature")),
            "rollout_stride": int(getattr(task.config, "rollout_stride")),
            "init_rollout_steps": int(getattr(task.config, "init_rollout_steps")),
            "max_length": int(getattr(task.config, "max_length")),
            "sequence_count": int(getattr(task.config, "sequence_count")),
            "eval_sequence_count": int(getattr(task.config, "eval_sequence_count")),
            "rule_count": (
                None
                if getattr(task.config, "rule_count") is None
                else int(getattr(task.config, "rule_count"))
            ),
            "eval_rule_count": (
                None
                if getattr(task.config, "eval_rule_count") is None
                else int(getattr(task.config, "eval_rule_count"))
            ),
            "complexity_min": float(getattr(task.config, "complexity_min")),
            "complexity_max": float(getattr(task.config, "complexity_max")),
            "complexity_probe_frames": int(getattr(task.config, "complexity_probe_frames")),
            "identity_bias": float(getattr(task.config, "identity_bias")),
            "min_frames": int(getattr(task.config, "min_frames")),
        },
        python_executable=python_executable,
        output_path=export_json_path,
    )
    return build_nca_reference_fixture_from_export_payload(
        task,
        payload=payload,
        preset_name=preset_name,
        seed=seed,
        metadata=metadata,
        source=source,
    )


def parse_lime_reference_example(*, src_line: str, tgt_line: str) -> dict[str, Any]:
    tokens = src_line.strip().split()
    if not tokens:
        raise ValueError("LIME source line must not be empty.")
    upper_vocab, lower_vocab, math_vocab, segments = _parse_lime_source_tokens(tokens)
    target_tokens = tgt_line.strip().split()
    if len(segments) != 2:
        raise ValueError("LIME reference rows must contain exactly two <space>-separated body segments.")
    first_segment, second_segment = segments

    if _is_substitution_segment(second_segment):
        substitutions = _parse_lime_substitutions(second_segment)
        if any(token in upper_vocab for token in target_tokens):
            mode = "induct"
            pattern = target_tokens
            result = first_segment
        else:
            mode = "deduct"
            pattern = first_segment
            result = target_tokens
    elif _is_substitution_segment(target_tokens):
        mode = "abduct"
        pattern = first_segment
        result = second_segment
        substitutions = _parse_lime_substitutions(target_tokens)
    else:
        raise ValueError("Unable to infer LIME mode from the provided source/target lines.")

    return {
        "mode": mode,
        "upper_vocab": upper_vocab,
        "lower_vocab": lower_vocab,
        "math_vocab": math_vocab,
        "pattern": pattern,
        "result": result,
        "substitution_pairs": [[symbol, replacement] for symbol, replacement in substitutions],
    }


def parse_procedural_reference_example(
    *,
    task_name: str,
    input_ids: Sequence[int],
    separator_token_id: int,
    pad_token_id: int,
) -> dict[str, Any]:
    sections = _split_reference_sections(input_ids, separator_token_id)
    if task_name in {"copy", "identity", "reverse", "sort", "set"}:
        if len(sections) != 2:
            raise ValueError(f"Procedural reference row for '{task_name}' must contain one separator.")
        inputs = [_strip_trailing_pad(sections[0], pad_token_id)]
        target = _strip_trailing_pad(sections[1], pad_token_id)
    elif task_name in {"union", "delete"}:
        if len(sections) != 3:
            raise ValueError(f"Procedural reference row for '{task_name}' must contain two separators.")
        inputs = [
            _strip_trailing_pad(sections[0], pad_token_id),
            _strip_trailing_pad(sections[1], pad_token_id),
        ]
        target = _strip_trailing_pad(sections[2], pad_token_id)
    else:
        raise ValueError(f"Unsupported procedural reference task '{task_name}'.")
    canonical_inputs, canonical_target = _canonicalize_symbol_sequences(inputs, target)
    return {
        "task": task_name,
        "inputs": canonical_inputs,
        "target": canonical_target,
    }


def parse_summarization_reference_example(
    *,
    task_name: str,
    article_lines: Sequence[str],
    summary_lines: Sequence[str],
) -> dict[str, Any]:
    input_document = [_tokenize_summarization_reference_line(line, task_name=task_name) for line in article_lines]
    target_document = [_tokenize_summarization_reference_line(line, task_name=task_name) for line in summary_lines]
    canonical_input, canonical_target = _canonicalize_summarization_documents(input_document, target_document)
    return {
        "task": task_name,
        "input_document": canonical_input,
        "target_document": canonical_target,
    }


def _parse_lime_example_pairs(src_lines: list[str], tgt_lines: list[str]) -> list[dict[str, Any]]:
    if len(src_lines) != len(tgt_lines):
        raise ValueError("LIME source and target files must contain the same number of rows.")
    return [
        parse_lime_reference_example(src_line=src_line, tgt_line=tgt_line)
        for src_line, tgt_line in zip(src_lines, tgt_lines)
    ]


def _normalize_lime_examples(task: Task, seed: int | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return _normalize_symbolic_task_examples(task, seed=seed, normalizer=_normalize_lime_example)


def _normalize_procedural_examples(task: Task, seed: int | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return _normalize_symbolic_task_examples(task, seed=seed, normalizer=_normalize_procedural_example)


def _normalize_summarization_examples(task: Task, seed: int | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return _normalize_symbolic_task_examples(task, seed=seed, normalizer=_normalize_summarization_example)


def _normalize_symbolic_task_examples(
    task: Task,
    *,
    seed: int | None,
    normalizer: Callable[[Task, SymbolicTask, ExecutedSymbolicTask], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(task, SymbolicTaskFamily):
        raise TypeError(f"Task '{task.name}' is not a symbolic task family.")
    rng = np.random.default_rng(seed)
    spec = task.tokenizer_spec()
    max_length = getattr(task.config, "max_length")

    def sample_split(count: int) -> list[dict[str, Any]]:
        examples: list[dict[str, Any]] = []
        for _ in range(count):
            for _attempt in range(task.max_sampling_attempts):
                symbolic = task.sample_task(rng)
                executed = task.execute_task(symbolic)
                tokens = task.serialize_task(executed, spec)
                if len(tokens) <= max_length + 1:
                    examples.append(normalizer(task, symbolic, executed))
                    break
            else:  # pragma: no cover - delegated to existing task sampling checks
                raise RuntimeError(
                    f"Failed to sample a normalized reference example for task '{task.name}' within max_length={max_length}."
                )
        return examples

    train_examples = sample_split(getattr(task.config, "sequence_count"))
    eval_examples = sample_split(getattr(task.config, "eval_sequence_count"))
    return train_examples, eval_examples


def _normalize_lime_example(task: Task, symbolic: SymbolicTask, executed: ExecutedSymbolicTask) -> dict[str, Any]:
    example = executed.payload
    if not isinstance(example, LIMEExample):
        raise TypeError("Expected a LIMEExample payload for LIME reference normalization.")
    return {
        "mode": symbolic.name,
        "upper_vocab": list(example.upper_vocab),
        "lower_vocab": list(example.lower_vocab),
        "math_vocab": list(example.math_vocab),
        "pattern": list(example.pattern),
        "result": list(example.result),
        "substitution_pairs": [[name, list(value)] for name, value in example.substitution_pairs],
    }


def _normalize_procedural_example(task: Task, symbolic: SymbolicTask, executed: ExecutedSymbolicTask) -> dict[str, Any]:
    task_name, remainder = str(executed.payload).split(":", 1)
    source, target = remainder.split("=>", 1)
    if task_name == "addition":
        left, right = source.split("+", 1)
        return {
            "task": task_name,
            "inputs": [left, right],
            "target": target,
        }
    if task_name in {"union", "delete"}:
        left, right = source.split("|", 1)
        inputs, canonical_target = _canonicalize_symbol_sequences([list(left), list(right)], list(target))
        return {
            "task": task_name,
            "inputs": inputs,
            "target": canonical_target,
        }
    inputs, canonical_target = _canonicalize_symbol_sequences([list(source)], list(target))
    return {
        "task": task_name,
        "inputs": inputs,
        "target": canonical_target,
    }


def _normalize_summarization_example(task: Task, symbolic: SymbolicTask, executed: ExecutedSymbolicTask) -> dict[str, Any]:
    example = executed.payload
    if not isinstance(example, DocumentExample):
        raise TypeError("Expected a DocumentExample payload for summarization reference normalization.")
    special_tokens = _summarization_special_token_mapping(task)
    input_document = [
        [_normalize_summarization_task_token(token, special_tokens) for token in sentence]
        for sentence in example.input_document
    ]
    target_document = [
        [_normalize_summarization_task_token(token, special_tokens) for token in sentence]
        for sentence in example.target_document
    ]
    canonical_input, canonical_target = _canonicalize_summarization_documents(input_document, target_document)
    return {
        "task": example.task,
        "input_document": canonical_input,
        "target_document": canonical_target,
    }


def _parse_lime_source_tokens(tokens: list[str]) -> tuple[list[str], list[str], list[str], list[list[str]]]:
    try:
        lower_index = tokens.index("<LOWER>")
        math_index = tokens.index("<MATH>")
        space_index = tokens.index("<space>")
    except ValueError as exc:
        raise ValueError("LIME source line is missing one of <LOWER>, <MATH>, or <space>.") from exc
    if lower_index <= 0 or tokens[0] != "<UPPER>":
        raise ValueError("LIME source line must start with <UPPER>.")
    upper_vocab = tokens[1:lower_index]
    lower_vocab = tokens[lower_index + 1 : math_index]
    math_vocab = tokens[math_index + 1 : space_index]
    body_tokens = tokens[space_index + 1 :]
    segments: list[list[str]] = []
    current: list[str] = []
    for token in body_tokens:
        if token == "<space>":
            segments.append(current)
            current = []
        else:
            current.append(token)
    segments.append(current)
    return upper_vocab, lower_vocab, math_vocab, segments


def _split_reference_sections(input_ids: Sequence[int], separator_token_id: int) -> list[list[int]]:
    sections: list[list[int]] = []
    current: list[int] = []
    for token in input_ids:
        value = int(token)
        if value == separator_token_id:
            sections.append(current)
            current = []
        else:
            current.append(value)
    sections.append(current)
    return sections


def _strip_trailing_pad(tokens: Sequence[int], pad_token_id: int) -> list[int]:
    trimmed = [int(token) for token in tokens]
    while trimmed and trimmed[-1] == pad_token_id:
        trimmed.pop()
    return trimmed


def _canonicalize_symbol_sequences(
    inputs: list[list[Any]],
    target: list[Any],
) -> tuple[list[list[int]], list[int]]:
    symbol_ids: dict[Any, int] = {}

    def encode(sequence: list[Any]) -> list[int]:
        encoded: list[int] = []
        for symbol in sequence:
            if symbol not in symbol_ids:
                symbol_ids[symbol] = len(symbol_ids)
            encoded.append(symbol_ids[symbol])
        return encoded

    canonical_inputs = [encode(sequence) for sequence in inputs]
    canonical_target = encode(target)
    return canonical_inputs, canonical_target


def _read_reference_lines(path: str | Path) -> list[str]:
    return [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _resolve_reference_dataset_dir(
    repo_root: str | Path,
    dataset_name: str,
    *,
    candidate_roots: Sequence[Sequence[str]],
) -> Path:
    repo_path = Path(repo_root)
    for root_parts in candidate_roots:
        candidate = repo_path.joinpath(*root_parts, dataset_name)
        if candidate.exists():
            return candidate
    searched = [str(repo_path.joinpath(*parts, dataset_name)) for parts in candidate_roots]
    raise FileNotFoundError(
        f"Could not find reference dataset '{dataset_name}'. Searched: {searched}"
    )


def _load_procedural_reference_rows(
    *,
    repo_root: str | Path,
    reference_task_name: str,
    seq_len: int,
    vocab_size: int,
    sequence_count: int,
    seed: int,
) -> list[list[int]]:
    dataset_class = _load_procedural_reference_dataset_class(repo_root, reference_task_name)
    dataset = dataset_class(
        seq_len=seq_len,
        vocab_size=vocab_size,
        num_samples=sequence_count,
        seed=seed,
    )
    rows: list[list[int]] = []
    for index in range(sequence_count):
        sample = dataset[index]
        rows.append([int(token) for token in sample["input_ids"]])
    return rows


def _load_procedural_reference_dataset_class(repo_root: str | Path, reference_task_name: str) -> type:
    module_path = Path(repo_root) / "procedural_data" / f"{reference_task_name}.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Procedural reference module not found: {module_path}")
    module_name = f"_pptrain_reference_procedural_{reference_task_name}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load procedural reference module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class_name = _PROCEDURAL_REFERENCE_DATASET_CLASSES.get(reference_task_name)
    if class_name is None:
        raise ValueError(f"Unsupported procedural reference task '{reference_task_name}'.")
    if not hasattr(module, class_name):
        raise AttributeError(f"Procedural reference module '{module_path}' does not define '{class_name}'.")
    return getattr(module, class_name)


def _tokenize_summarization_reference_line(line: str, *, task_name: str) -> list[Any]:
    tokens = line.strip().split()
    if tokens and tokens[-1] == ".":
        tokens = tokens[:-1]
    return [_normalize_summarization_reference_token(token, task_name=task_name) for token in tokens]


def _normalize_summarization_reference_token(token: str, *, task_name: str) -> Any:
    if token == "bullet":
        return "bullet"
    if token == "begin_quote":
        return "quote_open"
    if token == "end_quote":
        return "quote_close"
    if token == "<extra_id_0>":
        return "mask"
    keyword_match = _SUMMARIZATION_KEYWORD_RE.fullmatch(token)
    if keyword_match is not None:
        if task_name == "truncate_sentence":
            return "cutoff"
        return ("keyword", int(keyword_match.group(1)))
    return token


def _summarization_special_token_mapping(task: Task) -> dict[int, Any]:
    spec = task.tokenizer_spec()
    mapping: dict[int, Any] = {}
    for name, token_id in spec.extra_token_ids.items():
        if name in {"mask", "bullet", "quote_open", "quote_close", "cutoff"}:
            mapping[token_id] = name
        elif name.startswith("keyword:"):
            mapping[token_id] = ("keyword", int(name.split(":", 1)[1]))
    return mapping


def _normalize_summarization_task_token(token: int, special_tokens: dict[int, Any]) -> Any:
    return special_tokens.get(int(token), int(token))


def _canonicalize_summarization_documents(
    input_document: list[list[Any]],
    target_document: list[list[Any]],
) -> tuple[list[list[Any]], list[list[Any]]]:
    keyword_values = sorted(
        {
            int(symbol[1])
            for sentence in [*input_document, *target_document]
            for symbol in sentence
            if isinstance(symbol, tuple) and len(symbol) == 2 and symbol[0] == "keyword"
        }
    )
    keyword_mapping = {value: index for index, value in enumerate(keyword_values)}
    ordinary_mapping: dict[Any, int] = {}

    def canonicalize_symbol(symbol: Any) -> Any:
        if isinstance(symbol, tuple) and len(symbol) == 2 and symbol[0] == "keyword":
            return f"keyword:{keyword_mapping[int(symbol[1])]}"
        if symbol in _SUMMARIZATION_STRUCTURE_MARKERS:
            return symbol
        if symbol not in ordinary_mapping:
            ordinary_mapping[symbol] = len(ordinary_mapping)
        return ordinary_mapping[symbol]

    canonical_input = [[canonicalize_symbol(symbol) for symbol in sentence] for sentence in input_document]
    canonical_target = [[canonicalize_symbol(symbol) for symbol in sentence] for sentence in target_document]
    return canonical_input, canonical_target


def _is_substitution_segment(tokens: list[str]) -> bool:
    return ":" in tokens and "[" in tokens and "]" in tokens


def _parse_lime_substitutions(tokens: list[str]) -> list[tuple[str, list[str]]]:
    pairs: list[tuple[str, list[str]]] = []
    index = 0
    while index < len(tokens):
        if index + 3 >= len(tokens):
            raise ValueError("Malformed LIME substitution segment.")
        symbol = tokens[index]
        if tokens[index + 1] != ":" or tokens[index + 2] != "[":
            raise ValueError("Malformed LIME substitution segment.")
        index += 3
        replacement: list[str] = []
        while index < len(tokens) and tokens[index] != "]":
            replacement.append(tokens[index])
            index += 1
        if index >= len(tokens):
            raise ValueError("Malformed LIME substitution segment.")
        index += 1
        pairs.append((symbol, replacement))
        if index < len(tokens):
            if tokens[index] != ",":
                raise ValueError("Malformed LIME substitution segment.")
            index += 1
    return pairs


TASK_REFERENCE_ADAPTERS: dict[str, TaskReferenceAdapter] = {
    "lime": TaskReferenceAdapter(
        task_name="lime",
        comparison_target="normalized_examples",
        normalize_task_examples=_normalize_lime_examples,
        notes="Compare symbolic substitution examples after canonical normalization.",
    ),
    "summarization": TaskReferenceAdapter(
        task_name="summarization",
        comparison_target="normalized_examples",
        normalize_task_examples=_normalize_summarization_examples,
        notes="Compare transformed documents structurally rather than by local token ids.",
    ),
    "procedural": TaskReferenceAdapter(
        task_name="procedural",
        comparison_target="normalized_examples",
        normalize_task_examples=_normalize_procedural_examples,
        notes="Compare procedural input/output mappings independent of serialization details.",
    ),
    "nca": TaskReferenceAdapter(
        task_name="nca",
        comparison_target="dataset_bundle",
        normalize_task_examples=None,
        notes="Compare materialized sequence/label bundles directly.",
    ),
}


_PROCEDURAL_REFERENCE_DATASET_CLASSES = {
    "identity": "IdentityDataset",
    "reverse": "ReverseDataset",
    "sort": "SortDataset",
    "set": "SetDataset",
    "union": "UnionDataset",
    "delete": "DeleteDataset",
}


_SUMMARIZATION_KEYWORD_RE = re.compile(r"(?:__d\d+__)?keyword_(\d+)__?")
_SUMMARIZATION_STRUCTURE_MARKERS = {"mask", "bullet", "quote_open", "quote_close", "cutoff"}
