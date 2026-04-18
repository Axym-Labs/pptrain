from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pptrain.core.base import Task, TokenizerSpec
from pptrain.reference_parity_exporters import build_normalized_task_examples

FIXTURE_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class ReferenceSource:
    repo: str
    generator: str
    commit: str | None = None
    command: str | None = None
    notes: str | None = None


@dataclass(frozen=True, slots=True)
class ReferenceSplit:
    sequences: list[list[int]] | None = None
    labels: list[list[int]] | None = None
    examples: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ReferenceFixture:
    task_name: str
    tokenizer_spec: TokenizerSpec
    train: ReferenceSplit
    eval: ReferenceSplit | None = None
    comparison_target: str = "dataset_bundle"
    preset_name: str | None = None
    seed: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    source: ReferenceSource | None = None
    format_version: int = FIXTURE_FORMAT_VERSION


class ReferenceFixtureMismatch(AssertionError):
    pass


@dataclass(frozen=True, slots=True)
class ReferenceExporterSpec:
    task_name: str
    reference_repo: str
    generator_hint: str
    recommended_presets: tuple[str, ...]
    notes: str = ""
    comparison_target: str = "dataset_bundle"
    expected_splits: tuple[str, ...] = ("train", "eval")
    fixture_format_version: int = FIXTURE_FORMAT_VERSION


REFERENCE_EXPORTER_SPECS: dict[str, ReferenceExporterSpec] = {
    "lime": ReferenceExporterSpec(
        task_name="lime",
        reference_repo="https://github.com/tonywu95/LIME",
        generator_hint="reason/generate_data.py and rewrite_multi/generate_data.py",
        recommended_presets=("paper_benchmark_100k", "paper_mixed_5m"),
        comparison_target="normalized_examples",
        notes=(
            "Export fully materialized serialized examples after mixed-task sampling. "
            "When comparing rewrite-multi variants, record the exact source mode in the fixture metadata."
        ),
    ),
    "summarization": ReferenceExporterSpec(
        task_name="summarization",
        reference_repo="https://github.com/acmi-lab/pretraining-with-nonsense",
        generator_hint="pretraining_datasetgen/*",
        recommended_presets=("paper_ourtasks_subset_100k", "paper_copy_quoted_100k"),
        comparison_target="normalized_examples",
        notes=(
            "Export pretraining examples only; no model training artifacts are required. "
            "Keep the selected synthetic task family in fixture metadata."
        ),
    ),
    "procedural": ReferenceExporterSpec(
        task_name="procedural",
        reference_repo="https://github.com/zlshinnick/procedural-pretraining",
        generator_hint="procedural_data/ and procedural_pretraining/configs/*.yaml",
        recommended_presets=("paper_set_len64", "paper_reverse_len32"),
        comparison_target="normalized_examples",
        notes=(
            "Export the generated procedural strings after task execution and tokenizer serialization. "
            "Needle or semantic downstream runs are out of scope for generator parity."
        ),
    ),
    "nca": ReferenceExporterSpec(
        task_name="nca",
        reference_repo="https://github.com/danihyunlee/nca-pre-pretraining",
        generator_hint="src/nca_ppt.py",
        recommended_presets=("paper_web_text", "paper_code"),
        notes=(
            "Export concrete train and eval token sequences after rule filtering and trajectory serialization. "
            "Because the reference code resamples rules over epochs, generator-parity fixtures should represent one "
            "materialized split at a fixed seed rather than a whole training run."
        ),
    ),
}


def fixture_from_task(
    task: Task,
    *,
    preset_name: str | None = None,
    seed: int | None = None,
    comparison_target: str | None = None,
    metadata: dict[str, Any] | None = None,
    source: ReferenceSource | None = None,
) -> ReferenceFixture:
    resolved_target = comparison_target or "dataset_bundle"
    bundle = task.build_datasets(seed=seed)
    if resolved_target == "normalized_examples":
        train_examples, eval_examples = build_normalized_task_examples(task, seed=seed)
        train_split = ReferenceSplit(examples=train_examples)
        eval_split = ReferenceSplit(examples=eval_examples) if bundle.eval_dataset is not None else None
    else:
        train_split = _fixture_split_from_dataset(bundle.train_dataset)
        eval_split = _fixture_split_from_dataset(bundle.eval_dataset) if bundle.eval_dataset is not None else None
    return ReferenceFixture(
        task_name=task.name,
        comparison_target=resolved_target,
        preset_name=preset_name,
        seed=seed,
        tokenizer_spec=task.tokenizer_spec(),
        train=train_split,
        eval=eval_split,
        metadata=dict(bundle.metadata if metadata is None else metadata),
        source=source,
    )


def save_reference_fixture(fixture: ReferenceFixture, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_fixture_to_payload(fixture), indent=2), encoding="utf-8")
    return output_path


def load_reference_fixture(path: str | Path) -> ReferenceFixture:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require_fields(
        payload,
        required=("format_version", "task_name", "tokenizer_spec", "train"),
    )
    if payload["format_version"] != FIXTURE_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported reference fixture format_version={payload['format_version']}; "
            f"expected {FIXTURE_FORMAT_VERSION}."
        )
    return ReferenceFixture(
        task_name=str(payload["task_name"]),
        comparison_target=str(payload.get("comparison_target", "dataset_bundle")),
        preset_name=_optional_string(payload.get("preset_name")),
        seed=_optional_int(payload.get("seed")),
        tokenizer_spec=_tokenizer_spec_from_payload(payload["tokenizer_spec"]),
        train=_split_from_payload(payload["train"]),
        eval=_split_from_payload(payload["eval"]) if payload.get("eval") is not None else None,
        metadata=dict(payload.get("metadata", {})),
        source=_source_from_payload(payload.get("source")),
        format_version=int(payload["format_version"]),
    )


def assert_task_fixture_matches(task: Task, fixture: ReferenceFixture) -> None:
    if task.name != fixture.task_name:
        raise ReferenceFixtureMismatch(
            f"Task name mismatch: actual '{task.name}' != reference '{fixture.task_name}'."
        )
    actual_spec = task.tokenizer_spec().to_dict()
    expected_spec = fixture.tokenizer_spec.to_dict()
    if actual_spec != expected_spec:
        raise ReferenceFixtureMismatch(
            f"Tokenizer spec mismatch: actual {actual_spec!r} != reference {expected_spec!r}."
        )
    bundle = task.build_datasets(seed=fixture.seed)
    if fixture.comparison_target == "normalized_examples":
        _assert_normalized_fixture_matches(task, fixture)
    else:
        _assert_dataset_fixture_matches(bundle, fixture)
    for key, expected_value in fixture.metadata.items():
        actual_value = bundle.metadata.get(key)
        if actual_value != expected_value:
            raise ReferenceFixtureMismatch(
                f"metadata['{key}'] mismatch: actual {actual_value!r} != reference {expected_value!r}."
            )


def _assert_normalized_fixture_matches(task: Task, fixture: ReferenceFixture) -> None:
    train_examples, eval_examples = build_normalized_task_examples(task, seed=fixture.seed)
    _assert_examples_match("train", train_examples, fixture.train)
    if fixture.eval is None:
        if eval_examples:
            raise ReferenceFixtureMismatch("Reference fixture omits eval split but task produced one.")
    else:
        _assert_examples_match("eval", eval_examples, fixture.eval)


def _assert_dataset_fixture_matches(bundle: object, fixture: ReferenceFixture) -> None:
    _assert_split_matches("train", bundle.train_dataset, bundle.metadata, fixture.train)
    if fixture.eval is None:
        if bundle.eval_dataset is not None:
            raise ReferenceFixtureMismatch("Reference fixture omits eval split but task produced one.")
    else:
        if bundle.eval_dataset is None:
            raise ReferenceFixtureMismatch("Reference fixture expects eval split but task produced none.")
        _assert_split_matches("eval", bundle.eval_dataset, bundle.metadata, fixture.eval)
def _assert_examples_match(split_name: str, actual: list[dict[str, Any]], reference: ReferenceSplit) -> None:
    if reference.examples is None:
        raise ReferenceFixtureMismatch(f"{split_name} reference split omits normalized examples.")
    if actual != reference.examples:
        mismatch_index = _first_example_mismatch_index(actual, reference.examples)
        raise ReferenceFixtureMismatch(f"{split_name} normalized examples differ at index {mismatch_index}.")


def _assert_split_matches(
    split_name: str,
    dataset: object,
    bundle_metadata: dict[str, Any],
    reference: ReferenceSplit,
) -> None:
    if reference.sequences is None:
        raise ReferenceFixtureMismatch(f"{split_name} reference split omits sequences.")
    sequences = _coerce_sequence_field(dataset, "sequences")
    labels = _coerce_optional_sequence_field(dataset, "labels")
    if sequences != reference.sequences:
        mismatch_index = _first_mismatch_index(sequences, reference.sequences)
        raise ReferenceFixtureMismatch(f"{split_name} sequences differ at index {mismatch_index}.")
    if reference.labels is not None and labels != reference.labels:
        mismatch_index = _first_mismatch_index(labels or [], reference.labels)
        raise ReferenceFixtureMismatch(f"{split_name} labels differ at index {mismatch_index}.")
    for key, expected_value in reference.metadata.items():
        bundle_key = f"{split_name}_{key}"
        actual_value = bundle_metadata.get(bundle_key)
        if actual_value != expected_value:
            raise ReferenceFixtureMismatch(
                f"{split_name} metadata['{key}'] mismatch: actual {actual_value!r} != reference {expected_value!r}."
            )


def _first_mismatch_index(left: list[list[int]], right: list[list[int]]) -> int:
    for index, (left_item, right_item) in enumerate(zip(left, right)):
        if left_item != right_item:
            return index
    return min(len(left), len(right))


def _first_example_mismatch_index(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> int:
    for index, (left_item, right_item) in enumerate(zip(left, right)):
        if left_item != right_item:
            return index
    return min(len(left), len(right))


def _coerce_sequence_field(dataset: object, field_name: str) -> list[list[int]]:
    if not hasattr(dataset, field_name):
        raise TypeError(f"Dataset object must expose '{field_name}' for reference parity.")
    value = getattr(dataset, field_name)
    return [list(map(int, item)) for item in list(value)]


def _coerce_optional_sequence_field(dataset: object, field_name: str) -> list[list[int]] | None:
    value = getattr(dataset, field_name, None)
    if value is None:
        return None
    return [list(map(int, item)) for item in list(value)]


def _fixture_split_from_dataset(dataset: object) -> ReferenceSplit:
    return ReferenceSplit(
        sequences=_coerce_sequence_field(dataset, "sequences"),
        labels=_coerce_optional_sequence_field(dataset, "labels"),
    )


def _fixture_to_payload(fixture: ReferenceFixture) -> dict[str, Any]:
    return {
        "format_version": fixture.format_version,
        "task_name": fixture.task_name,
        "comparison_target": fixture.comparison_target,
        "preset_name": fixture.preset_name,
        "seed": fixture.seed,
        "tokenizer_spec": fixture.tokenizer_spec.to_dict(),
        "train": _split_to_payload(fixture.train),
        "eval": _split_to_payload(fixture.eval) if fixture.eval is not None else None,
        "metadata": dict(fixture.metadata),
        "source": _source_to_payload(fixture.source),
    }


def _split_to_payload(split: ReferenceSplit | None) -> dict[str, Any] | None:
    if split is None:
        return None
    return {
        "sequences": [list(item) for item in split.sequences] if split.sequences is not None else None,
        "labels": [list(item) for item in split.labels] if split.labels is not None else None,
        "examples": list(split.examples) if split.examples is not None else None,
        "metadata": dict(split.metadata),
    }


def _split_from_payload(payload: dict[str, Any]) -> ReferenceSplit:
    if payload.get("sequences") is None and payload.get("examples") is None:
        raise ValueError("Reference split must define either 'sequences' or 'examples'.")
    return ReferenceSplit(
        sequences=(
            [[int(token) for token in sequence] for sequence in payload["sequences"]]
            if payload.get("sequences") is not None
            else None
        ),
        labels=(
            [[int(token) for token in sequence] for sequence in payload["labels"]]
            if payload.get("labels") is not None
            else None
        ),
        examples=[dict(example) for example in payload["examples"]] if payload.get("examples") is not None else None,
        metadata=dict(payload.get("metadata", {})),
    )


def _source_to_payload(source: ReferenceSource | None) -> dict[str, Any] | None:
    if source is None:
        return None
    return {
        "repo": source.repo,
        "generator": source.generator,
        "commit": source.commit,
        "command": source.command,
        "notes": source.notes,
    }


def _source_from_payload(payload: dict[str, Any] | None) -> ReferenceSource | None:
    if payload is None:
        return None
    _require_fields(payload, required=("repo", "generator"))
    return ReferenceSource(
        repo=str(payload["repo"]),
        generator=str(payload["generator"]),
        commit=_optional_string(payload.get("commit")),
        command=_optional_string(payload.get("command")),
        notes=_optional_string(payload.get("notes")),
    )


def _tokenizer_spec_from_payload(payload: dict[str, Any]) -> TokenizerSpec:
    _require_fields(payload, required=("vocab_size", "pad_token_id"))
    return TokenizerSpec(
        vocab_size=int(payload["vocab_size"]),
        pad_token_id=int(payload["pad_token_id"]),
        bos_token_id=_optional_int(payload.get("bos_token_id")),
        eos_token_id=_optional_int(payload.get("eos_token_id")),
        extra_token_ids={str(key): int(value) for key, value in dict(payload.get("extra_token_ids", {})).items()},
    )


def _require_fields(payload: dict[str, Any], *, required: tuple[str, ...]) -> None:
    for field_name in required:
        if field_name not in payload:
            raise ValueError(f"Reference fixture is missing required field '{field_name}'.")


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)
