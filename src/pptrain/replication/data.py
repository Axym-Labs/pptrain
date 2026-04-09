from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
import time
from typing import Any

from transformers import PreTrainedTokenizerBase

from pptrain.core.base import DatasetBundle
from pptrain.core.collator import CausalLMCollator
from pptrain.core.datasets import ListSequenceDataset
from pptrain.replication.specs import TextDatasetSpec


def _require_datasets() -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install pptrain with the 'eval' extra to use replication datasets.") from exc
    return load_dataset


def _load_dataset_with_retries(*args: Any, **kwargs: Any) -> Any:
    load_dataset = _require_datasets()
    max_attempts = 30
    delay_seconds = 5.0
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return load_dataset(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - exercised with monkeypatched errors
            if not _is_retryable_hf_error(exc) or attempt == max_attempts:
                raise
            last_error = exc
            print(
                f"Retryable dataset load failure on attempt {attempt}/{max_attempts}: {exc}. "
                f"Retrying in {delay_seconds:.0f}s.",
                flush=True,
            )
            time.sleep(delay_seconds)
            delay_seconds = min(delay_seconds * 2.0, 120.0)
    if last_error is not None:  # pragma: no cover
        raise last_error
    raise RuntimeError("Dataset load failed without raising an exception.")  # pragma: no cover


def _is_retryable_hf_error(exc: Exception) -> bool:
    status_code = getattr(getattr(exc, "response", None), "status_code", None)
    if status_code in {408, 409, 425, 429, 500, 502, 503, 504}:
        return True
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "429 too many requests",
            "503 service unavailable",
            "504 gateway timeout",
            "502 bad gateway",
            "500 internal server error",
            "temporarily unavailable",
            "connection reset",
            "connection aborted",
            "timed out",
        )
    )


@dataclass(slots=True)
class TextSequenceBundle:
    dataset_bundle: DatasetBundle
    stats: dict[str, Any]


def build_text_sequence_bundle(
    *,
    tokenizer: PreTrainedTokenizerBase,
    dataset_spec: TextDatasetSpec,
    block_size: int,
    split: str,
    target_token_count: int | None = None,
) -> TextSequenceBundle:
    sequences, labels, metadata = _build_tokenized_sequences(
        tokenizer=tokenizer,
        dataset_spec=dataset_spec,
        block_size=block_size,
        split=split,
        target_token_count=target_token_count,
    )
    bundle = DatasetBundle(
        train_dataset=ListSequenceDataset(sequences, labels=labels),
        eval_dataset=None,
        data_collator=CausalLMCollator(pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id or 0),
        metadata=metadata,
    )
    return TextSequenceBundle(dataset_bundle=bundle, stats=metadata)


def build_text_train_eval_bundle(
    *,
    tokenizer: PreTrainedTokenizerBase,
    dataset_spec: TextDatasetSpec,
    block_size: int,
    train_target_token_count: int | None = None,
    eval_target_token_count: int | None = None,
) -> DatasetBundle:
    train_sequences, train_labels, train_metadata = _build_tokenized_sequences(
        tokenizer=tokenizer,
        dataset_spec=dataset_spec,
        block_size=block_size,
        split="train",
        target_token_count=train_target_token_count,
    )
    eval_sequences, eval_labels, eval_metadata = _build_tokenized_sequences(
        tokenizer=tokenizer,
        dataset_spec=dataset_spec,
        block_size=block_size,
        split="eval",
        target_token_count=eval_target_token_count,
    )
    return DatasetBundle(
        train_dataset=ListSequenceDataset(train_sequences, labels=train_labels),
        eval_dataset=ListSequenceDataset(eval_sequences, labels=eval_labels),
        data_collator=CausalLMCollator(pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id or 0),
        metadata={
            "train_sequence_count": len(train_sequences),
            "eval_sequence_count": len(eval_sequences),
            "train_avg_chunk_length": train_metadata["avg_chunk_length"],
            "eval_avg_chunk_length": eval_metadata["avg_chunk_length"],
        },
    )


def _build_tokenized_sequences(
    *,
    tokenizer: PreTrainedTokenizerBase,
    dataset_spec: TextDatasetSpec,
    block_size: int,
    split: str,
    target_token_count: int | None,
) -> tuple[list[list[int]], list[list[int]], dict[str, Any]]:
    if dataset_spec.source == "hf" and dataset_spec.streaming:
        return _tokenize_streaming_hf_texts(
            tokenizer=tokenizer,
            dataset_spec=dataset_spec,
            split=split,
            block_size=block_size,
            target_token_count=target_token_count,
        )
    texts = _load_texts(dataset_spec, split)
    return _tokenize_texts(tokenizer=tokenizer, texts=texts, block_size=block_size)


def _load_texts(dataset_spec: TextDatasetSpec, split: str) -> list[str]:
    if dataset_spec.source == "inline":
        return _load_inline_texts(dataset_spec, split)
    if dataset_spec.source == "hf":
        return _load_hf_texts(dataset_spec, split)
    raise KeyError(f"Unsupported text dataset source '{dataset_spec.source}'.")


def _load_inline_texts(dataset_spec: TextDatasetSpec, split: str) -> list[str]:
    if split == "warmup":
        return list(dataset_spec.inline_warmup_texts)
    if split == "train":
        return list(dataset_spec.inline_train_texts)
    if split == "eval":
        return list(dataset_spec.inline_eval_texts)
    raise KeyError(f"Unsupported inline split '{split}'.")


def _load_hf_texts(dataset_spec: TextDatasetSpec, split: str) -> list[str]:
    if dataset_spec.streaming:
        raise ValueError("Streaming datasets must be consumed through the token-budgeted loader.")
    split_name = {
        "warmup": dataset_spec.warmup_split,
        "train": dataset_spec.train_split,
        "eval": dataset_spec.eval_split,
    }[split]
    if split_name is None:
        return []
    if dataset_spec.dataset_name is None:
        raise ValueError("dataset_name must be provided for hf datasets.")
    args: list[str] = [dataset_spec.dataset_name]
    if dataset_spec.dataset_config_name is not None:
        args.append(dataset_spec.dataset_config_name)
    if dataset_spec.subset is not None:
        args.append(dataset_spec.subset)
    dataset = _load_dataset_with_retries(*args, split=split_name)
    return [_format_record(dataset_spec, record) for record in dataset]


def _tokenize_streaming_hf_texts(
    *,
    tokenizer: PreTrainedTokenizerBase,
    dataset_spec: TextDatasetSpec,
    split: str,
    block_size: int,
    target_token_count: int | None,
) -> tuple[list[list[int]], list[list[int]], dict[str, Any]]:
    if target_token_count is None:
        raise ValueError("target_token_count must be provided for streaming datasets.")
    split_name = {
        "warmup": dataset_spec.warmup_split,
        "train": dataset_spec.train_split,
        "eval": dataset_spec.eval_split,
    }[split]
    if split_name is None:
        return [], [], {"num_texts": 0, "avg_chunk_length": 0.0, "streaming": True, "raw_token_count": 0}
    if dataset_spec.dataset_name is None:
        raise ValueError("dataset_name must be provided for hf datasets.")
    args: list[str] = [dataset_spec.dataset_name]
    if dataset_spec.dataset_config_name is not None:
        args.append(dataset_spec.dataset_config_name)
    if dataset_spec.subset is not None:
        args.append(dataset_spec.subset)
    dataset = _load_dataset_with_retries(*args, split=split_name, streaming=True)
    if dataset_spec.shuffle_buffer_size is not None:
        dataset = dataset.shuffle(seed=dataset_spec.shuffle_seed, buffer_size=dataset_spec.shuffle_buffer_size)
    skip_records = {
        "warmup": dataset_spec.warmup_skip_records,
        "train": dataset_spec.train_skip_records,
        "eval": dataset_spec.eval_skip_records,
    }[split]
    if skip_records > 0:
        dataset = islice(dataset, skip_records, None)

    eos_token_id = tokenizer.eos_token_id or tokenizer.pad_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer must provide either eos_token_id or pad_token_id.")

    tokens: list[int] = []
    num_texts = 0
    raw_token_count = 0
    for record in dataset:
        text = _format_record(dataset_spec, record)
        encoded = tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"]
        if not encoded:
            continue
        tokens.extend(encoded)
        tokens.append(eos_token_id)
        num_texts += 1
        raw_token_count += len(encoded) + 1
        if raw_token_count >= target_token_count:
            break

    sequences, labels, metadata = _chunk_token_buffer(
        tokens=tokens,
        block_size=block_size,
        num_texts=num_texts,
    )
    metadata["streaming"] = True
    metadata["raw_token_count"] = raw_token_count
    metadata["target_token_count"] = target_token_count
    metadata["skip_records"] = skip_records
    return sequences, labels, metadata


def _format_record(dataset_spec: TextDatasetSpec, record: dict[str, Any]) -> str:
    if dataset_spec.formatter == "plain_text":
        return str(record[dataset_spec.text_field])
    if dataset_spec.formatter == "gsm8k_qa":
        return f"Question: {record['question']}\nAnswer: {record['answer']}"
    if dataset_spec.formatter == "cnn_dm_tldr":
        return f"Article:\n{record['article']}\n\nTL;DR:\n{record['highlights']}"
    raise KeyError(f"Unsupported dataset formatter '{dataset_spec.formatter}'.")


def _tokenize_texts(
    *,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    block_size: int,
) -> tuple[list[list[int]], list[list[int]], dict[str, Any]]:
    eos_token_id = tokenizer.eos_token_id or tokenizer.pad_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer must provide either eos_token_id or pad_token_id.")

    tokens: list[int] = []
    for text in texts:
        encoded = tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"]
        if not encoded:
            continue
        tokens.extend(encoded)
        tokens.append(eos_token_id)

    sequences, labels, metadata = _chunk_token_buffer(
        tokens=tokens,
        block_size=block_size,
        num_texts=len(texts),
    )
    metadata["streaming"] = False
    metadata["raw_token_count"] = len(tokens)
    return sequences, labels, metadata


def _chunk_token_buffer(
    *,
    tokens: list[int],
    block_size: int,
    num_texts: int,
) -> tuple[list[list[int]], list[list[int]], dict[str, Any]]:
    chunk_size = block_size + 1
    if len(tokens) < chunk_size:
        while len(tokens) < chunk_size and tokens:
            tokens.extend(tokens[: min(len(tokens), chunk_size - len(tokens))])

    sequences: list[list[int]] = []
    labels: list[list[int]] = []
    for start in range(0, max(len(tokens) - chunk_size + 1, 0), chunk_size):
        chunk = tokens[start : start + chunk_size]
        if len(chunk) < chunk_size:
            continue
        sequences.append(chunk[:-1])
        labels.append(chunk[1:])

    avg_chunk_length = (sum(len(item) for item in sequences) / len(sequences)) if sequences else 0.0
    return sequences, labels, {"num_texts": num_texts, "avg_chunk_length": avg_chunk_length}
