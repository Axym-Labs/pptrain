from __future__ import annotations

from dataclasses import dataclass
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
) -> TextSequenceBundle:
    texts = _load_texts(dataset_spec, split)
    sequences, labels, metadata = _tokenize_texts(tokenizer=tokenizer, texts=texts, block_size=block_size)
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
) -> DatasetBundle:
    train_texts = _load_texts(dataset_spec, "train")
    eval_texts = _load_texts(dataset_spec, "eval")
    train_sequences, train_labels, train_metadata = _tokenize_texts(
        tokenizer=tokenizer,
        texts=train_texts,
        block_size=block_size,
    )
    eval_sequences, eval_labels, eval_metadata = _tokenize_texts(
        tokenizer=tokenizer,
        texts=eval_texts,
        block_size=block_size,
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
    load_dataset = _require_datasets()
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
    dataset = load_dataset(*args, split=split_name)
    return [_format_record(dataset_spec, record) for record in dataset]


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
        encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not encoded:
            continue
        tokens.extend(encoded)
        tokens.append(eos_token_id)

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
    return sequences, labels, {"num_texts": len(texts), "avg_chunk_length": avg_chunk_length}
