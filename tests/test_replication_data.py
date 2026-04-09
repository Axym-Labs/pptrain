from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from pptrain.replication import data
from pptrain.replication.specs import TextDatasetSpec


class DummyTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = False, **kwargs) -> dict[str, list[int]]:
        del add_special_tokens
        del kwargs
        tokens = [index + 1 for index, _part in enumerate(text.split())]
        return {"input_ids": tokens}


@dataclass
class FakeStreamingDataset:
    records: list[dict[str, str]]

    def shuffle(self, seed: int, buffer_size: int):  # noqa: ARG002
        return self

    def __iter__(self):
        return iter(self.records)


def test_streaming_loader_stops_at_token_budget_and_respects_skip(monkeypatch) -> None:
    records = [{"text": "one two three"} for _ in range(6)]
    monkeypatch.setattr(
        data,
        "_require_datasets",
        lambda: (lambda *args, **kwargs: FakeStreamingDataset(records)),
    )
    spec = TextDatasetSpec(
        source="hf",
        formatter="plain_text",
        dataset_name="dummy",
        train_split="train",
        eval_split="train",
        streaming=True,
        shuffle_buffer_size=8,
        train_skip_records=2,
    )
    bundle = data.build_text_train_eval_bundle(
        tokenizer=DummyTokenizer(),
        dataset_spec=spec,
        block_size=4,
        train_target_token_count=8,
        eval_target_token_count=8,
    )
    assert len(bundle.train_dataset) >= 1
    assert bundle.metadata["train_sequence_count"] >= 1
    assert bundle.metadata["eval_sequence_count"] >= 1

    train_sequences, train_labels, metadata = data._build_tokenized_sequences(
        tokenizer=DummyTokenizer(),
        dataset_spec=spec,
        block_size=4,
        split="train",
        target_token_count=8,
    )
    assert len(train_sequences) == len(train_labels) >= 1
    assert metadata["streaming"] is True
    assert metadata["skip_records"] == 2
    assert metadata["target_token_count"] == 8
    assert metadata["raw_token_count"] >= 8


def test_streaming_loader_retries_retryable_dataset_errors(monkeypatch) -> None:
    records = [{"text": "one two three"} for _ in range(6)]
    attempts = {"count": 0}

    class FakeRateLimitError(Exception):
        def __init__(self) -> None:
            self.response = SimpleNamespace(status_code=429)
            super().__init__("429 Too Many Requests")

    def fake_load_dataset(*args, **kwargs):
        del args, kwargs
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise FakeRateLimitError()
        return FakeStreamingDataset(records)

    sleeps: list[float] = []
    monkeypatch.setattr(data, "_require_datasets", lambda: fake_load_dataset)
    monkeypatch.setattr(data.time, "sleep", sleeps.append)
    spec = TextDatasetSpec(
        source="hf",
        formatter="plain_text",
        dataset_name="dummy",
        train_split="train",
        streaming=True,
    )

    sequences, labels, metadata = data._build_tokenized_sequences(
        tokenizer=DummyTokenizer(),
        dataset_spec=spec,
        block_size=4,
        split="train",
        target_token_count=8,
    )

    assert attempts["count"] == 3
    assert sleeps == [5.0, 10.0]
    assert len(sequences) == len(labels) >= 1
    assert metadata["streaming"] is True
