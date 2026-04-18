from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Callable, ClassVar, Mapping

import numpy as np

from pptrain.core.collator import CausalLMCollator
from pptrain.core.datasets import ListSequenceDataset


@dataclass(slots=True)
class TokenizerSpec:
    vocab_size: int
    pad_token_id: int = 0
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    extra_token_ids: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "extra_token_ids": dict(self.extra_token_ids),
        }


@dataclass(slots=True)
class DatasetBundle:
    train_dataset: Any
    eval_dataset: Any | None = None
    data_collator: Callable[[list[Mapping[str, Any]]], Mapping[str, Any]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SymbolicTask:
    name: str
    payload: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutedSymbolicTask:
    name: str
    payload: Any
    metadata: dict[str, Any] = field(default_factory=dict)


class Task(ABC):
    name: ClassVar[str]

    def __init__(self, config: Any) -> None:
        self.config = config

    @abstractmethod
    def tokenizer_spec(self) -> TokenizerSpec:
        raise NotImplementedError

    @abstractmethod
    def build_datasets(self, seed: int | None = None) -> DatasetBundle:
        raise NotImplementedError

    def uses_epoch_train_dataset_refresh(self) -> bool:
        return False

    def refresh_train_dataset(
        self,
        train_dataset: Any,
        *,
        seed: int | None,
        epoch_index: int,
    ) -> dict[str, Any] | None:
        return None

    def default_transfer_policy_name(self) -> str:
        return "reinit_embeddings"

    def export_config(self) -> dict[str, Any]:
        if is_dataclass(self.config):
            return asdict(self.config)
        if hasattr(self.config, "model_dump"):
            return self.config.model_dump()  # pragma: no cover
        if isinstance(self.config, Mapping):
            return dict(self.config)
        return {"value": self.config}


class TokenSequenceTask(Task):
    max_sampling_attempts: ClassVar[int] = 1

    def build_datasets(self, seed: int | None = None) -> DatasetBundle:
        rng = np.random.default_rng(seed)
        spec = self.tokenizer_spec()
        train_inputs, train_labels, train_meta = self._generate_examples(
            rng,
            spec,
            getattr(self.config, "sequence_count"),
        )
        eval_inputs, eval_labels, eval_meta = self._generate_examples(
            rng,
            spec,
            getattr(self.config, "eval_sequence_count"),
        )
        metadata = {
            "train_sequence_count": len(train_inputs),
            "eval_sequence_count": len(eval_inputs),
            "config": self.export_config(),
        }
        metadata.update(self._split_metadata("train", train_meta))
        metadata.update(self._split_metadata("eval", eval_meta))
        return DatasetBundle(
            train_dataset=ListSequenceDataset(train_inputs, labels=train_labels),
            eval_dataset=ListSequenceDataset(eval_inputs, labels=eval_labels),
            data_collator=CausalLMCollator(pad_token_id=spec.pad_token_id),
            metadata=metadata,
        )

    def _generate_examples(
        self,
        rng: np.random.Generator,
        spec: TokenizerSpec,
        count: int,
    ) -> tuple[list[list[int]], list[list[int]], list[dict[str, Any]]]:
        inputs: list[list[int]] = []
        labels: list[list[int]] = []
        metadata: list[dict[str, Any]] = []
        max_length = getattr(self.config, "max_length")
        for _ in range(count):
            tokens, item_metadata = self._sample_bounded_example(rng, spec)
            inputs.append(tokens[:-1][:max_length])
            labels.append(tokens[:-1][:max_length])
            metadata.append(item_metadata)
        return inputs, labels, metadata

    def _sample_bounded_example(
        self,
        rng: np.random.Generator,
        spec: TokenizerSpec,
    ) -> tuple[list[int], dict[str, Any]]:
        max_length = getattr(self.config, "max_length")
        for _ in range(self.max_sampling_attempts):
            tokens, metadata = self.sample_example(rng, spec)
            if len(tokens) <= max_length + 1:
                return tokens, metadata
        raise RuntimeError(f"Failed to sample a {self.name} example within max_length={max_length}.")

    @abstractmethod
    def sample_example(
        self,
        rng: np.random.Generator,
        spec: TokenizerSpec,
    ) -> tuple[list[int], dict[str, Any]]:
        raise NotImplementedError

    def _split_metadata(self, split: str, items: list[dict[str, Any]]) -> dict[str, Any]:
        return {}


class SymbolicTaskFamily(TokenSequenceTask):
    task_group_metadata_key: ClassVar[str | None] = "task"

    def sample_example(
        self,
        rng: np.random.Generator,
        spec: TokenizerSpec,
    ) -> tuple[list[int], dict[str, Any]]:
        task = self.sample_task(rng)
        executed = self.execute_task(task)
        tokens = self.serialize_task(executed, spec)
        metadata = dict(task.metadata)
        metadata.update(executed.metadata)
        if self.task_group_metadata_key is not None:
            metadata.setdefault(self.task_group_metadata_key, executed.name)
        return tokens, metadata

    @abstractmethod
    def sample_task(self, rng: np.random.Generator) -> SymbolicTask:
        raise NotImplementedError

    @abstractmethod
    def execute_task(self, task: SymbolicTask) -> ExecutedSymbolicTask:
        raise NotImplementedError

    @abstractmethod
    def serialize_task(
        self,
        executed: ExecutedSymbolicTask,
        spec: TokenizerSpec,
    ) -> list[int]:
        raise NotImplementedError

    def numeric_metadata_fields(self) -> tuple[str, ...]:
        return ()

    def _split_metadata(self, split: str, items: list[dict[str, Any]]) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        group_key = self.task_group_metadata_key
        if group_key is not None:
            values = [str(item[group_key]) for item in items if group_key in item]
            if values:
                summary[f"{split}_{group_key}_counts"] = dict(Counter(values))
        for field_name in self.numeric_metadata_fields():
            values = [float(item[field_name]) for item in items if field_name in item]
            if values:
                summary[f"{split}_avg_{field_name}"] = float(np.mean(values))
        return summary


# Backward-compatible aliases during the terminology transition.
Mechanism = Task
TokenSequenceMechanism = TokenSequenceTask
SymbolicTaskMechanism = SymbolicTaskFamily
