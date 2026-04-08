from __future__ import annotations

from abc import ABC, abstractmethod
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


class Mechanism(ABC):
    name: ClassVar[str]

    def __init__(self, config: Any) -> None:
        self.config = config

    @abstractmethod
    def tokenizer_spec(self) -> TokenizerSpec:
        raise NotImplementedError

    @abstractmethod
    def build_datasets(self, seed: int | None = None) -> DatasetBundle:
        raise NotImplementedError

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


class TokenSequenceMechanism(Mechanism):
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
            tokens, item_metadata = self.sample_tokens(rng, spec)
            inputs.append(tokens[:-1][:max_length])
            labels.append(tokens[1:][:max_length])
            metadata.append(item_metadata)
        return inputs, labels, metadata

    @abstractmethod
    def sample_tokens(
        self,
        rng: np.random.Generator,
        spec: TokenizerSpec,
    ) -> tuple[list[int], dict[str, Any]]:
        raise NotImplementedError

    def _split_metadata(self, split: str, items: list[dict[str, Any]]) -> dict[str, Any]:
        return {}
