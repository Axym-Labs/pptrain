from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Callable, ClassVar, Mapping


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

