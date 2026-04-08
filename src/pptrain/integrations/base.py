from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import torch.nn as nn

from pptrain.core.base import TokenizerSpec


class CausalLMAdapter(Protocol):
    config: Any

    def create_prepretrain_model(self, tokenizer_spec: TokenizerSpec) -> nn.Module:
        raise NotImplementedError

    def load_downstream_model(self) -> nn.Module:
        raise NotImplementedError

    def load_downstream_tokenizer(self) -> Any | None:
        raise NotImplementedError


@dataclass(slots=True)
class CallableAdapterConfig:
    name: str = "callable"

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name}


class CallableCausalLMAdapter:
    def __init__(
        self,
        *,
        create_prepretrain_model: Callable[[TokenizerSpec], nn.Module],
        load_downstream_model: Callable[[], nn.Module],
        load_downstream_tokenizer: Callable[[], Any] | None = None,
        name: str = "callable",
    ) -> None:
        self._create_prepretrain_model = create_prepretrain_model
        self._load_downstream_model = load_downstream_model
        self._load_downstream_tokenizer = load_downstream_tokenizer
        self.config = CallableAdapterConfig(name=name)

    def create_prepretrain_model(self, tokenizer_spec: TokenizerSpec) -> nn.Module:
        return self._create_prepretrain_model(tokenizer_spec)

    def load_downstream_model(self) -> nn.Module:
        return self._load_downstream_model()

    def load_downstream_tokenizer(self) -> Any | None:
        if self._load_downstream_tokenizer is None:
            return None
        return self._load_downstream_tokenizer()


class VocabSizeCausalLMAdapter(CallableCausalLMAdapter):
    def __init__(
        self,
        *,
        create_prepretrain_model: Callable[[int], nn.Module],
        load_downstream_model: Callable[[], nn.Module],
        load_downstream_tokenizer: Callable[[], Any] | None = None,
        name: str = "vocab-size",
    ) -> None:
        super().__init__(
            create_prepretrain_model=lambda tokenizer_spec: create_prepretrain_model(tokenizer_spec.vocab_size),
            load_downstream_model=load_downstream_model,
            load_downstream_tokenizer=load_downstream_tokenizer,
            name=name,
        )
