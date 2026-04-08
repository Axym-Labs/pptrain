from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pptrain.core.base import TokenizerSpec


@dataclass(slots=True)
class HFModelConfig:
    model_name_or_path: str
    trust_remote_code: bool = False
    config_overrides: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name_or_path": self.model_name_or_path,
            "trust_remote_code": self.trust_remote_code,
            "config_overrides": dict(self.config_overrides),
        }


class HFCausalLMAdapter:
    def __init__(self, config: HFModelConfig) -> None:
        self.config = config

    def create_prepretrain_model(self, tokenizer_spec: TokenizerSpec):
        config = AutoConfig.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=self.config.trust_remote_code,
        )
        config.vocab_size = tokenizer_spec.vocab_size
        config.pad_token_id = tokenizer_spec.pad_token_id
        if tokenizer_spec.bos_token_id is not None:
            config.bos_token_id = tokenizer_spec.bos_token_id
        if tokenizer_spec.eos_token_id is not None:
            config.eos_token_id = tokenizer_spec.eos_token_id
        for key, value in self.config.config_overrides.items():
            setattr(config, key, value)
        return AutoModelForCausalLM.from_config(config, trust_remote_code=self.config.trust_remote_code)

    def load_downstream_model(self):
        return AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=self.config.trust_remote_code,
        )

    def load_downstream_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=self.config.trust_remote_code,
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
