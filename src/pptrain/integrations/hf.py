from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import transformers
from packaging.version import Version
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pptrain.core.base import TokenizerSpec
from pptrain.core.config import require_float32_trainable_parameters


def float32_dtype_kwargs() -> dict[str, torch.dtype]:
    """Loader kwarg that forces float32 weights.

    Without it, recent transformers (verified on 4.57.6) build models in the
    config's ``torch_dtype`` (float16 for e.g. Pythia), which the Trainer would then
    train without mixed precision. transformers 4.56 renamed ``torch_dtype`` to ``dtype``.
    """
    key = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
    return {key: torch.float32}


def apply_config_overrides(config: Any, overrides: dict[str, Any]) -> Any:
    """Set config attributes and keep per-layer lists consistent with the layer count.

    Configs such as Qwen3's carry ``layer_types`` with one entry per layer; changing
    ``num_hidden_layers`` alone produces a config that cannot be saved and reloaded.
    """
    for key, value in overrides.items():
        setattr(config, key, value)
    layer_types = getattr(config, "layer_types", None)
    num_layers = getattr(config, "num_hidden_layers", None)
    if "layer_types" not in overrides and layer_types is not None and num_layers is not None and len(layer_types) != num_layers:
        if len(set(layer_types)) != 1:
            raise ValueError("num_hidden_layers override needs an explicit layer_types override for mixed layer types")
        config.layer_types = [layer_types[0]] * num_layers
    if "max_window_layers" not in overrides and num_layers is not None and getattr(config, "max_window_layers", None) is not None:
        config.max_window_layers = min(config.max_window_layers, num_layers)
    return config


def causal_lm_from_config(config: Any, *, trust_remote_code: bool = False) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_config(
        config,
        trust_remote_code=trust_remote_code,
        **float32_dtype_kwargs(),
    )
    require_float32_trainable_parameters(model)
    return model


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
        apply_config_overrides(config, self.config.config_overrides)
        return causal_lm_from_config(config, trust_remote_code=self.config.trust_remote_code)

    def load_downstream_model(self):
        return AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=self.config.trust_remote_code,
            **float32_dtype_kwargs(),
        )

    def load_downstream_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=self.config.trust_remote_code,
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
