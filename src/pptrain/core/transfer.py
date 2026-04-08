from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file as load_safetensors
from transformers import PreTrainedModel


@dataclass(slots=True)
class TransferReport:
    loaded_parameter_count: int
    skipped_parameters: list[str]
    missing_parameters: list[str]


@dataclass(slots=True)
class TransferBundle:
    run_dir: Path
    model_dir: Path
    tokenizer_spec: dict[str, Any]
    mechanism_name: str
    mechanism_config: dict[str, Any]
    transfer_policy_name: str

    def save(self) -> Path:
        path = self.run_dir / "transfer_bundle.json"
        payload = {
            "run_dir": str(self.run_dir),
            "model_dir": str(self.model_dir),
            "tokenizer_spec": self.tokenizer_spec,
            "mechanism_name": self.mechanism_name,
            "mechanism_config": self.mechanism_config,
            "transfer_policy_name": self.transfer_policy_name,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, run_dir: str | Path) -> "TransferBundle":
        path = Path(run_dir) / "transfer_bundle.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            run_dir=Path(payload["run_dir"]),
            model_dir=Path(payload["model_dir"]),
            tokenizer_spec=payload["tokenizer_spec"],
            mechanism_name=payload["mechanism_name"],
            mechanism_config=payload["mechanism_config"],
            transfer_policy_name=payload["transfer_policy_name"],
        )


class ReinitializeEmbeddingTransferPolicy:
    name = "reinit_embeddings"

    def apply_bundle(self, bundle: TransferBundle, target_model: PreTrainedModel) -> TransferReport:
        state_dict = self._load_state_dict(bundle.model_dir)
        return self.apply_state_dict(state_dict, target_model)

    def apply_state_dict(
        self,
        source_state_dict: dict[str, torch.Tensor],
        target_model: torch.nn.Module,
    ) -> TransferReport:
        target_state = target_model.state_dict()
        skip_names = self._embedding_parameter_names(target_model)
        compatible: dict[str, torch.Tensor] = {}
        skipped: list[str] = []
        for name, tensor in source_state_dict.items():
            if name in skip_names:
                skipped.append(name)
                continue
            if name not in target_state or target_state[name].shape != tensor.shape:
                skipped.append(name)
                continue
            compatible[name] = tensor
        load_result = target_model.load_state_dict(compatible, strict=False)
        return TransferReport(
            loaded_parameter_count=len(compatible),
            skipped_parameters=sorted(skipped),
            missing_parameters=sorted(load_result.missing_keys),
        )

    @staticmethod
    def _load_state_dict(model_dir: Path) -> dict[str, torch.Tensor]:
        safe_path = model_dir / "model.safetensors"
        bin_path = model_dir / "pytorch_model.bin"
        if safe_path.exists():
            return load_safetensors(str(safe_path))
        if bin_path.exists():
            return torch.load(bin_path, map_location="cpu")
        raise FileNotFoundError(f"No model weights found in {model_dir}")

    @staticmethod
    def _embedding_parameter_names(model: torch.nn.Module) -> set[str]:
        if not hasattr(model, "get_input_embeddings") or not hasattr(model, "get_output_embeddings"):
            return set()
        skip_ids: set[int] = set()
        input_embeddings = model.get_input_embeddings()
        if input_embeddings is not None:
            skip_ids.add(id(input_embeddings.weight))
        output_embeddings = model.get_output_embeddings()
        if output_embeddings is not None and hasattr(output_embeddings, "weight"):
            skip_ids.add(id(output_embeddings.weight))

        names: set[str] = set()
        for name, parameter in model.named_parameters():
            if id(parameter) in skip_ids:
                names.add(name)
        return names


class SkipParametersTransferPolicy:
    def __init__(
        self,
        *,
        skip_parameter_names: tuple[str, ...] = (),
        skip_parameter_prefixes: tuple[str, ...] = (),
    ) -> None:
        self.skip_parameter_names = tuple(skip_parameter_names)
        self.skip_parameter_prefixes = tuple(skip_parameter_prefixes)

    def apply_state_dict(
        self,
        source_state_dict: dict[str, torch.Tensor],
        target_model: torch.nn.Module,
    ) -> TransferReport:
        target_state = target_model.state_dict()
        compatible: dict[str, torch.Tensor] = {}
        skipped: list[str] = []
        for name, tensor in source_state_dict.items():
            if name in self.skip_parameter_names or name.startswith(self.skip_parameter_prefixes):
                skipped.append(name)
                continue
            if name not in target_state or target_state[name].shape != tensor.shape:
                skipped.append(name)
                continue
            compatible[name] = tensor
        load_result = target_model.load_state_dict(compatible, strict=False)
        return TransferReport(
            loaded_parameter_count=len(compatible),
            skipped_parameters=sorted(skipped),
            missing_parameters=sorted(load_result.missing_keys),
        )

    def apply_bundle(self, bundle: TransferBundle, target_model: torch.nn.Module) -> TransferReport:
        state_dict = ReinitializeEmbeddingTransferPolicy._load_state_dict(bundle.model_dir)
        return self.apply_state_dict(state_dict, target_model)
