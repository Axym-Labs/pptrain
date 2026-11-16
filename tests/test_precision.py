from __future__ import annotations

from pathlib import Path

import pytest
import torch
from transformers import GPTNeoXConfig, Trainer

from pptrain.core.base import TokenizerSpec
from pptrain.core.config import RunConfig, require_float32_trainable_parameters
from pptrain.integrations.hf import HFCausalLMAdapter, HFModelConfig
from pptrain.replication.runner import _with_auto_precision
from pptrain.replication.training import build_random_init_downstream_model, train_downstream_stage


class _Tokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def __len__(self) -> int:
        return 64


def _float16_config_dir(tmp_path: Path) -> Path:
    """A tiny GPT-NeoX config that, like Pythia's, declares torch_dtype float16."""
    config = GPTNeoXConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=64,
        torch_dtype="float16",
    )
    model_dir = tmp_path / "fp16_config"
    config.save_pretrained(model_dir)
    assert "float16" in (model_dir / "config.json").read_text(encoding="utf-8")
    return model_dir


def _models_from_float16_config(tmp_path: Path) -> dict[str, torch.nn.Module]:
    model_config = HFModelConfig(model_name_or_path=str(_float16_config_dir(tmp_path)))
    return {
        "prepretrain": HFCausalLMAdapter(model_config).create_prepretrain_model(
            TokenizerSpec(vocab_size=48, pad_token_id=0, bos_token_id=1, eos_token_id=2)
        ),
        "downstream": build_random_init_downstream_model(
            model_config=model_config,
            tokenizer=_Tokenizer(),
            context_length=32,
        ),
    }


def test_models_from_float16_config_have_float32_parameters(tmp_path: Path) -> None:
    for name, model in _models_from_float16_config(tmp_path).items():
        dtypes = {parameter.dtype for parameter in model.parameters()}
        assert dtypes == {torch.float32}, name


def test_float16_parameters_are_rejected_before_training(tmp_path: Path) -> None:
    model = _models_from_float16_config(tmp_path)["downstream"].half()
    with pytest.raises(ValueError, match="float32 master weights"):
        require_float32_trainable_parameters(model)
    with pytest.raises(ValueError, match="float32 master weights"):
        train_downstream_stage(
            model=model,
            datasets=None,  # rejected before the datasets are touched
            run_config=RunConfig(output_dir=str(tmp_path / "unused")),
            output_dir=tmp_path / "unused",
            metadata={},
        )


class _ToyDataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ids = torch.arange(index, index + 16) % 48
        return {"input_ids": ids, "labels": ids, "attention_mask": torch.ones_like(ids)}


def _precision_cases() -> list[str]:
    cases = ["fp32"]
    if torch.cuda.is_available():
        cases.append("auto")  # replication runner: bf16 (or fp16) AMP on CUDA
    return cases


@pytest.mark.parametrize("case", _precision_cases())
def test_training_args_keep_float32_optimizer_state(tmp_path: Path, case: str) -> None:
    model = _models_from_float16_config(tmp_path)["downstream"]
    run_config = RunConfig(
        output_dir=str(tmp_path / "run"),
        per_device_train_batch_size=2,
        max_steps=1,
        learning_rate=1e-3,
        report_to=(),
    )
    if case == "auto":
        run_config = _with_auto_precision(run_config, tmp_path / "run", seed=0)
        assert run_config.bf16 or run_config.fp16
    trainer = Trainer(
        model=model,
        args=run_config.to_training_arguments(has_eval=False),
        train_dataset=_ToyDataset(),
    )
    trainer.train()

    state_dtypes = {
        value.dtype
        for state in trainer.optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value) and value.is_floating_point() and value.ndim > 0
    }
    assert state_dtypes == {torch.float32}
    assert {parameter.dtype for parameter in model.parameters()} == {torch.float32}
