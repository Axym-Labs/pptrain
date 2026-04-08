from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer

from pptrain.core.base import DatasetBundle
from pptrain.core.config import RunConfig
from pptrain.core.plotting import save_training_summary_plot
from pptrain.core.transfer import ReinitializeEmbeddingTransferPolicy, TransferBundle, TransferReport
from pptrain.integrations.hf import HFModelConfig


@dataclass(slots=True)
class DownstreamStageResult:
    run_dir: Path
    model_dir: Path
    metrics: dict[str, Any]
    plot_path: Path | None
    log_history: list[dict[str, Any]]


def load_tokenizer(model_config: HFModelConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_random_init_downstream_model(
    *,
    model_config: HFModelConfig,
    tokenizer,
    context_length: int,
) -> torch.nn.Module:
    config = AutoConfig.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
    )
    config.vocab_size = len(tokenizer)
    config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.bos_token_id is not None:
        config.bos_token_id = tokenizer.bos_token_id
    if tokenizer.eos_token_id is not None:
        config.eos_token_id = tokenizer.eos_token_id
    _apply_context_length(config, context_length)
    for key, value in model_config.config_overrides.items():
        setattr(config, key, value)
    return AutoModelForCausalLM.from_config(config, trust_remote_code=model_config.trust_remote_code)


def train_downstream_stage(
    *,
    model: torch.nn.Module,
    datasets: DatasetBundle,
    run_config: RunConfig,
    output_dir: Path,
    metadata: dict[str, Any],
) -> DownstreamStageResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_run_config = RunConfig(**{**asdict(run_config), "output_dir": str(output_dir)})
    trainer = Trainer(
        model=model,
        args=stage_run_config.to_training_arguments(has_eval=datasets.eval_dataset is not None),
        train_dataset=datasets.train_dataset,
        eval_dataset=datasets.eval_dataset,
        data_collator=datasets.data_collator,
    )

    train_result = trainer.train()
    metrics = dict(train_result.metrics)
    if datasets.eval_dataset is not None:
        metrics.update(trainer.evaluate())
    model_dir = output_dir / "model"
    trainer.save_model(str(model_dir))
    trainer.save_state()
    plot_path = save_training_summary_plot(
        log_history=trainer.state.log_history,
        metrics=metrics,
        dataset_metadata=datasets.metadata,
        output_path=output_dir / "training_summary.png",
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run": asdict(stage_run_config),
                "dataset_metadata": datasets.metadata,
                "metrics": metrics,
                "extra_metadata": metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return DownstreamStageResult(
        run_dir=output_dir,
        model_dir=model_dir,
        metrics=metrics,
        plot_path=plot_path,
        log_history=list(trainer.state.log_history),
    )


def apply_transfer_bundle(
    *,
    bundle: TransferBundle,
    target_model: torch.nn.Module,
) -> TransferReport:
    return ReinitializeEmbeddingTransferPolicy().apply_bundle(bundle, target_model)


def _apply_context_length(config: Any, context_length: int) -> None:
    for attribute in ("max_position_embeddings", "n_positions", "n_ctx"):
        if hasattr(config, attribute):
            setattr(config, attribute, context_length)
