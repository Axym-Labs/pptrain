from __future__ import annotations

import inspect
from dataclasses import dataclass, field

import torch
from torch import nn
from transformers import TrainingArguments


def require_float32_trainable_parameters(model: nn.Module) -> None:
    """Fail if any trainable parameter is not float32.

    The HF Trainer keeps AdamW state in the parameter dtype and only applies loss
    scaling or autocast on top of float32 master weights (``fp16``/``bf16``
    TrainingArguments). Low-precision parameters would therefore train with
    low-precision optimizer state (AdamW ``eps=1e-8`` underflows to 0 in float16).
    """
    if not isinstance(model, nn.Module):
        return
    low_precision = sorted(
        {str(parameter.dtype) for parameter in model.parameters() if parameter.requires_grad}
        - {str(torch.float32)}
    )
    if low_precision:
        raise ValueError(
            "pptrain trains float32 master weights; got trainable parameters in "
            f"{', '.join(low_precision)}. Build the model with dtype=torch.float32 and use "
            "RunConfig(bf16=True) or RunConfig(fp16=True) for mixed precision."
        )


@dataclass(slots=True)
class RunConfig:
    output_dir: str
    seed: int = 42
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-4
    weight_decay: float = 0.0
    num_train_epochs: float = 1.0
    max_steps: int = -1
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    warmup_steps: int = 0
    save_total_limit: int = 2
    dataloader_num_workers: int = 0
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = False
    remove_checkpoints: bool = True
    report_to: tuple[str, ...] = field(default_factory=tuple)

    def to_training_arguments(self, has_eval: bool) -> TrainingArguments:
        evaluation_strategy = "steps" if has_eval else "no"
        kwargs = dict(
            output_dir=self.output_dir,
            seed=self.seed,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            num_train_epochs=self.num_train_epochs,
            max_steps=self.max_steps,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            eval_steps=self.eval_steps,
            warmup_steps=self.warmup_steps,
            save_total_limit=self.save_total_limit,
            dataloader_num_workers=self.dataloader_num_workers,
            fp16=self.fp16,
            bf16=self.bf16,
            gradient_checkpointing=self.gradient_checkpointing,
            report_to=list(self.report_to),
            remove_unused_columns=False,
        )
        signature = inspect.signature(TrainingArguments.__init__)
        if "evaluation_strategy" in signature.parameters:
            kwargs["evaluation_strategy"] = evaluation_strategy
        else:
            kwargs["eval_strategy"] = evaluation_strategy
        return TrainingArguments(**kwargs)
