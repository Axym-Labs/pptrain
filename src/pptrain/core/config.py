from __future__ import annotations

import inspect
from dataclasses import dataclass, field

from transformers import TrainingArguments


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
