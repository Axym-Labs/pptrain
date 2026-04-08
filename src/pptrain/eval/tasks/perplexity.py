from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from pptrain.eval.base import EvalResult, EvalTask
from pptrain.eval.generation import resolve_device


def _require_datasets() -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install pptrain with the 'eval' extra to use dataset-backed eval tasks.") from exc
    return load_dataset


@dataclass(slots=True)
class PerplexityTask(EvalTask):
    dataset_name: str
    split: str = "validation[:128]"
    text_field: str = "text"
    max_length: int = 256
    name: str = "perplexity"

    def run(
        self,
        *,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        **_: Any,
    ) -> EvalResult:
        load_dataset = _require_datasets()
        dataset = load_dataset(self.dataset_name, split=self.split)
        losses = []
        device = resolve_device(model)
        model.eval()
        for record in dataset:
            text = record[self.text_field]
            if not text or not text.strip():
                continue
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            ).to(device)
            with torch.no_grad():
                loss = model(**encoded, labels=encoded["input_ids"]).loss
            losses.append(loss.item())
        mean_loss = sum(losses) / max(len(losses), 1)
        return EvalResult(
            name=self.name,
            metrics={
                "loss": mean_loss,
                "perplexity": math.exp(mean_loss),
            },
            artifacts={"num_examples": len(losses)},
        )

