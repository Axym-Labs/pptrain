from __future__ import annotations

import pytest
import torch
from torch import nn

from pptrain.core.registry import create_task
from pptrain.replication import diagnostics

SYNTHETIC_TASKS = ("nca", "lime", "simpler_tasks", "procedural", "dyck", "summarization")


@pytest.mark.parametrize("task_name", SYNTHETIC_TASKS)
def test_synthetic_task_labels_are_unshifted(task_name: str) -> None:
    """HF causal-LM losses shift labels internally, so every task must emit labels
    aligned with input_ids (either the same token or -100), never pre-shifted ones."""
    task = create_task(task_name, {"preset": "smoke"})
    bundle = task.build_datasets(seed=3)
    checked_positions = 0
    for dataset in (bundle.train_dataset, bundle.eval_dataset):
        for index in range(min(len(dataset), 8)):
            sample = dataset[index]
            input_ids = sample["input_ids"]
            labels = sample["labels"]
            assert input_ids.shape == labels.shape
            supervised = labels != -100
            assert bool((labels[supervised] == input_ids[supervised]).all()), (
                f"{task_name} labels are shifted relative to input_ids"
            )
            checked_positions += int(supervised.sum())
    assert checked_positions > 0


class _NextTokenOracle(nn.Module):
    """Predicts input_ids[i + 1] at position i, i.e. a perfect next-token model."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        del attention_mask
        next_ids = torch.roll(input_ids, shifts=-1, dims=1)
        logits = nn.functional.one_hot(next_ids, self.vocab_size).float()
        return type("Output", (), {"logits": logits})()


def test_nca_synthetic_token_accuracy_scores_next_token_predictions(monkeypatch) -> None:
    task = create_task("nca", {"preset": "smoke"})
    vocab_size = task.tokenizer_spec().vocab_size
    monkeypatch.setattr(
        diagnostics.AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: _NextTokenOracle(vocab_size),
    )
    monkeypatch.setattr(diagnostics, "_resolve_device", lambda: torch.device("cpu"))

    accuracy = diagnostics.compute_nca_synthetic_token_accuracy(
        task=task, model_dir="unused", seed=3, max_batches=2
    )

    assert accuracy == pytest.approx(1.0)
