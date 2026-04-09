from __future__ import annotations

import shutil
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments


class _ToySequenceDataset(Dataset):
    def __init__(self, *, n: int, seq_len: int, mode: str) -> None:
        self.items: list[tuple[torch.Tensor, torch.Tensor]] = []
        for start in range(n):
            sequence = [(start + offset) % 16 for offset in range(seq_len)]
            if mode == "aligned":
                input_ids = sequence
                labels = sequence
            elif mode == "pre_shifted":
                input_ids = sequence[:-1]
                labels = sequence[1:]
            else:  # pragma: no cover - defensive guard for test setup
                raise ValueError(mode)
            self.items.append((torch.tensor(input_ids), torch.tensor(labels)))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        input_ids, labels = self.items[index]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids),
        }


def _collate(features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_length = max(feature["input_ids"].shape[0] for feature in features)
    input_ids = []
    labels = []
    attention_masks = []
    for feature in features:
        pad = max_length - feature["input_ids"].shape[0]
        input_ids.append(torch.cat([feature["input_ids"], torch.zeros(pad, dtype=torch.long)]))
        labels.append(torch.cat([feature["labels"], torch.full((pad,), -100, dtype=torch.long)]))
        attention_masks.append(torch.cat([feature["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_masks),
    }


def _train_eval_loss(mode: str, *, tmp_path: Path) -> float:
    torch.manual_seed(0)
    output_dir = tmp_path / mode
    shutil.rmtree(output_dir, ignore_errors=True)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=16,
            n_positions=32,
            n_ctx=32,
            n_embd=32,
            n_layer=2,
            n_head=2,
            bos_token_id=0,
            eos_token_id=0,
            pad_token_id=0,
        )
    )
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            max_steps=100,
            learning_rate=5e-3,
            logging_steps=20,
            eval_strategy="steps",
            eval_steps=20,
            save_strategy="no",
            report_to=[],
            do_train=True,
            do_eval=True,
            remove_unused_columns=False,
        ),
        train_dataset=_ToySequenceDataset(n=256, seq_len=16, mode=mode),
        eval_dataset=_ToySequenceDataset(n=64, seq_len=16, mode="aligned"),
        data_collator=_collate,
    )
    trainer.train()
    return float(trainer.evaluate()["eval_loss"])


def test_causal_lm_requires_unshifted_labels(tmp_path: Path) -> None:
    aligned_loss = _train_eval_loss("aligned", tmp_path=tmp_path)
    pre_shifted_loss = _train_eval_loss("pre_shifted", tmp_path=tmp_path)

    assert aligned_loss < 0.1
    assert pre_shifted_loss > 1.0

