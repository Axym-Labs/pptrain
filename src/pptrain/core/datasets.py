from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass(slots=True)
class ListSequenceDataset(Dataset):
    sequences: list[list[int]]
    labels: list[list[int]] | None = None

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sequence = torch.tensor(self.sequences[index], dtype=torch.long)
        labels = self.labels[index] if self.labels is not None else self.sequences[index]
        label_tensor = torch.tensor(labels, dtype=torch.long)
        attention_mask = torch.ones_like(sequence)
        return {
            "input_ids": sequence,
            "labels": label_tensor,
            "attention_mask": attention_mask,
        }
