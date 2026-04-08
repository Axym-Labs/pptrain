from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class CausalLMCollator:
    pad_token_id: int = 0

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_length = max(feature["input_ids"].shape[0] for feature in features)
        input_ids = []
        labels = []
        attention_mask = []
        for feature in features:
            ids = feature["input_ids"]
            pad_length = max_length - ids.shape[0]
            pad = torch.full((pad_length,), self.pad_token_id, dtype=torch.long)
            mask_pad = torch.zeros((pad_length,), dtype=torch.long)
            input_ids.append(torch.cat([ids, pad], dim=0))
            labels.append(torch.cat([feature["labels"], pad], dim=0))
            attention_mask.append(torch.cat([feature["attention_mask"], mask_pad], dim=0))
        batch = {
            "input_ids": torch.stack(input_ids, dim=0),
            "labels": torch.stack(labels, dim=0),
            "attention_mask": torch.stack(attention_mask, dim=0),
        }
        batch["labels"][batch["attention_mask"] == 0] = -100
        return batch

