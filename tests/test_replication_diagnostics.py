import numpy as np
import torch
from torch import nn

from pptrain.replication.diagnostics import _effective_rank, _forward_diagnostic_batch


class TinyBlock(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + 1.0


class TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, 4)
        self.layers = nn.ModuleList(TinyBlock() for _ in range(3))
        self.last_use_cache = None

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        use_cache: bool = True,
        return_dict: bool = True,
    ):
        del attention_mask, output_hidden_states
        self.last_use_cache = use_cache
        hidden = self.embedding(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        if return_dict:
            return type("TinyOutput", (), {"last_hidden_state": hidden})()
        return (hidden,)


class TinyCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gpt_neox = TinyBackbone()
        self.embed_out = nn.Linear(4, 7)

    def forward(self, **kwargs):  # pragma: no cover - the test must avoid this path
        del kwargs
        raise AssertionError("diagnostics should not materialize full-sequence logits")


def test_effective_rank_handles_non_finite_inputs() -> None:
    hidden = np.array(
        [
            [1.0, np.nan, 2.0],
            [np.inf, 3.0, -np.inf],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float32,
    )

    rank = _effective_rank(hidden)

    assert np.isfinite(rank)
    assert rank >= 0.0


def test_diagnostic_forward_uses_selected_logits_fast_path() -> None:
    model = TinyCausalLM()
    batch = {
        "input_ids": torch.arange(10).view(2, 5),
        "attention_mask": torch.ones(2, 5, dtype=torch.long),
        "positions": torch.tensor([[0, 1], [1, 3]], dtype=torch.long),
    }

    logits, hidden = _forward_diagnostic_batch(model=model, batch=batch)

    assert logits.shape == (2, 7)
    assert hidden.shape == (2, 4)
    assert model.gpt_neox.last_use_cache is False
