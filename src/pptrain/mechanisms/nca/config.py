from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class NCAConfig:
    grid_size: int = 12
    rollout_steps: int = 24
    num_states: int = 10
    patch_size: int = 2
    hidden_dim: int = 32
    stochasticity: float = 0.01
    sequence_count: int = 512
    eval_sequence_count: int = 64
    max_length: int = 1024
    complexity_min: float = 0.5
    complexity_max: float = 1.0
    max_rule_attempts_per_sequence: int = 12

