from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class NCAConfig:
    grid_size: int = 12
    num_states: int = 10
    patch_size: int = 2
    perception_dim: int = 4
    hidden_dim: int = 16
    identity_bias: float = 0.0
    temperature: float = 1e-4
    rollout_stride: int = 1
    init_rollout_steps: int = 10
    min_frames: int = 1
    complexity_probe_frames: int = 10
    sequence_count: int = 512
    eval_sequence_count: int = 64
    rule_count: int | None = None
    eval_rule_count: int | None = None
    max_length: int = 1024
    complexity_min: float = 0.5
    complexity_max: float = 1.0
    max_rule_attempts_per_sequence: int = 12
