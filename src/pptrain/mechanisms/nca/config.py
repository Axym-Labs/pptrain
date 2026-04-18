from __future__ import annotations

from dataclasses import dataclass

from pptrain.core.presets import TaskPreset, sequence_preset


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
    regenerate_train_each_epoch: bool = True


NCA_PRESETS: tuple[TaskPreset, ...] = (
    sequence_preset(
        "smoke",
        "Tiny local NCA smoke run for API and transfer validation.",
        sequence_count=64,
        eval_sequence_count=16,
        reference="pptrain",
        grid_size=8,
        num_states=6,
        patch_size=2,
        max_length=256,
        rule_count=16,
        eval_rule_count=8,
        complexity_min=0.3,
        complexity_max=1.0,
        regenerate_train_each_epoch=True,
    ),
    sequence_preset(
        "paper_web_text",
        "Paper-scale web-text NCA preset with the 50%+ gzip complexity band.",
        sequence_count=8_000_000,
        eval_sequence_count=200_000,
        reference="Lee et al. 2026",
        grid_size=12,
        num_states=10,
        patch_size=2,
        temperature=1e-4,
        init_rollout_steps=10,
        max_length=1024,
        rule_count=16_000,
        eval_rule_count=2_000,
        complexity_min=0.5,
        complexity_max=1.0,
        regenerate_train_each_epoch=True,
    ),
    sequence_preset(
        "paper_code",
        "Paper-scale code-domain NCA preset with the 30-40% gzip complexity band.",
        sequence_count=8_000_000,
        eval_sequence_count=200_000,
        reference="Lee et al. 2026",
        grid_size=12,
        num_states=10,
        patch_size=2,
        temperature=1e-4,
        init_rollout_steps=10,
        max_length=1024,
        rule_count=16_000,
        eval_rule_count=2_000,
        complexity_min=0.3,
        complexity_max=0.4,
        regenerate_train_each_epoch=True,
    ),
)
