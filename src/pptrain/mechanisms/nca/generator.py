from __future__ import annotations

import gzip
import math
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class NCARule:
    conv3x3_weight: np.ndarray
    conv3x3_bias: np.ndarray
    conv1x1_hidden_weight: np.ndarray
    conv1x1_hidden_bias: np.ndarray
    conv1x1_out_weight: np.ndarray
    conv1x1_out_bias: np.ndarray


def sample_rule(
    rng: np.random.Generator,
    num_states: int,
    hidden_dim: int,
    perception_dim: int,
) -> NCARule:
    return NCARule(
        conv3x3_weight=_lecun_normal(rng, (perception_dim, num_states, 3, 3), fan_in=9 * num_states),
        conv3x3_bias=np.zeros((perception_dim,), dtype=np.float32),
        conv1x1_hidden_weight=_lecun_normal(rng, (hidden_dim, perception_dim, 1, 1), fan_in=perception_dim),
        conv1x1_hidden_bias=np.zeros((hidden_dim,), dtype=np.float32),
        conv1x1_out_weight=_lecun_normal(rng, (num_states, hidden_dim, 1, 1), fan_in=hidden_dim),
        conv1x1_out_bias=np.zeros((num_states,), dtype=np.float32),
    )


def generate_rule_pool(
    rng: np.random.Generator,
    *,
    rule_count: int,
    sequence_count: int,
    num_states: int,
    hidden_dim: int,
    perception_dim: int,
    grid_size: int,
    patch_size: int,
    complexity_probe_frames: int,
    rollout_stride: int,
    init_rollout_steps: int,
    complexity_min: float,
    complexity_max: float,
    max_rule_attempts_per_sequence: int,
    identity_bias: float,
    temperature: float,
) -> tuple[list[NCARule], list[float]]:
    target_rule_count = max(rule_count, 1)
    max_attempts = max(sequence_count * max_rule_attempts_per_sequence, target_rule_count)
    selected_rules: list[NCARule] = []
    selected_ratios: list[float] = []
    attempts = 0
    while len(selected_rules) < target_rule_count and attempts < max_attempts:
        attempts += 1
        rule = sample_rule(
            rng=rng,
            num_states=num_states,
            hidden_dim=hidden_dim,
            perception_dim=perception_dim,
        )
        probe = rollout_rule(
            rng=rng,
            rule=rule,
            grid_size=grid_size,
            num_states=num_states,
            frame_count=complexity_probe_frames,
            rollout_stride=rollout_stride,
            init_rollout_steps=init_rollout_steps,
            identity_bias=identity_bias,
            temperature=temperature,
        )
        ratio = compression_ratio(patch_token_payload(probe, num_states=num_states, patch_size=patch_size))
        if complexity_min <= ratio <= complexity_max:
            selected_rules.append(rule)
            selected_ratios.append(ratio)
    if len(selected_rules) < target_rule_count:
        raise RuntimeError(
            "Failed to generate enough NCA rules in the requested complexity band. "
            "Relax the band or increase max_rule_attempts_per_sequence."
        )
    return selected_rules, selected_ratios


def rollout_rule(
    rng: np.random.Generator,
    *,
    rule: NCARule,
    grid_size: int,
    num_states: int,
    frame_count: int,
    rollout_stride: int,
    init_rollout_steps: int,
    identity_bias: float,
    temperature: float,
) -> np.ndarray:
    grid = sample_initial_grid(rng, grid_size=grid_size, num_states=num_states)
    sample_steps = {
        init_rollout_steps + step * rollout_stride
        for step in range(frame_count)
    }
    total_steps = init_rollout_steps + frame_count * rollout_stride
    frames: list[np.ndarray] = []
    for step in range(total_steps):
        if step in sample_steps:
            frames.append(grid.copy())
        grid = step_grid(
            rng=rng,
            grid=grid,
            rule=rule,
            num_states=num_states,
            identity_bias=identity_bias,
            temperature=temperature,
        )
    if len(frames) != frame_count:
        raise RuntimeError("Unexpected NCA rollout length mismatch.")
    return np.stack(frames, axis=0)


def sample_initial_grid(rng: np.random.Generator, *, grid_size: int, num_states: int) -> np.ndarray:
    logits = rng.normal(size=(num_states,)).astype(np.float32)
    return categorical_sample(rng, np.broadcast_to(logits, (grid_size, grid_size, num_states)))


def step_grid(
    rng: np.random.Generator,
    *,
    grid: np.ndarray,
    rule: NCARule,
    num_states: int,
    identity_bias: float,
    temperature: float,
) -> np.ndarray:
    state_one_hot = np.eye(num_states, dtype=np.float32)[grid]
    logits = circular_conv3x3(state_one_hot, rule.conv3x3_weight, rule.conv3x3_bias)
    logits = conv1x1(logits, rule.conv1x1_hidden_weight, rule.conv1x1_hidden_bias)
    logits = np.maximum(logits, 0.0)
    logits = conv1x1(logits, rule.conv1x1_out_weight, rule.conv1x1_out_bias)
    logits = logits + identity_bias * state_one_hot
    if temperature <= 0.0:
        return logits.argmax(axis=-1).astype(np.int64)
    return categorical_sample(rng, logits / max(temperature, 1e-6))


def circular_conv3x3(inputs: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    outputs = np.zeros((*inputs.shape[:2], weight.shape[0]), dtype=np.float32)
    for row_shift in range(3):
        for col_shift in range(3):
            shifted = np.roll(np.roll(inputs, row_shift - 1, axis=0), col_shift - 1, axis=1)
            outputs += np.tensordot(shifted, weight[:, :, row_shift, col_shift], axes=([2], [1]))
    outputs += bias
    return outputs


def conv1x1(inputs: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    outputs = np.tensordot(inputs, weight[:, :, 0, 0], axes=([2], [1]))
    outputs += bias
    return outputs.astype(np.float32)


def categorical_sample(rng: np.random.Generator, logits: np.ndarray) -> np.ndarray:
    gumbel = -np.log(-np.log(rng.uniform(size=logits.shape).clip(1e-9, 1 - 1e-9)))
    return np.argmax(logits + gumbel, axis=-1).astype(np.int64)


def patchify_trajectory(
    trajectory: np.ndarray,
    *,
    num_states: int,
    patch_size: int,
    start_token_id: int,
    end_token_id: int,
) -> list[int]:
    tokens: list[int] = []
    for frame in trajectory:
        tokens.append(start_token_id)
        tokens.extend(frame_patch_tokens(frame, num_states=num_states, patch_size=patch_size))
        tokens.append(end_token_id)
    return tokens


def frame_patch_tokens(frame: np.ndarray, *, num_states: int, patch_size: int) -> list[int]:
    tokens: list[int] = []
    powers = num_states ** np.arange(patch_size * patch_size, dtype=np.int64)
    for row in range(0, frame.shape[0], patch_size):
        for col in range(0, frame.shape[1], patch_size):
            patch = frame[row : row + patch_size, col : col + patch_size].reshape(-1)
            tokens.append(int(np.dot(patch.astype(np.int64), powers)))
    return tokens


def patch_token_payload(trajectory: np.ndarray, *, num_states: int, patch_size: int) -> bytes:
    patch_tokens: list[int] = []
    for frame in trajectory:
        patch_tokens.extend(frame_patch_tokens(frame, num_states=num_states, patch_size=patch_size))
    payload = np.asarray(patch_tokens, dtype=np.int32)
    return payload.tobytes()


def compression_ratio(payload: bytes) -> float:
    compressed = gzip.compress(payload)
    return len(compressed) / max(len(payload), 1)


def create_training_example(
    tokens: list[int],
    *,
    max_length: int,
    frame_token_length: int,
    min_frames: int,
) -> tuple[list[int], list[int]]:
    masked_targets = list(tokens)
    for idx, token in enumerate(masked_targets):
        position_in_frame = idx % frame_token_length
        if position_in_frame in (0, frame_token_length - 1):
            masked_targets[idx] = -100
    prefix = min(min_frames * frame_token_length, len(masked_targets))
    for idx in range(prefix):
        masked_targets[idx] = -100

    input_ids = tokens[:-1][:max_length]
    labels = masked_targets[:-1][:max_length]
    return input_ids, labels


def required_frame_count(*, max_length: int, grid_size: int, patch_size: int) -> int:
    patches_per_frame = (grid_size // patch_size) ** 2
    frame_token_length = patches_per_frame + 2
    return max(1, math.ceil((max_length + 1) / frame_token_length))


def _lecun_normal(rng: np.random.Generator, shape: tuple[int, ...], *, fan_in: int) -> np.ndarray:
    std = math.sqrt(1.0 / fan_in)
    return rng.normal(scale=std, size=shape).astype(np.float32)
