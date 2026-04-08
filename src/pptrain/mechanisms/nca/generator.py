from __future__ import annotations

import gzip
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class NCARule:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray


def sample_rule(rng: np.random.Generator, num_states: int, hidden_dim: int) -> NCARule:
    feature_dim = 9 * num_states
    return NCARule(
        w1=rng.normal(scale=0.35, size=(feature_dim, hidden_dim)).astype(np.float32),
        b1=rng.normal(scale=0.05, size=(hidden_dim,)).astype(np.float32),
        w2=rng.normal(scale=0.35, size=(hidden_dim, num_states)).astype(np.float32),
        b2=rng.normal(scale=0.05, size=(num_states,)).astype(np.float32),
    )


def rollout_rule(
    rng: np.random.Generator,
    rule: NCARule,
    grid_size: int,
    num_states: int,
    rollout_steps: int,
    stochasticity: float,
) -> np.ndarray:
    grid = rng.integers(0, num_states, size=(grid_size, grid_size), endpoint=False, dtype=np.int64)
    trajectory = [grid.copy()]
    for _ in range(rollout_steps - 1):
        grid = step_grid(rng, grid, rule, num_states=num_states, stochasticity=stochasticity)
        trajectory.append(grid.copy())
    return np.stack(trajectory, axis=0)


def step_grid(
    rng: np.random.Generator,
    grid: np.ndarray,
    rule: NCARule,
    num_states: int,
    stochasticity: float,
) -> np.ndarray:
    neighborhoods = [
        np.roll(np.roll(grid, shift_y, axis=0), shift_x, axis=1)
        for shift_y in (-1, 0, 1)
        for shift_x in (-1, 0, 1)
    ]
    stacked = np.stack(neighborhoods, axis=-1)
    one_hot = np.eye(num_states, dtype=np.float32)[stacked]
    features = one_hot.reshape(grid.shape[0], grid.shape[1], 9 * num_states)
    hidden = np.tanh(features @ rule.w1 + rule.b1)
    logits = hidden @ rule.w2 + rule.b2
    if stochasticity > 0:
        logits = logits + rng.normal(scale=stochasticity, size=logits.shape)
    return logits.argmax(axis=-1).astype(np.int64)


def compression_ratio(trajectory: np.ndarray) -> float:
    if trajectory.max() < 256:
        payload = trajectory.astype(np.uint8).tobytes()
    else:
        payload = trajectory.astype(np.uint16).tobytes()
    compressed = gzip.compress(payload)
    return len(compressed) / max(len(payload), 1)


def patchify_trajectory(
    trajectory: np.ndarray,
    num_states: int,
    patch_size: int,
    bos_token_id: int,
    eos_token_id: int,
    time_token_id: int,
    offset: int,
    max_length: int,
) -> list[int]:
    tokens = [bos_token_id]
    height, width = trajectory.shape[1:]
    for frame in trajectory:
        tokens.append(time_token_id)
        for row in range(0, height, patch_size):
            for col in range(0, width, patch_size):
                patch = frame[row : row + patch_size, col : col + patch_size].reshape(-1)
                token = 0
                for value in patch:
                    token = token * num_states + int(value)
                tokens.append(offset + token)
                if len(tokens) >= max_length - 1:
                    tokens.append(eos_token_id)
                    return tokens
    tokens.append(eos_token_id)
    return tokens

