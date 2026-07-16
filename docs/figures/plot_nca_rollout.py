"""Render a deterministic pptrain NCA rollout for publication."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from pptrain.tasks.nca.generator import rollout_rule, sample_rule


COLORS = [
    "#17132A",
    "#3F21B6",
    "#7560D0",
    "#B4A7E8",
    "#176B68",
    "#64A7A2",
    "#A85B35",
    "#D59765",
    "#8E3F67",
    "#D181A4",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("nca-rollout.png"))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    rule = sample_rule(rng, num_states=10, hidden_dim=16, perception_dim=4)
    frames = rollout_rule(
        rng=rng,
        rule=rule,
        grid_size=24,
        num_states=10,
        frame_count=4,
        rollout_stride=8,
        init_rollout_steps=0,
        identity_bias=0.0,
        temperature=1e-4,
    )

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.facecolor": "none",
        "figure.facecolor": "none",
        "savefig.facecolor": "none",
    })
    figure, axes = plt.subplots(1, 4, figsize=(7.2, 1.85), constrained_layout=True)
    cmap = ListedColormap(COLORS)
    for index, (axis, frame) in enumerate(zip(axes, frames, strict=True)):
        axis.imshow(frame, cmap=cmap, vmin=0, vmax=9, interpolation="nearest")
        axis.set_xlabel(f"Step {index * 8}", labelpad=5)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#D7D7D7")
            spine.set_linewidth(0.6)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=240, bbox_inches="tight", transparent=True)
    plt.close(figure)


if __name__ == "__main__":
    main()
