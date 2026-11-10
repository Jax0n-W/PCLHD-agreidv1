#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw a compact Ground/Near/Far clustering schematic for paper figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


SEED = 17

DOMAINS = {
    "Ground": {
        "color": "#70A6D2",
        "marker": "^",
        "centers": [(-1.38, 0.34), (-1.08, -0.35), (-0.72, 0.02)],
        "counts": [6, 6, 6],
    },
    "Near": {
        "color": "#D69A5B",
        "marker": "^",
        "centers": [(-0.40, 0.37), (-0.05, -0.37), (0.27, 0.03)],
        "counts": [6, 6, 6],
    },
    "Far": {
        "color": "#75AD79",
        "marker": "^",
        "centers": [(0.36, 0.39), (0.72, -0.34), (1.05, 0.08)],
        "counts": [6, 6, 6],
    },
}


def sample_cluster(rng, center, count, cluster_index):
    """Generate a compact, slightly anisotropic 2-D Gaussian cluster."""
    angles = (0.30, -0.52, 0.14)
    scales = ((0.23, 0.16), (0.22, 0.18), (0.24, 0.15))
    angle = angles[cluster_index]
    rotation = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])
    noise = rng.normal(size=(count, 2))
    noise = noise @ np.diag(scales[cluster_index]) @ rotation.T
    return noise + np.asarray(center)


def draw(output_dir):
    rng = np.random.default_rng(SEED)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "font.weight": "normal",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })

    figure, axis = plt.subplots(figsize=(5.5, 2.8), facecolor="white")
    axis.set_facecolor("none")
    axis.set_xlim(-2.00, 1.72)
    axis.set_ylim(-1.12, 1.05)
    axis.set_aspect("equal", adjustable="box")

    container = FancyBboxPatch(
        (-1.92, -1.02), 3.55, 1.96,
        boxstyle="round,pad=0.035,rounding_size=0.15",
        linewidth=0.9, edgecolor="#D2D5D8", facecolor="#F7F8F8",
        zorder=0, clip_on=False,
    )
    axis.add_patch(container)

    for domain, config in DOMAINS.items():
        for index, (center, count) in enumerate(zip(
                config["centers"], config["counts"])):
            points = sample_cluster(rng, center, count, index)
            axis.scatter(
                points[:, 0], points[:, 1],
                s=25, marker=config["marker"], c=config["color"],
                alpha=0.84, edgecolors="none", zorder=2,
            )
            axis.scatter(
                [center[0]], [center[1]],
                s=72, marker=config["marker"], facecolors="none",
                edgecolors=config["color"], linewidths=1.35, zorder=3,
            )

    label_positions = {
        "Ground": (-1.18, -0.84),
        "Near": (-0.05, -0.84),
        "Far": (0.87, -0.84),
    }
    for domain, position in label_positions.items():
        axis.text(
            position[0], position[1], domain,
            ha="center", va="center", fontsize=9.5,
            color="#34373A", zorder=4,
        )

    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)

    figure.subplots_adjust(left=0.015, right=0.985, bottom=0.035, top=0.965)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "ground_near_far_schematic.png"
    svg_path = output_dir / "ground_near_far_schematic.svg"
    figure.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight",
                   pad_inches=0.025)
    figure.savefig(svg_path, facecolor="white", bbox_inches="tight",
                   pad_inches=0.025)
    plt.close(figure)
    print("Saved {}".format(png_path))
    print("Saved {}".format(svg_path))


if __name__ == "__main__":
    draw("figures/ground_near_far_schematic")
