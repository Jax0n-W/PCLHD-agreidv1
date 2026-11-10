#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw a score-guided Far/Near/Ground schematic (not a feature t-SNE)."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "A_far": "#3B82C4",
    "A_near": "#F28E2B",
    "Ground": "#59A14F",
}
MARKERS = {"A_far": "o", "A_near": "^", "Ground": "s"}
LABELS = {
    "A_far": r"$A_{\mathrm{far}}$",
    "A_near": r"$A_{\mathrm{near}}$",
    "Ground": "Ground",
}


def read_scores(path, score_column):
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = row.get(score_column, "")
            if value in (None, ""):
                continue
            rows.append((row["image_path"], row["camera"], float(value)))
    if not rows:
        raise RuntimeError("No usable scores in {} column {}".format(path, score_column))
    return rows


def camera_percentiles(rows):
    by_camera = {}
    for path, camera, score in rows:
        by_camera.setdefault(camera, []).append((path, score))
    output = []
    cameras = sorted(by_camera)
    camera_offset = {
        camera: (index - (len(cameras) - 1) / 2.0) * 0.055
        for index, camera in enumerate(cameras)
    }
    for camera in cameras:
        ordered = sorted(by_camera[camera], key=lambda item: (item[1], item[0]))
        count = len(ordered)
        for rank, (path, score) in enumerate(ordered):
            percentile = (rank + 0.5) / float(count)
            output.append((path, camera, score, percentile, camera_offset[camera]))
    return output


def select_equal(points, group, count, rng):
    indexes = np.flatnonzero(points["group"] == group)
    if len(indexes) > count:
        indexes = rng.choice(indexes, size=count, replace=False)
    return np.sort(indexes)


def build_projection(rows, near_ratio, samples_per_group, seed):
    ranked = camera_percentiles(rows)
    rng = np.random.default_rng(seed)
    threshold = 1.0 - near_ratio
    dtype = [("group", "U10"), ("x", "f8"), ("y", "f8")]
    aerial = np.empty(len(ranked), dtype=dtype)
    for index, (_, _, _, percentile, camera_offset) in enumerate(ranked):
        group = "A_near" if percentile >= threshold else "A_far"
        # The ordered x coordinate is score-derived; small fixed noise only
        # prevents overplotting. The y coordinate is schematic jitter.
        within_group = (percentile / threshold if group == "A_far" else
                        (percentile - threshold) / near_ratio)
        if group == "A_far":
            # Preserve a gradual Far-to-Near transition instead of making the
            # two aerial groups artificially separable.
            x = -1.05 + 0.78 * within_group + rng.normal(0.0, 0.25)
        else:
            # Near-view aerial samples overlap Ground more strongly, while
            # retaining a distinct mean position.
            x = -0.20 + 0.78 * within_group + rng.normal(0.0, 0.27)
        aerial[index] = (
            group,
            x,
            0.12 * np.sin(percentile * 3.0 * np.pi)
            + camera_offset + rng.normal(0.0, 0.39),
        )

    selected = []
    for group in ("A_far", "A_near"):
        selected.append(aerial[select_equal(aerial, group, samples_per_group, rng)])
    aerial = np.concatenate(selected)

    ground = np.empty(samples_per_group, dtype=dtype)
    ground["group"] = "Ground"
    ground["x"] = rng.normal(0.72, 0.38, samples_per_group)
    ground["y"] = rng.normal(0.02, 0.42, samples_per_group)
    return np.concatenate((aerial, ground))


def draw_panel(axis, points, title):
    for group in ("A_far", "A_near", "Ground"):
        subset = points[points["group"] == group]
        axis.scatter(
            subset["x"], subset["y"], s=9, marker=MARKERS[group],
            c=COLORS[group], alpha=0.58, linewidths=0,
            rasterized=True)
    axis.set_title(title, fontsize=10, pad=3)
    axis.set_xlim(-1.75, 1.75)
    axis.set_ylim(-1.45, 1.40)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agreid-scores", required=True)
    parser.add_argument("--lag-scores", required=True)
    parser.add_argument("--score-column", default="p_ground_foreground")
    parser.add_argument("--near-ratio", type=float, default=0.50)
    parser.add_argument("--samples-per-group", type=int, default=180)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", default="figures/three_domain_score_schematic")
    args = parser.parse_args()

    if not 0.0 < args.near_ratio < 1.0:
        raise ValueError("--near-ratio must be in (0, 1)")

    datasets = (
        ("AG-ReID.v1", args.agreid_scores, args.seed),
        ("LAGPeR", args.lag_scores, args.seed + 17),
    )
    projections = []
    for title, path, seed in datasets:
        rows = read_scores(path, args.score_column)
        projection = build_projection(
            rows, args.near_ratio, args.samples_per_group, seed)
        projections.append((title, projection))
        counts = {group: int(np.sum(projection["group"] == group))
                  for group in ("A_far", "A_near", "Ground")}
        print("{}: input={}, plotted={}".format(title, len(rows), counts))

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    figure, axes = plt.subplots(1, 2, figsize=(5.6, 2.25), sharey=True)
    for axis, (title, points) in zip(axes, projections):
        draw_panel(axis, points, title)
    figure.tight_layout(pad=0.15, w_pad=0.8)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "far_near_ground_score_schematic.png"
    pdf = output_dir / "far_near_ground_score_schematic.pdf"
    figure.savefig(png, dpi=600, bbox_inches="tight")
    figure.savefig(pdf, dpi=600, bbox_inches="tight")
    plt.close(figure)
    print("Saved {}".format(png))
    print("Saved {}".format(pdf))


if __name__ == "__main__":
    main()
