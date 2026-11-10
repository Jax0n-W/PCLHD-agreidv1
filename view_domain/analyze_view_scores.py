#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw, ImageOps

try:
    from .common import add_view_groups, camera_sort_key, write_csv, write_json
except ImportError:
    from common import add_view_groups, camera_sort_key, write_csv, write_json  # type: ignore


def read_scores(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"image_path", "pid", "camera", "p_ground"}
    missing = required - set(rows[0] if rows else ())
    if missing:
        raise ValueError("Score CSV is missing: {}".format(", ".join(sorted(missing))))
    for row in rows:
        row["p_ground"] = float(row["p_ground"])
    return rows


def camera_statistics(rows: List[dict]) -> List[dict]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["camera"]].append(row)
    output = []
    for camera in sorted(grouped, key=camera_sort_key):
        camera_rows = grouped[camera]
        scores = [float(row["p_ground"]) for row in camera_rows]
        global_near = sum(
            row["view_group_global"] == "A_near" for row in camera_rows
        )
        per_camera_near = sum(
            row["view_group_per_camera"] == "A_near" for row in camera_rows
        )
        output.append(
            {
                "camera": camera,
                "count": len(scores),
                "mean_p_ground": statistics.mean(scores),
                "median_p_ground": statistics.median(scores),
                "std_p_ground": statistics.pstdev(scores),
                "min_p_ground": min(scores),
                "max_p_ground": max(scores),
                "global_near_count": global_near,
                "global_near_ratio": global_near / len(camera_rows),
                "per_camera_near_count": per_camera_near,
                "per_camera_near_ratio": per_camera_near / len(camera_rows),
            }
        )
    return output


def save_histogram(rows: List[dict], path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; skipped score_histogram.png")
        return False
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        grouped[row["camera"]].append(float(row["p_ground"]))
    figure, axis = plt.subplots(figsize=(9, 5))
    for camera in sorted(grouped, key=camera_sort_key):
        axis.hist(grouped[camera], bins=30, alpha=0.45, label=camera, density=True)
    axis.set_xlabel("OOF p_ground")
    axis.set_ylabel("Density")
    axis.set_title("Aerial view-score distribution by camera")
    axis.legend()
    figure.tight_layout()
    figure.savefig(str(path), dpi=160)
    plt.close(figure)
    return True


def save_sample_grid(rows: List[dict], path: Path, samples_per_side: int) -> bool:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["camera"]].append(row)
    cameras = sorted(grouped, key=camera_sort_key)
    cell_width, image_height, label_height = 144, 288, 28
    columns = samples_per_side * 2
    canvas = Image.new(
        "RGB",
        (columns * cell_width, len(cameras) * (image_height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    loaded = 0
    for row_index, camera in enumerate(cameras):
        ranked = sorted(grouped[camera], key=lambda row: float(row["p_ground"]))
        selected = ranked[:samples_per_side] + ranked[-samples_per_side:]
        for column, row in enumerate(selected):
            try:
                with Image.open(row["image_path"]) as handle:
                    image = ImageOps.fit(handle.convert("RGB"), (cell_width, image_height))
            except (FileNotFoundError, OSError):
                continue
            x = column * cell_width
            y = row_index * (image_height + label_height)
            canvas.paste(image, (x, y))
            draw.rectangle(
                (x, y + image_height, x + cell_width, y + image_height + label_height),
                fill="white",
            )
            draw.text(
                (x + 3, y + image_height + 6),
                "{} p={:.3f}".format(camera, float(row["p_ground"])),
                fill="black",
            )
            loaded += 1
    if not loaded:
        print("No source images were readable; skipped score_sample_grid.jpg")
        return False
    canvas.save(str(path), quality=92)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze camera dominance and visualize aerial OOF view scores."
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=Path("view_domain/outputs/aerial_view_scores_oof.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("view_domain/outputs/analysis")
    )
    parser.add_argument("--near-ratio", type=float, default=0.30)
    parser.add_argument("--samples-per-side", type=int, default=3)
    args = parser.parse_args()
    if not 0.0 < args.near_ratio < 1.0:
        parser.error("--near-ratio must be between 0 and 1")
    if args.samples_per_side < 1:
        parser.error("--samples-per-side must be positive")
    return args


def main() -> None:
    args = parse_args()
    rows, thresholds = add_view_groups(read_scores(args.scores), args.near_ratio)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_fields = list(rows[0])
    global_rows = [dict(row, view_group=row["view_group_global"]) for row in rows]
    per_camera_rows = [
        dict(row, view_group=row["view_group_per_camera"]) for row in rows
    ]
    output_fields = base_fields + ([] if "view_group" in base_fields else ["view_group"])
    write_csv(args.output_dir / "aerial_view_scores_global.csv", output_fields, global_rows)
    write_csv(
        args.output_dir / "aerial_view_scores_per_camera.csv",
        output_fields,
        per_camera_rows,
    )
    stats = camera_statistics(rows)
    write_csv(args.output_dir / "camera_summary.csv", list(stats[0]), stats)
    global_near = [row for row in rows if row["view_group_global"] == "A_near"]
    near_by_camera = Counter(row["camera"] for row in global_near)
    max_share = max(near_by_camera.values()) / len(global_near)
    summary = {
        "image_count": len(rows),
        "near_ratio_requested": args.near_ratio,
        "global_near_count": len(global_near),
        "global_near_camera_counts": dict(near_by_camera),
        "largest_camera_share_of_global_near": max_share,
        "thresholds": thresholds,
        "camera_statistics": stats,
    }
    write_json(args.output_dir / "summary.json", summary)
    save_histogram(rows, args.output_dir / "score_histogram.png")
    save_sample_grid(
        rows, args.output_dir / "score_sample_grid.jpg", args.samples_per_side
    )
    print("Wrote analysis to {}".format(args.output_dir))
    print(
        "Largest camera share of Global Top {:.0%}: {:.1%}".format(
            args.near_ratio, max_share
        )
    )


if __name__ == "__main__":
    main()

