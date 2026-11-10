#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List

try:
    from .common import (
        ViewRecord,
        camera_sort_key,
        discover_images,
        parse_image_name,
        write_manifest,
    )
except ImportError:
    from common import (  # type: ignore
        ViewRecord,
        camera_sort_key,
        discover_images,
        parse_image_name,
        write_manifest,
    )




def comma_list(value: str) -> List[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def parse_ground_folds(value: str, expected_folds: int) -> List[List[str]]:
    groups = [comma_list(group) for group in value.split(";") if group.strip()]
    if len(groups) != expected_folds or any(not group for group in groups):
        raise argparse.ArgumentTypeError(
            "--ground-folds must contain {} non-empty semicolon-separated groups"
            .format(expected_folds)
        )
    flat = [camera for group in groups for camera in group]
    if len(flat) != len(set(flat)):
        raise argparse.ArgumentTypeError("a ground camera appears in more than one fold")
    return groups


def scan_domain(root: Path, domain: str) -> List[dict]:
    rows = []
    for path in discover_images(root):
        pid, camera = parse_image_name(path)
        rows.append(
            {"image_path": str(path), "pid": pid, "camera": camera, "domain": domain}
        )
    return rows


def build_records(args: argparse.Namespace) -> List[ViewRecord]:
    rows = scan_domain(args.aerial_dir, "aerial") + scan_domain(
        args.ground_dir, "ground"
    )
    aerial_present = sorted(
        {row["camera"] for row in rows if row["domain"] == "aerial"},
        key=camera_sort_key,
    )
    ground_present = sorted(
        {row["camera"] for row in rows if row["domain"] == "ground"},
        key=camera_sort_key,
    )
    aerial_cameras = (
        comma_list(args.aerial_cameras) if args.aerial_cameras else aerial_present
    )
    if len(aerial_cameras) < 2 or len(set(aerial_cameras)) != len(aerial_cameras):
        raise ValueError("at least two distinct aerial cameras are required")
    if set(aerial_present) != set(aerial_cameras):
        raise ValueError(
            "Aerial cameras in data ({}) do not match --aerial-cameras ({})".format(
                ", ".join(aerial_present), ", ".join(aerial_cameras)
            )
        )

    num_folds = len(aerial_cameras)
    if args.ground_folds:
        ground_groups = parse_ground_folds(args.ground_folds, num_folds)
    else:
        if len(ground_present) % num_folds != 0:
            raise ValueError(
                "Ground camera count {} is not divisible by {} folds. Pass an "
                "explicit --ground-folds mapping.".format(
                    len(ground_present), num_folds
                )
            )
        cameras_per_fold = len(ground_present) // num_folds
        ground_groups = [
            ground_present[index * cameras_per_fold : (index + 1) * cameras_per_fold]
            for index in range(num_folds)
        ]
    configured_ground = {camera for group in ground_groups for camera in group}
    if configured_ground != set(ground_present):
        raise ValueError(
            "Ground cameras in data ({}) do not match --ground-folds ({})".format(
                ", ".join(ground_present), ", ".join(sorted(configured_ground, key=camera_sort_key))
            )
        )

    fold_by_domain_camera: Dict[tuple, int] = {}
    for fold, camera in enumerate(aerial_cameras):
        fold_by_domain_camera[("aerial", camera)] = fold
    for fold, cameras in enumerate(ground_groups):
        for camera in cameras:
            fold_by_domain_camera[("ground", camera)] = fold

    records = [
        ViewRecord(
            image_path=row["image_path"],
            pid=row["pid"],
            camera=row["camera"],
            domain=row["domain"],
            fold=fold_by_domain_camera[(row["domain"], row["camera"])],
        )
        for row in rows
    ]
    records.sort(key=lambda row: (row.fold, row.domain, row.camera, row.pid, row.image_path))
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a camera-disjoint aerial/ground cross-validation manifest."
    )
    parser.add_argument("--aerial-dir", type=Path, required=True)
    parser.add_argument("--ground-dir", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("view_domain/outputs/view_manifest.csv")
    )
    parser.add_argument(
        "--aerial-cameras",
        default="",
        help="Ordered comma-separated aerial cameras. When omitted, discovered "
        "cameras are sorted numerically; their count defines the fold count.",
    )
    parser.add_argument(
        "--ground-folds",
        default="",
        help="Semicolon-separated ground-camera groups, one group per fold. "
        "When omitted, sorted ground cameras are divided equally across folds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = build_records(args)
    write_manifest(args.output, records)
    counts = Counter((row.fold, row.domain) for row in records)
    print("Wrote {} records to {}".format(len(records), args.output))
    for fold in sorted({record.fold for record in records}):
        print(
            "fold {} validation cameras: aerial={} ground={} | images: aerial={} ground={}".format(
                fold,
                sorted({r.camera for r in records if r.fold == fold and r.domain == "aerial"}, key=camera_sort_key),
                sorted({r.camera for r in records if r.fold == fold and r.domain == "ground"}, key=camera_sort_key),
                counts[(fold, "aerial")],
                counts[(fold, "ground")],
            )
        )


if __name__ == "__main__":
    main()

