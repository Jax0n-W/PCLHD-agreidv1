#!/usr/bin/env python3
"""Score same-basename enhanced aerial images with the original OOF folds."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

try:
    from .common import (IMAGE_SUFFIXES, ViewRecord, add_view_groups,
                         manifest_folds, write_csv, write_json)
    from .train_view_classifier import IMAGENET_MEAN, IMAGENET_STD
    from .view_classifier import load_view_checkpoint
    from .view_dataset import ForegroundRenderer, ViewDataset
except ImportError:
    from common import (IMAGE_SUFFIXES, ViewRecord, add_view_groups,
                        manifest_folds, write_csv, write_json)  # type: ignore
    from train_view_classifier import IMAGENET_MEAN, IMAGENET_STD  # type: ignore
    from view_classifier import load_view_checkpoint  # type: ignore
    from view_dataset import ForegroundRenderer, ViewDataset  # type: ignore


REQUIRED_REFERENCE_COLUMNS = ("image_path", "pid", "camera", "fold")


def _enhanced_images(root: Path) -> Dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError("Enhanced aerial directory does not exist: {}".format(root))
    result: Dict[str, Path] = {}
    for path in sorted(
            item.resolve() for item in root.rglob("*")
            if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES):
        if path.name in result:
            raise ValueError("Duplicate enhanced basename: {}".format(path.name))
        result[path.name] = path
    if not result:
        raise RuntimeError("No enhanced aerial images found under {}".format(root))
    return result


def build_enhanced_records(reference_scores: Path, enhanced_root: Path) -> List[ViewRecord]:
    images = _enhanced_images(enhanced_root)
    records: List[ViewRecord] = []
    missing = []
    with reference_scores.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        absent = set(REQUIRED_REFERENCE_COLUMNS) - set(reader.fieldnames or ())
        if absent:
            raise ValueError(
                "Reference score CSV is missing columns: {}".format(
                    ", ".join(sorted(absent))))
        for row in reader:
            basename = Path(row["image_path"]).name
            enhanced = images.get(basename)
            if enhanced is None:
                missing.append(row["image_path"])
                continue
            records.append(ViewRecord(
                image_path=str(enhanced), pid=row["pid"].strip(),
                camera=row["camera"].strip().lower(), domain="aerial",
                fold=int(row["fold"])))
    if missing:
        raise FileNotFoundError(
            "Enhanced directory does not cover the reference CSV: missing={}, "
            "first_missing={}".format(len(missing), missing[0]))
    if len(records) != len(images):
        record_names = {Path(record.image_path).name for record in records}
        extra = sorted(set(images) - record_names)
        raise ValueError(
            "Enhanced/reference counts differ: records={}, enhanced={}, "
            "first_extra={}".format(
                len(records), len(images), extra[0] if extra else "none"))
    return records


def _checkpoint_path(root: Path, fold: int) -> Path:
    path = root / "fold_{}".format(fold) / "best.pt"
    if not path.is_file():
        raise FileNotFoundError("Missing fold checkpoint: {}".format(path))
    return path


def score_checkpoint_set(records, checkpoint_root, expected_mode, device,
                         batch_size, workers, renderer):
    scores: Dict[str, float] = {}
    folds = manifest_folds(records)
    for fold in folds:
        model, checkpoint = load_view_checkpoint(
            _checkpoint_path(checkpoint_root, fold), device)
        metadata = checkpoint.get("metadata", {})
        checkpoint_fold = int(metadata.get("fold", fold))
        checkpoint_mode = metadata.get("mode", expected_mode)
        if checkpoint_fold != fold or checkpoint_mode != expected_mode:
            raise ValueError(
                "Checkpoint metadata mismatch in fold {}: fold={!r}, mode={!r}".format(
                    fold, checkpoint_fold, checkpoint_mode))
        transform = transforms.Compose([
            transforms.Resize((int(metadata.get("height", 288)),
                               int(metadata.get("width", 144)))),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        fold_records = [record for record in records if record.fold == fold]
        loader = DataLoader(
            ViewDataset(fold_records, transform, renderer),
            batch_size=batch_size, shuffle=False, num_workers=workers,
            pin_memory=device.type == "cuda")
        with torch.no_grad():
            for batch in loader:
                logits = model(batch["image"].to(device, non_blocking=True))
                probabilities = torch.sigmoid(logits).cpu().tolist()
                for path, probability in zip(batch["image_path"], probabilities):
                    if path in scores:
                        raise RuntimeError("Image scored more than once: {}".format(path))
                    scores[path] = float(probability)
        print("Scored fold {} with {} model: {} images".format(
            fold, expected_mode, len(fold_records)))
    if len(scores) != len(records):
        raise RuntimeError("OOF scoring covered {} of {} images".format(
            len(scores), len(records)))
    return scores


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score enhanced aerial images with original OOF fold checkpoints")
    parser.add_argument("--reference-scores", type=Path, required=True)
    parser.add_argument("--enhanced-aerial-dir", type=Path, required=True)
    parser.add_argument("--raw-checkpoint-dir", type=Path)
    parser.add_argument("--foreground-checkpoint-dir", type=Path)
    parser.add_argument("--aerial-mask-root", type=Path)
    parser.add_argument("--background-gray", type=int, default=128)
    parser.add_argument("--near-ratio", type=float, default=0.30)
    parser.add_argument("--grouping", choices=("global", "per-camera"),
                        default="per-camera")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.raw_checkpoint_dir is None and args.foreground_checkpoint_dir is None:
        parser.error("at least one checkpoint directory is required")
    if args.foreground_checkpoint_dir is not None and args.aerial_mask_root is None:
        parser.error("foreground scoring requires --aerial-mask-root")
    if not 0.0 < args.near_ratio < 1.0:
        parser.error("--near-ratio must be between 0 and 1")
    return args


def main():
    args = parse_args()
    records = build_enhanced_records(
        args.reference_scores, args.enhanced_aerial_dir)
    folds = manifest_folds(records)
    print("Enhanced records: {}; OOF folds: {}".format(len(records), folds))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    raw_scores = None
    foreground_scores = None
    if args.raw_checkpoint_dir is not None:
        raw_scores = score_checkpoint_set(
            records, args.raw_checkpoint_dir, "raw", device,
            args.batch_size, args.workers, None)
    if args.foreground_checkpoint_dir is not None:
        renderer = ForegroundRenderer(
            source_roots={"aerial": args.enhanced_aerial_dir,
                          "ground": args.enhanced_aerial_dir},
            mask_roots={"aerial": args.aerial_mask_root,
                        "ground": args.aerial_mask_root},
            background_gray=args.background_gray)
        foreground_scores = score_checkpoint_set(
            records, args.foreground_checkpoint_dir, "foreground", device,
            args.batch_size, args.workers, renderer)

    rows = []
    for record in records:
        raw = raw_scores[record.image_path] if raw_scores is not None else None
        foreground = (foreground_scores[record.image_path]
                      if foreground_scores is not None else None)
        available = [value for value in (raw, foreground) if value is not None]
        rows.append({
            "image_path": record.image_path, "pid": record.pid,
            "camera": record.camera, "fold": record.fold,
            "p_ground_raw": "" if raw is None else raw,
            "p_ground_foreground": "" if foreground is None else foreground,
            "p_ground": sum(available) / len(available),
        })
    rows, thresholds = add_view_groups(rows, args.near_ratio)
    selected = ("view_group_global" if args.grouping == "global"
                else "view_group_per_camera")
    for row in rows:
        row["view_group"] = row[selected]
    rows.sort(key=lambda row: (row["camera"], row["image_path"]))
    fieldnames = [
        "image_path", "pid", "camera", "fold", "p_ground_raw",
        "p_ground_foreground", "p_ground", "view_group",
        "view_group_global", "view_group_per_camera"]
    write_csv(args.output, fieldnames, rows)
    write_json(args.output.with_name(args.output.stem + "_thresholds.json"), {
        "near_ratio": args.near_ratio, "selected_grouping": args.grouping,
        "folds": folds, "thresholds": thresholds})
    print("Wrote {} enhanced OOF scores to {}".format(len(rows), args.output))


if __name__ == "__main__":
    main()

