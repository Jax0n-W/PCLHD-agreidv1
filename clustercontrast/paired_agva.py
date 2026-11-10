# -*- coding: utf-8 -*-
"""Paired original/AGVA utilities for the three-domain Stage 2 experiment.

The module deliberately contains no loss, Trainer, clustering, or BGM code.
It only makes the two existing modules operate on the same aerial sample:

* strict same-basename original/enhanced image pairing;
* original/enhanced OOF score fusion and per-camera Far/Near assignment;
* normalized online/EMA feature fusion;
* one deterministic, stratified epoch path decision shared by every loader.
"""

from __future__ import absolute_import

import collections
import csv
import math
import os
import os.path as osp
import random

import torch
import torch.nn.functional as F


IMAGE_EXTENSIONS = {
    ".bmp", ".jpeg", ".jpg", ".png", ".ppm", ".tif", ".tiff", ".webp",
}


def validate_unit_interval(value, name):
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("{} must be a finite number in [0, 1]".format(name))
    return value


def _basename(path):
    return osp.basename(str(path).strip().replace("\\", "/"))


def _unique_sample_basenames(samples):
    by_name = {}
    duplicates = []
    for sample in samples:
        path = sample[0]
        name = _basename(path)
        if name in by_name:
            duplicates.append(name)
        else:
            by_name[name] = path
    if duplicates:
        raise ValueError(
            "original aerial samples contain duplicate basenames: {}".format(
                ", ".join(sorted(set(duplicates))[:5])))
    return by_name


def index_image_directory(root):
    """Index all supported images below *root* by unique basename."""
    root = osp.abspath(osp.expanduser(str(root)))
    if not osp.isdir(root):
        raise FileNotFoundError(
            "enhanced aerial directory does not exist: {}".format(root))
    by_name = {}
    duplicates = []
    for directory, _, filenames in os.walk(root):
        for filename in filenames:
            if osp.splitext(filename)[1].lower() not in IMAGE_EXTENSIONS:
                continue
            path = osp.join(directory, filename)
            if filename in by_name:
                duplicates.append(filename)
            else:
                by_name[filename] = path
    if duplicates:
        raise ValueError(
            "enhanced aerial directory contains duplicate basenames: {}".format(
                ", ".join(sorted(set(duplicates))[:5])))
    return by_name


def build_aerial_pair_map(original_samples, enhanced_root):
    """Return a strict original-path -> enhanced-path bijection."""
    original_by_name = _unique_sample_basenames(original_samples)
    enhanced_by_name = index_image_directory(enhanced_root)
    missing = sorted(set(original_by_name) - set(enhanced_by_name))
    extra = sorted(set(enhanced_by_name) - set(original_by_name))
    if missing:
        raise FileNotFoundError(
            "enhanced aerial images do not completely cover training data: "
            "matched={}, expected={}, missing={}, first_missing={}".format(
                len(original_by_name) - len(missing), len(original_by_name),
                len(missing), missing[0]))
    pair_map = {
        original_path: enhanced_by_name[name]
        for name, original_path in original_by_name.items()
    }
    stats = {
        "pairs": len(pair_map),
        "missing": len(missing),
        "extra_enhanced_images": len(extra),
    }
    return pair_map, stats


def build_enhanced_trainset(original_samples, pair_map):
    output = []
    missing = []
    for original_path, label, camera in original_samples:
        enhanced_path = pair_map.get(original_path)
        if enhanced_path is None:
            missing.append(original_path)
        else:
            output.append((enhanced_path, label, camera))
    if missing:
        raise KeyError(
            "pair map does not cover original samples: missing={}, first={}".format(
                len(missing), missing[0]))
    return output


def _read_score_csv(path, score_column):
    path = osp.abspath(osp.expanduser(str(path)))
    if not osp.isfile(path):
        raise FileNotFoundError("view-score CSV does not exist: {}".format(path))
    rows = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "image_path", "pid", "camera", "fold", score_column,
        }
        missing_columns = sorted(required - set(reader.fieldnames or []))
        if missing_columns:
            raise ValueError(
                "view-score CSV is missing columns {}: {}".format(
                    ", ".join(missing_columns), path))
        for line_number, row in enumerate(reader, start=2):
            name = _basename(row.get("image_path", ""))
            if not name:
                raise ValueError(
                    "empty image_path at {}:{}".format(path, line_number))
            if name in rows:
                raise ValueError(
                    "duplicate basename {} in {}".format(name, path))
            try:
                score = float(row.get(score_column, ""))
                fold = int(row.get("fold", ""))
            except (TypeError, ValueError):
                raise ValueError(
                    "invalid score/fold at {}:{}".format(path, line_number))
            if not math.isfinite(score):
                raise ValueError(
                    "non-finite score at {}:{}".format(path, line_number))
            rows[name] = {
                "image_path": (row.get("image_path") or "").strip(),
                "pid": (row.get("pid") or "").strip(),
                "camera": (row.get("camera") or "").strip().lower(),
                "fold": fold,
                "score": score,
            }
    if not rows:
        raise ValueError("view-score CSV has no data rows: {}".format(path))
    return rows


def load_paired_view_scores(original_csv, enhanced_csv, original_samples,
                            original_score_column="p_ground_foreground",
                            enhanced_score_column="p_ground_foreground",
                            pair_score_alpha=0.5):
    """Load and validate two OOF score files, then fuse scores by basename."""
    alpha = validate_unit_interval(pair_score_alpha, "pair_score_alpha")
    original_rows = _read_score_csv(original_csv, original_score_column)
    enhanced_rows = _read_score_csv(enhanced_csv, enhanced_score_column)
    sample_by_name = _unique_sample_basenames(original_samples)
    expected = set(sample_by_name)
    original_names = set(original_rows)
    enhanced_names = set(enhanced_rows)
    errors = []
    for label, names in (
            ("original CSV", original_names),
            ("enhanced CSV", enhanced_names)):
        missing = sorted(expected - names)
        extra = sorted(names - expected)
        if missing:
            errors.append(
                "{} missing {} (first={})".format(label, len(missing), missing[0]))
        if extra:
            errors.append(
                "{} has {} unexpected rows (first={})".format(
                    label, len(extra), extra[0]))
    if errors:
        raise ValueError("paired score coverage mismatch: " + "; ".join(errors))

    entries = {}
    metadata_mismatches = []
    for name in sorted(expected):
        original = original_rows[name]
        enhanced = enhanced_rows[name]
        original_metadata = (
            original["pid"], original["camera"], original["fold"])
        enhanced_metadata = (
            enhanced["pid"], enhanced["camera"], enhanced["fold"])
        if original_metadata != enhanced_metadata:
            metadata_mismatches.append(
                (name, original_metadata, enhanced_metadata))
            continue
        original_path = sample_by_name[name]
        pair_score = (
            (1.0 - alpha) * original["score"]
            + alpha * enhanced["score"])
        entries[original_path] = {
            "basename": name,
            "pid": original["pid"],
            "camera": original["camera"],
            "fold": original["fold"],
            "original_score": original["score"],
            "enhanced_score": enhanced["score"],
            "pair_score": pair_score,
            "original_csv_path": original["image_path"],
            "enhanced_csv_path": enhanced["image_path"],
        }
    if metadata_mismatches:
        first = metadata_mismatches[0]
        raise ValueError(
            "original/enhanced score metadata mismatch: count={}, first={} "
            "original={} enhanced={}".format(
                len(metadata_mismatches), first[0], first[1], first[2]))
    return entries, {
        "pairs": len(entries),
        "pair_score_alpha": alpha,
        "original_csv_rows": len(original_rows),
        "enhanced_csv_rows": len(enhanced_rows),
    }


def assign_paired_view_groups(entries, near_ratio):
    """Assign deterministic per-camera top-ratio A_near/A_far groups."""
    near_ratio = float(near_ratio)
    if not math.isfinite(near_ratio) or not 0.0 < near_ratio < 1.0:
        raise ValueError("near_ratio must be a finite number in (0, 1)")
    grouped = collections.defaultdict(list)
    for original_path, entry in entries.items():
        grouped[entry["camera"]].append((original_path, entry))

    matched = {}
    camera_stats = {}
    for camera in sorted(grouped):
        ordered = sorted(
            grouped[camera],
            key=lambda item: (-item[1]["pair_score"], item[1]["basename"]))
        count = len(ordered)
        near_count = int(math.ceil(count * near_ratio))
        for rank, (original_path, entry) in enumerate(ordered):
            output = dict(entry)
            output["rank_in_camera"] = rank
            output["camera_count"] = count
            output["percentile_score"] = 1.0 - float(rank) / float(count)
            output["view_group"] = (
                "A_near" if rank < near_count else "A_far")
            matched[original_path] = output
        camera_stats[camera] = {
            "count": count,
            "near": near_count,
            "far": count - near_count,
        }
    return matched, {
        "near_ratio": near_ratio,
        "near": sum(v["near"] for v in camera_stats.values()),
        "far": sum(v["far"] for v in camera_stats.values()),
        "camera_stats": camera_stats,
    }


def write_pair_score_audit(path, entries, pair_map):
    """Write the exact paired scores/groups used by Stage 2."""
    directory = osp.dirname(osp.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    fields = [
        "basename", "pid", "camera", "fold", "original_path",
        "enhanced_path", "original_score", "enhanced_score", "pair_score",
        "rank_in_camera", "camera_count", "percentile_score", "view_group",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for original_path in sorted(entries, key=lambda p: entries[p]["basename"]):
            entry = entries[original_path]
            writer.writerow({
                "basename": entry["basename"],
                "pid": entry["pid"],
                "camera": entry["camera"],
                "fold": entry["fold"],
                "original_path": original_path,
                "enhanced_path": pair_map[original_path],
                "original_score": "{:.10f}".format(entry["original_score"]),
                "enhanced_score": "{:.10f}".format(entry["enhanced_score"]),
                "pair_score": "{:.10f}".format(entry["pair_score"]),
                "rank_in_camera": entry["rank_in_camera"],
                "camera_count": entry["camera_count"],
                "percentile_score": "{:.10f}".format(
                    entry["percentile_score"]),
                "view_group": entry["view_group"],
            })


@torch.no_grad()
def fuse_pair_features(original_features, enhanced_features, alpha=0.5):
    """Fuse aligned original/enhanced features and preserve exact endpoints."""
    alpha = validate_unit_interval(alpha, "pair_feature_alpha")
    if original_features.shape != enhanced_features.shape:
        raise ValueError(
            "paired feature shapes differ: {} vs {}".format(
                tuple(original_features.shape),
                tuple(enhanced_features.shape)))
    if alpha == 0.0:
        return original_features
    if alpha == 1.0:
        return enhanced_features
    original = F.normalize(original_features, p=2, dim=1)
    enhanced = F.normalize(enhanced_features, p=2, dim=1)
    return F.normalize(
        (1.0 - alpha) * original + alpha * enhanced, p=2, dim=1)


def select_stratified_epoch_paths(samples, pair_map, view_entries,
                                  probability=0.5, seed=1):
    """Select an exact enhanced fraction inside every camera/view stratum."""
    probability = validate_unit_interval(
        probability, "paired_agva_probability")
    strata = collections.defaultdict(list)
    for original_path, _, sample_camera in samples:
        entry = view_entries.get(original_path)
        if entry is None:
            raise KeyError(
                "view entry missing for sample: {}".format(original_path))
        if original_path not in pair_map:
            raise KeyError(
                "enhanced pair missing for sample: {}".format(original_path))
        camera = entry["camera"] or str(sample_camera)
        strata[(camera, entry["view_group"])].append(original_path)

    selected_enhanced = set()
    stratum_stats = {}
    for stratum in sorted(strata):
        paths = sorted(strata[stratum])
        # Independent stratum RNGs make the result stable if another stratum
        # is added or reordered.
        rng = random.Random(
            "{}|{}|{}|{}".format(seed, stratum[0], stratum[1], len(paths)))
        rng.shuffle(paths)
        count = int(math.floor(len(paths) * probability + 0.5))
        chosen = paths[:count]
        selected_enhanced.update(chosen)
        stratum_stats["{}|{}".format(*stratum)] = {
            "total": len(paths),
            "enhanced": count,
            "original": len(paths) - count,
        }

    selected = {
        path: pair_map[path] if path in selected_enhanced else path
        for path, _, _ in samples
    }
    by_group = collections.defaultdict(lambda: {
        "total": 0, "enhanced": 0, "original": 0})
    for path, _, _ in samples:
        group = view_entries[path]["view_group"]
        by_group[group]["total"] += 1
        key = "enhanced" if path in selected_enhanced else "original"
        by_group[group][key] += 1
    stats = {
        "total": len(samples),
        "enhanced": len(selected_enhanced),
        "original": len(samples) - len(selected_enhanced),
        "realized_probability": (
            len(selected_enhanced) / float(len(samples)) if samples else 0.0),
        "by_group": dict(by_group),
        "by_stratum": stratum_stats,
    }
    return selected, stats


def remap_pseudo_dataset(dataset, selected_paths):
    return [
        (selected_paths.get(path, path), label, camera)
        for path, label, camera in dataset
    ]

