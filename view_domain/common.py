from __future__ import annotations

import csv
import json
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


DOMAINS = ("aerial", "ground")
REQUIRED_MANIFEST_COLUMNS = ("image_path", "pid", "camera", "domain", "fold")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CAMERA_TOKEN_RE = re.compile(r"^c\d+$", re.IGNORECASE)


@dataclass(frozen=True)
class ViewRecord:
    image_path: str
    pid: str
    camera: str
    domain: str
    fold: int

    @property
    def target(self) -> int:
        return 1 if self.domain == "ground" else 0


def camera_sort_key(camera: str) -> Tuple[int, str]:
    match = re.fullmatch(r"c(\d+)", camera.lower())
    return (int(match.group(1)), camera) if match else (10**9, camera)


def parse_image_name(path: Path) -> Tuple[str, str]:
    tokens = path.stem.split("_")
    cameras = [token.lower() for token in tokens if CAMERA_TOKEN_RE.fullmatch(token)]
    if not tokens[0] or len(cameras) != 1:
        raise ValueError(
            "Cannot uniquely parse pid/camera from {!r}; supported examples are "
            "0001_c3_000002.jpg and P0001_T04041_C00_F121.jpg"
            .format(path.name)
        )
    return tokens[0], cameras[0]


def discover_images(root: Path) -> List[Path]:
    if not root.is_dir():
        raise FileNotFoundError("Image directory does not exist: {}".format(root))
    images = sorted(
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise RuntimeError("No supported images found under {}".format(root))
    return images


def read_manifest(path: Path) -> List[ViewRecord]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(REQUIRED_MANIFEST_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                "Manifest {} is missing columns: {}".format(
                    path, ", ".join(sorted(missing))
                )
            )
        records = []
        for line_number, row in enumerate(reader, start=2):
            domain = row["domain"].strip().lower()
            if domain not in DOMAINS:
                raise ValueError(
                    "Invalid domain {!r} at {}:{}".format(
                        row["domain"], path, line_number
                    )
                )
            records.append(
                ViewRecord(
                    image_path=row["image_path"].strip(),
                    pid=row["pid"].strip(),
                    camera=row["camera"].strip().lower(),
                    domain=domain,
                    fold=int(row["fold"]),
                )
            )
    if not records:
        raise RuntimeError("Manifest is empty: {}".format(path))
    return records


def manifest_folds(records: Sequence[ViewRecord]) -> List[int]:
    folds = sorted({record.fold for record in records})
    expected = list(range(len(folds)))
    if len(folds) < 2 or folds != expected:
        raise ValueError(
            "Manifest folds must be contiguous from 0 and contain at least two "
            "folds; found {}".format(folds)
        )
    return folds

def write_manifest(path: Path, records: Sequence[ViewRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REQUIRED_MANIFEST_COLUMNS))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def add_view_groups(
    rows: Sequence[MutableMapping], ratio: float
) -> Tuple[List[MutableMapping], Dict[str, float]]:
    """Add deterministic global and per-camera top-ratio view groups.

    Ranking, rather than a numeric threshold comparison, guarantees the requested
    group size even when several images have identical scores.
    """
    if not 0.0 < ratio < 1.0:
        raise ValueError("near ratio must be between 0 and 1")
    output = [dict(row) for row in rows]
    for row in output:
        row["view_group_global"] = "A_far"
        row["view_group_per_camera"] = "A_far"

    def mark(indices: Sequence[int], column: str) -> float:
        ranked = sorted(
            indices,
            key=lambda idx: (-float(output[idx]["p_ground"]), output[idx]["image_path"]),
        )
        count = max(1, int(len(ranked) * ratio + 0.999999))
        for idx in ranked[:count]:
            output[idx][column] = "A_near"
        return float(output[ranked[count - 1]]["p_ground"])

    thresholds = {"global": mark(list(range(len(output))), "view_group_global")}
    cameras = sorted({str(row["camera"]) for row in output}, key=camera_sort_key)
    for camera in cameras:
        indices = [idx for idx, row in enumerate(output) if row["camera"] == camera]
        thresholds[camera] = mark(indices, "view_group_per_camera")
    return output, thresholds

