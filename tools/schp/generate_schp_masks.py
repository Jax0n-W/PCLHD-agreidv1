"""Batch-generate binary person masks for AGVA with compatible SCHP inference.

Example (PowerShell):
    python generate_schp_masks.py `
      --input-dir "D:\\AGReID\\train_aerial" `
      --output-dir "D:\\AGReID\\schp_masks" `
      --checkpoint "checkpoints\\exp-schp-201908261155-lip.pth" `
      --device cuda `
      --preview-count 30

Output mirrors the input directory. Every source image gets a PNG mask with
0=background and 255=person. A CSV records quality indicators for auditing.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

from schp_compatible import SCHPParser

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def find_images(root: Path) -> List[Path]:
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def quality_status(foreground_ratio: float, bbox, component_count: int) -> str:
    if bbox is None or foreground_ratio <= 0:
        return "empty"
    if foreground_ratio < 0.015:
        return "too_small"
    if foreground_ratio > 0.90:
        return "too_large"
    top, bottom, left, right = bbox
    if bottom - top + 1 < 24 or right - left + 1 < 16:
        return "bbox_small"
    if component_count > 5:
        return "fragmented"
    return "ok"


def make_preview(samples: List[Tuple[Path, Path, Dict[str, object]]], output: Path) -> None:
    if not samples:
        return
    rows = len(samples)
    fig, axes = plt.subplots(rows, 3, figsize=(10, max(3, rows * 3)))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, (image_path, mask_path, meta) in enumerate(samples):
        image = np.asarray(Image.open(image_path).convert("RGB"))
        mask = np.asarray(Image.open(mask_path).convert("L"))
        overlay = image.copy()
        tint = np.zeros_like(image)
        tint[..., 1] = 255
        alpha = (mask.astype(np.float32) / 255.0 * 0.40)[..., None]
        overlay = np.clip(image * (1.0 - alpha) + tint * alpha, 0, 255).astype(np.uint8)

        axes[row, 0].imshow(image)
        axes[row, 0].set_title(image_path.name)
        axes[row, 1].imshow(mask, cmap="gray", vmin=0, vmax=255)
        axes[row, 1].set_title(
            f"mask fg={float(meta['foreground_ratio']):.3f}\n{meta['status']}"
        )
        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title("overlay")
        for col in range(3):
            axes[row, col].axis("off")
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AGVA binary masks with SCHP")
    parser.add_argument("--input-dir", required=True, help="Aerial training image root")
    parser.add_argument("--output-dir", required=True, help="Output mask root")
    parser.add_argument("--checkpoint", required=True, help="LIP SCHP .pth checkpoint")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0 ...")
    parser.add_argument("--preview-count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-largest-component", action="store_true")
    parser.add_argument("--morphology-kernel", type=int, default=3)
    parser.add_argument("--min-component-area", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    images = find_images(input_dir)
    if not images:
        raise RuntimeError(f"No images found under: {input_dir}")

    print(f"[SCHP] Images: {len(images)}")
    print(f"[SCHP] Loading checkpoint: {args.checkpoint}")
    parser = SCHPParser(
        checkpoint_path=args.checkpoint,
        device=args.device,
        morphology_kernel=args.morphology_kernel,
        keep_largest_component=not args.no_largest_component,
        min_component_area=args.min_component_area,
    )
    report = parser.checkpoint_report
    serious_missing = [k for k in report.missing_keys if not k.endswith("num_batches_tracked")]
    print(f"[SCHP] Device: {parser.device}")
    print(f"[SCHP] Checkpoint tensor keys: {report.parameter_keys}")
    print(f"[SCHP] Serious missing keys: {len(serious_missing)}")
    print(f"[SCHP] Unexpected keys: {len(report.unexpected_keys)}")

    records: List[Dict[str, object]] = []
    preview_candidates: List[Tuple[Path, Path, Dict[str, object]]] = []
    counts: Dict[str, int] = {}

    for image_path in tqdm(images, desc="Generating masks"):
        relative = image_path.relative_to(input_dir)
        mask_path = (output_dir / relative).with_suffix(".png")
        mask_path.parent.mkdir(parents=True, exist_ok=True)

        if args.skip_existing and mask_path.is_file():
            mask_arr = np.asarray(Image.open(mask_path).convert("L"))
            ys, xs = np.where(mask_arr > 0)
            bbox = None if ys.size == 0 else (
                int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
            )
            result_meta = {
                "foreground_ratio": float(np.mean(mask_arr > 0)),
                "bbox": bbox,
                "component_count": -1,
                "touches_border": False,
            }
        else:
            result = parser.parse(image_path)
            Image.fromarray(result.binary_mask, mode="L").save(mask_path)
            result_meta = {
                "foreground_ratio": result.foreground_ratio,
                "bbox": result.bbox,
                "component_count": result.component_count,
                "touches_border": result.touches_border,
            }

        status = quality_status(
            float(result_meta["foreground_ratio"]),
            result_meta["bbox"],
            int(result_meta["component_count"]),
        )
        counts[status] = counts.get(status, 0) + 1
        bbox = result_meta["bbox"]
        record: Dict[str, object] = {
            "image": str(image_path),
            "relative_path": str(relative),
            "mask": str(mask_path),
            "status": status,
            "foreground_ratio": float(result_meta["foreground_ratio"]),
            "bbox_top": "" if bbox is None else bbox[0],
            "bbox_bottom": "" if bbox is None else bbox[1],
            "bbox_left": "" if bbox is None else bbox[2],
            "bbox_right": "" if bbox is None else bbox[3],
            "component_count_before_largest": int(result_meta["component_count"]),
            "touches_border": bool(result_meta["touches_border"]),
        }
        records.append(record)
        preview_candidates.append((image_path, mask_path, record))

    csv_path = output_dir / "schp_mask_report.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    random.seed(args.seed)
    preview_n = min(max(args.preview_count, 0), len(preview_candidates))
    if preview_n:
        # Prioritize suspicious masks, then fill with random OK samples.
        suspicious = [item for item in preview_candidates if item[2]["status"] != "ok"]
        ok_items = [item for item in preview_candidates if item[2]["status"] == "ok"]
        random.shuffle(suspicious)
        random.shuffle(ok_items)
        selected = (suspicious + ok_items)[:preview_n]
        preview_path = output_dir / "schp_mask_preview.jpg"
        make_preview(selected, preview_path)
        print(f"[SCHP] Preview: {preview_path}")

    print(f"[SCHP] Report: {csv_path}")
    print("[SCHP] Quality counts:")
    for key in sorted(counts):
        print(f"  {key:12s}: {counts[key]}")
    ok_ratio = counts.get("ok", 0) / len(images)
    print(f"[SCHP] Masks generated/verified: {len(records)}/{len(images)}")
    print(f"[SCHP] Automatic OK ratio: {ok_ratio:.1%}")
    print("[SCHP] You must still inspect the preview on real aerial images.")


if __name__ == "__main__":
    main()
