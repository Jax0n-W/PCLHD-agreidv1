#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

try:
    from .common import ViewRecord, manifest_folds, read_manifest, seed_everything, write_csv, write_json
    from .view_classifier import ViewClassifier
    from .view_dataset import BalancedDomainBatchSampler, ForegroundRenderer, ViewDataset
except ImportError:
    from common import ViewRecord, manifest_folds, read_manifest, seed_everything, write_csv, write_json  # type: ignore
    from view_classifier import ViewClassifier  # type: ignore
    from view_dataset import BalancedDomainBatchSampler, ForegroundRenderer, ViewDataset  # type: ignore


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(height: int, width: int):
    train_transform = transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.Pad(10, padding_mode="edge"),
            transforms.RandomCrop((height, width)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.35, scale=(0.02, 0.18), ratio=(0.3, 3.3)),
        ]
    )
    validation_transform = transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, validation_transform


def make_renderer(args: argparse.Namespace):
    if args.mode != "foreground":
        return None
    paths = (
        args.aerial_source_root,
        args.ground_source_root,
        args.aerial_mask_root,
        args.ground_mask_root,
    )
    if any(path is None for path in paths):
        raise ValueError(
            "foreground mode requires --aerial-source-root, --ground-source-root, "
            "--aerial-mask-root and --ground-mask-root"
        )
    return ForegroundRenderer(
        source_roots={
            "aerial": args.aerial_source_root,
            "ground": args.ground_source_root,
        },
        mask_roots={
            "aerial": args.aerial_mask_root,
            "ground": args.ground_mask_root,
        },
        background_gray=args.background_gray,
    )


def split_records(
    records: Sequence[ViewRecord], fold: int, split_mode: str
) -> Tuple[List[ViewRecord], List[ViewRecord], dict]:
    validation = [row for row in records if row.fold == fold]
    training = [row for row in records if row.fold != fold]
    validation_pids = {row.pid for row in validation}
    removed_for_identity = 0
    if split_mode == "camera_identity":
        before = len(training)
        training = [row for row in training if row.pid not in validation_pids]
        removed_for_identity = before - len(training)
    if not training or not validation:
        raise RuntimeError(
            "Fold {} has an empty train/validation split after applying {!r}. "
            "Strict identity exclusion can be impossible when every identity appears "
            "in every camera; use --split-mode camera or define identity-disjoint data."
            .format(fold, split_mode)
        )
    train_cameras = {(row.domain, row.camera) for row in training}
    validation_cameras = {(row.domain, row.camera) for row in validation}
    overlap = train_cameras & validation_cameras
    if overlap:
        raise RuntimeError("Camera leakage in fold {}: {}".format(fold, sorted(overlap)))
    if split_mode == "camera_identity":
        train_pids = {row.pid for row in training}
        pid_overlap = train_pids & validation_pids
        if pid_overlap:
            raise RuntimeError("Identity leakage in strict split: {}".format(sorted(pid_overlap)))
    for split_name, split in (("train", training), ("validation", validation)):
        domains = {row.domain for row in split}
        if domains != {"aerial", "ground"}:
            raise RuntimeError(
                "Fold {} {} split does not contain both domains: {}".format(
                    fold, split_name, sorted(domains)
                )
            )
    audit = {
        "fold": fold,
        "split_mode": split_mode,
        "train_images": len(training),
        "validation_images": len(validation),
        "train_pids": len({row.pid for row in training}),
        "validation_pids": len(validation_pids),
        "removed_train_images_for_identity_disjoint": removed_for_identity,
        "train_cameras": sorted("{}:{}".format(*item) for item in train_cameras),
        "validation_cameras": sorted(
            "{}:{}".format(*item) for item in validation_cameras
        ),
    }
    return training, validation, audit


def binary_metrics(targets: List[int], probabilities: List[float]) -> Dict[str, float]:
    predictions = [int(value >= 0.5) for value in probabilities]
    tp = sum(pred == 1 and target == 1 for pred, target in zip(predictions, targets))
    tn = sum(pred == 0 and target == 0 for pred, target in zip(predictions, targets))
    positives = sum(targets)
    negatives = len(targets) - positives
    tpr = tp / positives if positives else float("nan")
    tnr = tn / negatives if negatives else float("nan")
    accuracy = sum(pred == target for pred, target in zip(predictions, targets)) / len(targets)
    try:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(targets, probabilities))
    except (ImportError, ValueError):
        auc = float("nan")
    return {
        "accuracy": accuracy,
        "balanced_accuracy": (tpr + tnr) / 2.0,
        "aerial_accuracy": tnr,
        "ground_accuracy": tpr,
        "auc": auc,
    }


def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    amp: bool,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item()) * images.size(0)
        total_examples += images.size(0)
    return total_loss / total_examples


def evaluate(model, loader, criterion, device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    targets: List[int] = []
    probabilities: List[float] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            batch_targets = batch["target"].to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, batch_targets)
            batch_probabilities = torch.sigmoid(logits)
            total_loss += float(loss.item()) * images.size(0)
            total_examples += images.size(0)
            targets.extend(int(value) for value in batch_targets.cpu().tolist())
            probabilities.extend(float(value) for value in batch_probabilities.cpu().tolist())
    metrics = binary_metrics(targets, probabilities)
    metrics["loss"] = total_loss / total_examples
    return metrics


def save_checkpoint(path: Path, model, optimizer, epoch: int, metrics: dict, metadata: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "metadata": metadata,
        },
        str(path),
    )


def train_fold(
    args: argparse.Namespace,
    records: Sequence[ViewRecord],
    fold: int,
    device: torch.device,
) -> dict:
    fold_seed = args.seed + fold
    seed_everything(fold_seed)
    training, validation, audit = split_records(records, fold, args.split_mode)
    renderer = make_renderer(args)
    train_transform, validation_transform = build_transforms(args.height, args.width)
    train_dataset = ViewDataset(training, train_transform, renderer)
    validation_dataset = ViewDataset(validation, validation_transform, renderer)
    batch_sampler = BalancedDomainBatchSampler(training, args.batch_size, fold_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = ViewClassifier(pretrained=args.pretrained).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs)
    )
    amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    output_dir = args.checkpoint_root / args.mode / "fold_{}".format(fold)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "split_audit.json", audit)
    metadata = {
        "fold": fold,
        "mode": args.mode,
        "split_mode": args.split_mode,
        "height": args.height,
        "width": args.width,
        "background_gray": args.background_gray,
        "manifest": str(args.manifest),
        "fold_count": len(manifest_folds(records)),
    }
    best_score = -float("inf")
    best_epoch = -1
    history = []
    print("\nFold {} | mode={} | {}".format(fold, args.mode, audit))
    for epoch in range(args.epochs):
        started = time.time()
        batch_sampler.set_epoch(epoch)
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, amp
        )
        metrics = evaluate(model, validation_loader, criterion, device)
        scheduler.step()
        selection_score = metrics["auc"]
        if math.isnan(selection_score):
            selection_score = metrics["balanced_accuracy"]
        row = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "validation_loss": metrics["loss"],
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "aerial_accuracy": metrics["aerial_accuracy"],
            "ground_accuracy": metrics["ground_accuracy"],
            "auc": metrics["auc"],
            "seconds": time.time() - started,
        }
        history.append(row)
        write_csv(output_dir / "history.csv", list(row), history)
        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch + 1, metrics, metadata)
        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch + 1
            save_checkpoint(
                output_dir / "best.pt", model, optimizer, epoch + 1, metrics, metadata
            )
        print(
            "epoch {}/{} train_loss={:.4f} val_loss={:.4f} bal_acc={:.4f} "
            "auc={:.4f}".format(
                epoch + 1,
                args.epochs,
                train_loss,
                metrics["loss"],
                metrics["balanced_accuracy"],
                metrics["auc"],
            )
        )
    result = {
        "fold": fold,
        "mode": args.mode,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_checkpoint": str(output_dir / "best.pt"),
    }
    write_json(output_dir / "result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a balanced camera-disjoint view classifier."
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("view_domain/outputs/view_manifest.csv")
    )
    folds = parser.add_mutually_exclusive_group(required=True)
    folds.add_argument("--fold", type=int)
    folds.add_argument("--all-folds", action="store_true")
    parser.add_argument("--mode", choices=("raw", "foreground"), default="raw")
    parser.add_argument(
        "--split-mode", choices=("camera", "camera_identity"), default="camera"
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("view_domain/outputs/checkpoints"),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--width", type=int, default=144)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    pretrained = parser.add_mutually_exclusive_group()
    pretrained.add_argument("--pretrained", dest="pretrained", action="store_true")
    pretrained.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.set_defaults(pretrained=True)
    parser.add_argument("--aerial-source-root", type=Path)
    parser.add_argument("--ground-source-root", type=Path)
    parser.add_argument("--aerial-mask-root", type=Path)
    parser.add_argument("--ground-mask-root", type=Path)
    parser.add_argument("--background-gray", type=int, default=128)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if not 0 <= args.background_gray <= 255:
        parser.error("--background-gray must be in [0, 255]")
    return args


def main() -> None:
    args = parse_args()
    records = read_manifest(args.manifest)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    available_folds = manifest_folds(records)
    if args.all_folds:
        selected_folds = available_folds
    else:
        if args.fold not in available_folds:
            raise ValueError(
                "Requested fold {} is not present in manifest folds {}"
                .format(args.fold, available_folds)
            )
        selected_folds = [args.fold]
    results = [
        train_fold(args, records, int(fold), device) for fold in selected_folds
    ]
    summary_path = args.checkpoint_root / args.mode / "cv_summary.json"
    write_json(
        summary_path,
        {"folds": available_folds, "fold_count": len(available_folds),
         "results": results},
    )
    print("Wrote cross-validation summary to {}".format(summary_path))


if __name__ == "__main__":
    main()

