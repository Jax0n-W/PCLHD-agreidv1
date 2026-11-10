#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot A_far, A_near and Ground feature distributions.

The script loads a released AGW checkpoint, reconstructs the hard top-q
Far/Near partition from the view-score CSV, extracts normalized training
features, and exports a publication-ready t-SNE or UMAP scatter plot.
"""

from __future__ import absolute_import, print_function

import argparse
import csv
import os
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clustercontrast import datasets, models
from clustercontrast.models.agw_lag_stable import agw_lag_stable
from clustercontrast.viewbridge import load_view_score_map, match_view_score_map


DOMAIN_ORDER = ("A_far", "A_near", "Ground")
DOMAIN_STYLE = {
    "A_far": {"color": "#3B82C4", "marker": "o", "label": r"$A_{\mathrm{far}}$"},
    "A_near": {"color": "#F28E2B", "marker": "^", "label": r"$A_{\mathrm{near}}$"},
    "Ground": {"color": "#59A14F", "marker": "s", "label": "Ground"},
}


class FeatureDataset(Dataset):
    def __init__(self, samples, image_root, transform):
        self.samples = list(samples)
        self.image_root = image_root
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        fname, pid, camid = self.samples[index]
        path = fname
        if not os.path.isabs(path):
            path = os.path.join(self.image_root, path)
        image = Image.open(path).convert("RGB")
        return self.transform(image), fname, int(pid), int(camid)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Plot A_far/A_near/Ground feature distributions")
    parser.add_argument("--dataset", choices=("agreid", "lag"), required=True)
    parser.add_argument("--data-dir", required=True,
                        help="AG-ReID or LAGPeR prepared training root")
    parser.add_argument("--checkpoint", required=True,
                        help="Stage-1 or Stage-2 checkpoint.pth.tar")
    parser.add_argument("--view-score-file", required=True)
    parser.add_argument("--view-score-column", default="p_ground_foreground")
    parser.add_argument("--view-score-path-mode",
                        choices=("absolute", "basename", "relative"),
                        default="basename")
    parser.add_argument("--view-near-ratio", type=float, default=0.50)
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--width", type=int, default=144)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--samples-per-domain", type=int, default=1000,
                        help="Equal upper bound for each domain; <=0 uses all samples")
    parser.add_argument("--method", choices=("tsne", "umap"), default="tsne")
    parser.add_argument("--pca-dim", type=int, default=50)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-flip", action="store_true",
                        help="Disable horizontal-flip test-time feature averaging")
    parser.add_argument("--output-dir", default="three_domain_visualization")
    parser.add_argument("--output-name", default="three_domain_feature_distribution")
    parser.add_argument("--title", default="")
    return parser


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_datasets(dataset_key, data_dir, trial):
    if dataset_key == "agreid":
        aerial_name, ground_name = "agreid_ir", "agreid_rgb"
    else:
        aerial_name, ground_name = "lag_ir", "lag_rgb"
    aerial = datasets.create(aerial_name, data_dir, trial=trial)
    ground = datasets.create(ground_name, data_dir, trial=trial)
    return aerial, ground


def build_model(dataset_key):
    if dataset_key == "lag":
        return agw_lag_stable(
            num_features=0, norm=True, dropout=0, num_classes=0,
            pooling_type="gem")
    return models.create(
        "agw", num_features=0, norm=True, dropout=0, num_classes=0,
        pooling_type="gem")


def load_checkpoint(model, checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu",
                                weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    cleaned = {}
    for key, value in state.items():
        cleaned[key[7:] if key.startswith("module.") else key] = value
    incompatible = model.load_state_dict(cleaned, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint/model mismatch: missing={}, unexpected={}".format(
                incompatible.missing_keys[:10], incompatible.unexpected_keys[:10]))
    print("Loaded checkpoint: {} (epoch={})".format(
        checkpoint_path, checkpoint.get("epoch", "unknown")))


def resolve_groups(aerial_dataset, ground_dataset, args):
    score_map, stats = load_view_score_map(
        args.view_score_file,
        score_column=args.view_score_column,
        near_ratio=args.view_near_ratio,
        weight_mode="hard_top",
        path_mode=args.view_score_path_mode,
        relative_root=args.data_dir)
    aerial_train = sorted(aerial_dataset.train)
    matched, match_stats = match_view_score_map(
        [sample[0] for sample in aerial_train], score_map,
        path_mode=args.view_score_path_mode,
        relative_root=args.data_dir,
        image_root=aerial_dataset.images_dir)
    if match_stats["missing"]:
        raise RuntimeError(
            "View-score CSV misses {} aerial images; first={}".format(
                match_stats["missing"], match_stats["missing_paths"][0]))

    groups = {"A_far": [], "A_near": [], "Ground": sorted(ground_dataset.train)}
    score_by_path = {}
    for sample in aerial_train:
        entry = matched[sample[0]]
        group = entry["view_group"]
        if group not in ("A_far", "A_near"):
            raise RuntimeError("Invalid view group {} for {}".format(group, sample[0]))
        groups[group].append(sample)
        score_by_path[sample[0]] = float(entry["raw_score"])
    print("View split: Far={}, Near={}, Ground={} (CSV rows={})".format(
        len(groups["A_far"]), len(groups["A_near"]), len(groups["Ground"]),
        stats["csv_rows"]))
    return groups, score_by_path


def sample_groups(groups, max_samples, seed):
    rng = np.random.default_rng(seed)
    selected = {}
    for group in DOMAIN_ORDER:
        samples = groups[group]
        count = len(samples) if max_samples <= 0 else min(len(samples), max_samples)
        indexes = np.arange(len(samples))
        if count < len(samples):
            indexes = np.sort(rng.choice(indexes, size=count, replace=False))
        selected[group] = [samples[index] for index in indexes]
        print("Selected {} samples from {}".format(count, group))
    return selected


def extract_group_features(model, samples, image_root, modal, args, device):
    transform = transforms.Compose([
        transforms.Resize((args.height, args.width), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    loader = DataLoader(
        FeatureDataset(samples, image_root, transform),
        batch_size=args.batch_size, num_workers=args.workers,
        shuffle=False, pin_memory=device.type == "cuda")
    features = []
    metadata = []
    model.eval()
    with torch.no_grad():
        for batch_index, (images, paths, pids, camids) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            output = model(images, images, modal=modal)
            if not args.no_flip:
                flipped = torch.flip(images, dims=(3,))
                output = 0.5 * (output + model(flipped, flipped, modal=modal))
            output = F.normalize(output, p=2, dim=1)
            if not torch.isfinite(output).all():
                raise RuntimeError("Non-finite feature at batch {}".format(batch_index))
            features.append(output.cpu().numpy().astype(np.float32))
            metadata.extend(zip(paths, pids.tolist(), camids.tolist()))
            if (batch_index + 1) % 20 == 0:
                print("  extracted {}/{} batches".format(batch_index + 1, len(loader)))
    return np.concatenate(features, axis=0), metadata


def reduce_features(features, args):
    pca_dim = min(args.pca_dim, features.shape[1], features.shape[0] - 1)
    if pca_dim >= 2 and pca_dim < features.shape[1]:
        reduced = PCA(n_components=pca_dim, random_state=args.seed).fit_transform(features)
        print("PCA: {} -> {} dimensions".format(features.shape[1], pca_dim))
    else:
        reduced = features

    if args.method == "tsne":
        max_perplexity = max(1.0, (len(reduced) - 1) / 3.0)
        perplexity = min(args.perplexity, max_perplexity)
        print("Running t-SNE: n={}, perplexity={:.2f}".format(len(reduced), perplexity))
        return TSNE(
            n_components=2, perplexity=perplexity, init="pca",
            learning_rate="auto", random_state=args.seed).fit_transform(reduced)

    try:
        import umap
    except ImportError as error:
        raise RuntimeError(
            "UMAP requested but umap-learn is not installed. Use --method tsne "
            "or install umap-learn.") from error
    print("Running UMAP: n={}".format(len(reduced)))
    return umap.UMAP(
        n_components=2, n_neighbors=30, min_dist=0.15,
        metric="cosine", random_state=args.seed).fit_transform(reduced)


def save_coordinates(path, coordinates, records):
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "domain", "x", "y", "image_path", "pid", "camid", "view_score"))
        writer.writeheader()
        for (group, fname, pid, camid, score), point in zip(records, coordinates):
            writer.writerow({
                "domain": group, "x": "{:.8f}".format(point[0]),
                "y": "{:.8f}".format(point[1]), "image_path": fname,
                "pid": pid, "camid": camid,
                "view_score": "" if score is None else "{:.8f}".format(score),
            })


def plot_distribution(coordinates, records, args, output_base):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    figure, axis = plt.subplots(figsize=(5.4, 4.4))
    groups = np.asarray([record[0] for record in records])
    for group in DOMAIN_ORDER:
        mask = groups == group
        style = DOMAIN_STYLE[group]
        axis.scatter(
            coordinates[mask, 0], coordinates[mask, 1],
            s=16, marker=style["marker"], c=style["color"],
            alpha=0.58, edgecolors="none", label=style["label"],
            rasterized=True)

    axis.set_xlabel("{}-1".format(args.method.upper()))
    axis.set_ylabel("{}-2".format(args.method.upper()))
    if args.title:
        axis.set_title(args.title, pad=8)
    axis.legend(loc="best", frameon=False, markerscale=1.45, ncol=3,
                handletextpad=0.35, columnspacing=0.9)
    axis.tick_params(direction="out", length=3, width=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout(pad=0.5)
    figure.savefig(str(output_base) + ".png", dpi=600, bbox_inches="tight")
    figure.savefig(str(output_base) + ".pdf", dpi=600, bbox_inches="tight")
    plt.close(figure)


def main():
    args = build_parser().parse_args()
    if not 0.0 < args.view_near_ratio < 1.0:
        raise ValueError("--view-near-ratio must be in (0, 1)")
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))

    aerial_dataset, ground_dataset = load_datasets(
        args.dataset, args.data_dir, args.trial)
    groups, scores = resolve_groups(aerial_dataset, ground_dataset, args)
    selected = sample_groups(groups, args.samples_per_domain, args.seed)

    model = build_model(args.dataset)
    load_checkpoint(model, args.checkpoint)
    model.to(device)

    all_features = []
    all_records = []
    for group in DOMAIN_ORDER:
        dataset = aerial_dataset if group != "Ground" else ground_dataset
        modal = 2 if group != "Ground" else 1
        print("Extracting {} features...".format(group))
        features, metadata = extract_group_features(
            model, selected[group], dataset.images_dir, modal, args, device)
        all_features.append(features)
        for fname, pid, camid in metadata:
            all_records.append((
                group, fname, pid, camid,
                scores.get(fname) if group != "Ground" else None))

    features = np.concatenate(all_features, axis=0)
    coordinates = reduce_features(features, args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_base = output_dir / args.output_name
    np.savez_compressed(
        str(output_base) + "_features.npz", features=features,
        domains=np.asarray([record[0] for record in all_records]),
        paths=np.asarray([record[1] for record in all_records]))
    save_coordinates(
        str(output_base) + "_coordinates.csv", coordinates, all_records)
    plot_distribution(coordinates, all_records, args, output_base)
    print("Saved: {}.png".format(output_base))
    print("Saved: {}.pdf".format(output_base))
    print("Saved: {}_coordinates.csv".format(output_base))
    print("Saved: {}_features.npz".format(output_base))


if __name__ == "__main__":
    main()
