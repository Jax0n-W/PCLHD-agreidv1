# View-domain classifier and aerial OOF scoring

This directory contains the complete prerequisite pipeline used to produce the
aerial view scores consumed by the three-domain Stage 2 experiment. It does
not use person identity labels as classifier targets and does not modify PCLHD
losses, clustering, BGM, trainers, or evaluation.

The binary classifier predicts:

```text
p_ground = P(image has a ground-like viewpoint)
```

Aerial is class 0 and ground is class 1. Training batches contain equal numbers
of aerial and ground images. Camera-disjoint cross-validation prevents a model
from scoring aerial images from a camera that it saw during training. Thus every
published aerial score is an out-of-fold (OOF) prediction.

## Files

- `build_view_manifest.py`: discover images and assign camera-disjoint folds.
- `train_view_classifier.py`: train a balanced ResNet-18 binary classifier.
- `view_classifier.py`: model definition and checkpoint loader.
- `view_dataset.py`: raw/foreground datasets and SCHP foreground rendering.
- `score_aerial_images.py`: score original aerial images with held-out models.
- `score_enhanced_aerial_images.py`: score same-basename enhanced aerial images.
- `analyze_view_scores.py`: score distribution and camera-bias diagnostics.
- `common.py`: filename parsing, manifest I/O, fold validation, and Far/Near ranking.

The implementation supports any contiguous fold count of at least two. The two
released protocols are:

| Dataset | Aerial cameras / folds | Ground validation cameras | Published score |
| --- | --- | --- | --- |
| LAGPeR | `c3,c6,c9,c13` (4 folds) | `c1,c2; c4,c5; c7,c8; c10,c11` | mean of raw and foreground |
| AG-ReID | `c00,c10,c20` (3 folds) | `c03; c13; c23` | foreground only |

## 1. Build manifests

LAGPeR:

```bash
python view_domain/build_view_manifest.py \
  --aerial-dir /home/lab338/Jaxon/dataset/LAGPeR_train/aerial_modify/1/bounding_box_train \
  --ground-dir /home/lab338/Jaxon/dataset/LAGPeR_train/ground_modify/1/bounding_box_train \
  --aerial-cameras c3,c6,c9,c13 \
  --ground-folds 'c1,c2;c4,c5;c7,c8;c10,c11' \
  --output view_domain/outputs/lag/view_manifest.csv
```

AG-ReID:

```bash
python view_domain/build_view_manifest.py \
  --aerial-dir /home/lab338/Jaxon/dataset/AGreid_train/aerial_modify/1/bounding_box_train \
  --ground-dir /home/lab338/Jaxon/dataset/AGreid_train/ground_modify/1/bounding_box_train \
  --aerial-cameras c00,c10,c20 \
  --ground-folds 'c03;c13;c23' \
  --output view_domain/outputs/agreid/view_manifest.csv
```

The camera arguments may be omitted when numeric sorting and equal partitioning
produce exactly these mappings. Keeping them explicit is recommended for a
paper reproduction. The script fails on missing, extra, or duplicate cameras.
It parses both LAGPeR names such as `0001_c3_000002.jpg` and AG-ReID names such
as `P0001_T04041_C00_F121.jpg`.

## 2. Generate SCHP masks for foreground classifiers

Generate masks for both aerial and ground training images. The bundled SCHP
checkpoint has SHA-256:

```text
24fa3254ceeb74c8435458994a64b522fb439a3635b7b86ff470457e0413da00
```

Example for LAGPeR aerial images:

```bash
python generate_schp_masks.py \
  --input-dir /home/lab338/Jaxon/dataset/LAGPeR_train/aerial_modify/1/bounding_box_train \
  --output-dir /home/lab338/Jaxon/dataset/LAGPeR_train_masks/aerial_modify/1/bounding_box_train \
  --checkpoint checkpoints/exp-schp-201908261155-lip.pth \
  --device cuda --seed 1 --skip-existing
```

Run the same command for the ground directory, changing `aerial_modify` to
`ground_modify`. Repeat with the AG-ReID roots when training its classifier.
Nonzero SCHP labels are treated as person foreground; background is replaced by
mid-gray. Missing masks are fatal during foreground classifier training.

## 3. Train LAGPeR classifiers

Raw classifier (C1):

```bash
python view_domain/train_view_classifier.py \
  --manifest view_domain/outputs/lag/view_manifest.csv \
  --all-folds --mode raw --split-mode camera \
  --checkpoint-root view_domain/outputs/lag/checkpoints \
  --batch-size 64 --epochs 30 --seed 1 --amp
```

Foreground classifier (C2):

```bash
python view_domain/train_view_classifier.py \
  --manifest view_domain/outputs/lag/view_manifest.csv \
  --all-folds --mode foreground --split-mode camera \
  --checkpoint-root view_domain/outputs/lag/checkpoints \
  --aerial-source-root /home/lab338/Jaxon/dataset/LAGPeR_train/aerial_modify/1/bounding_box_train \
  --ground-source-root /home/lab338/Jaxon/dataset/LAGPeR_train/ground_modify/1/bounding_box_train \
  --aerial-mask-root /home/lab338/Jaxon/dataset/LAGPeR_train_masks/aerial_modify/1/bounding_box_train \
  --ground-mask-root /home/lab338/Jaxon/dataset/LAGPeR_train_masks/ground_modify/1/bounding_box_train \
  --batch-size 64 --epochs 30 --seed 1 --amp
```

Outputs are stored as:

```text
view_domain/outputs/lag/checkpoints/raw/fold_0/best.pt
view_domain/outputs/lag/checkpoints/foreground/fold_0/best.pt
...
```

## 4. Train the AG-ReID foreground classifier

```bash
python view_domain/train_view_classifier.py \
  --manifest view_domain/outputs/agreid/view_manifest.csv \
  --all-folds --mode foreground --split-mode camera \
  --checkpoint-root view_domain/outputs/agreid/checkpoints \
  --aerial-source-root /home/lab338/Jaxon/dataset/AGreid_train/aerial_modify/1/bounding_box_train \
  --ground-source-root /home/lab338/Jaxon/dataset/AGreid_train/ground_modify/1/bounding_box_train \
  --aerial-mask-root /home/lab338/Jaxon/dataset/AGreid_train_SCHP/aerial_modify/1/bounding_box_train \
  --ground-mask-root /home/lab338/Jaxon/dataset/AGreid_train_SCHP/ground_modify/1/bounding_box_train \
  --batch-size 64 --epochs 30 --seed 1 --amp
```

`--split-mode camera_identity` is available for a stricter protocol that also
removes training images whose PID appears in the validation fold. The command
fails explicitly if that leaves insufficient data; it never silently falls back
to the camera-only split.

## 5. Generate original-image OOF scores

LAGPeR raw + foreground ensemble:

```bash
python view_domain/score_aerial_images.py \
  --manifest view_domain/outputs/lag/view_manifest.csv \
  --raw-checkpoint-dir view_domain/outputs/lag/checkpoints/raw \
  --foreground-checkpoint-dir view_domain/outputs/lag/checkpoints/foreground \
  --aerial-source-root /home/lab338/Jaxon/dataset/LAGPeR_train/aerial_modify/1/bounding_box_train \
  --aerial-mask-root /home/lab338/Jaxon/dataset/LAGPeR_train_masks/aerial_modify/1/bounding_box_train \
  --near-ratio 0.30 --grouping per-camera \
  --output view_domain/outputs/lag/aerial_view_scores_oof.csv
```

AG-ReID foreground-only scores:

```bash
python view_domain/score_aerial_images.py \
  --manifest view_domain/outputs/agreid/view_manifest.csv \
  --foreground-checkpoint-dir view_domain/outputs/agreid/checkpoints/foreground \
  --aerial-source-root /home/lab338/Jaxon/dataset/AGreid_train/aerial_modify/1/bounding_box_train \
  --aerial-mask-root /home/lab338/Jaxon/dataset/AGreid_train_SCHP/aerial_modify/1/bounding_box_train \
  --near-ratio 0.30 --grouping per-camera \
  --output view_domain/outputs/agreid/agreid_aerial_view_scores_foreground.csv
```

For each aerial image, the scorer loads only the checkpoint corresponding to
that image's held-out fold. With both checkpoint sets, `p_ground` is the mean of
`p_ground_raw` and `p_ground_foreground`; with one set, it equals that score.
The CSV includes global and per-camera rank assignments. `near-ratio` changes
only the Far/Near labels and thresholds, not the continuous `p_ground` values.
Stage 2 can therefore apply a different `--view-near-ratio` to the same scores.

## 6. Score enhanced aerial images (optional)

When an enhanced dataset preserves original basenames:

```bash
python view_domain/score_enhanced_aerial_images.py \
  --reference-scores view_domain/outputs/lag/aerial_view_scores_oof.csv \
  --enhanced-aerial-dir /path/to/enhanced/aerial/images \
  --foreground-checkpoint-dir view_domain/outputs/lag/checkpoints/foreground \
  --aerial-mask-root /path/to/enhanced/aerial/masks \
  --near-ratio 0.30 --grouping per-camera \
  --output view_domain/outputs/lag/aerial_view_scores_enhanced_oof.csv
```

## 7. Diagnose score quality and camera dominance

```bash
python view_domain/analyze_view_scores.py \
  --scores view_domain/outputs/lag/aerial_view_scores_oof.csv \
  --output-dir view_domain/outputs/lag/analysis \
  --near-ratio 0.30
```

The analysis writes camera statistics, grouping CSVs, thresholds, a score
histogram, and low/high-score sample grids. Inspect these artifacts before using
the scores: “ground-like viewpoint” is not equivalent to reliable identity
matching, and camera background can otherwise dominate the classifier.

## Reproducibility boundary

The release includes the complete classifier/training/scoring source and the
final score CSVs used by Stage 2. Classifier checkpoints are not bundled because
they are reproducible intermediate artifacts and substantially increase the
archive size. Regenerating them requires the original datasets and GPU training.