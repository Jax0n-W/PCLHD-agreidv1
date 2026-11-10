# -*- coding: utf-8 -*-
"""AG-ReID three-domain Stage 2: A_far <-> A_near <-> Ground.

This is a separate experimental entry.  It does not modify ``train_agreid.py``
or the original trainer.  Stage 1 and the AG-ReID evaluation protocol remain
unchanged.  Stage 2 is based on the no-EMA-loss-only ablation: ``loss_ema`` is
excluded, while the EMA encoder update and the second ALL-memory optimization
are retained.
"""

from __future__ import absolute_import, print_function

import argparse
import os.path as osp
import random
import sys
import time
from datetime import timedelta

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from sklearn.cluster import DBSCAN
from torch.backends import cudnn

from ChannelAug import ChannelRandomErasing
from clustercontrast.agva_three_domain import (
    build_candidate_map, remap_samples_to_enhanced_root, select_epoch_paths,
    select_stratified_epoch_paths)
from clustercontrast.evaluators import extract_features
from clustercontrast.models.cm import ClusterMemory
from clustercontrast.transforms_clean_global_resolution import (
    RandomGlobalResolutionDegradation,
)
from clustercontrast.three_domain import (
    build_bidirectional_bgm, generate_cluster_features,
    generate_joint_cluster_features, generate_random_features,
    joint_cross_modal_flags, split_aerial_trainset)
from clustercontrast.trainers_three_domain import \
    ClusterContrastTrainer_PCLMP_ThreeDomain
from clustercontrast.utils.data import transforms as T
from clustercontrast.utils.logging import Logger
from clustercontrast.utils.serialization import save_checkpoint
from clustercontrast.viewbridge import (VIEW_PATH_MODES, load_view_score_map,
                                        match_view_score_map)


DEFAULT_STAGE2_LOG_NAME = "agreid_three_domain_release_s2"


def get_stage2_output_directory(stage2_log_name, trial, logs_root="logs"):
    name = str(stage2_log_name).strip()
    if not name:
        raise ValueError("--stage2-log-name cannot be empty")
    return osp.normpath(osp.join(logs_root, name, str(trial)))


def check_stage2_output_directory(output_dir, allow_overwrite=False):
    if allow_overwrite or not osp.isdir(output_dir):
        return []
    conflicts = []
    for filename in sorted(os.listdir(output_dir)):
        path = osp.join(output_dir, filename)
        if not osp.isfile(path):
            continue
        lower = filename.lower()
        if (lower in ("checkpoint.pth.tar", "model_best.pth.tar") or
                lower.endswith(".log") or lower.endswith("log.txt")):
            conflicts.append(filename)
    if conflicts:
        raise FileExistsError(
            "Stage 2 output already contains results: {}; conflicts: {}. "
            "Use --allow-log-overwrite only when intentional.".format(
                output_dir, ", ".join(conflicts)))
    return conflicts


# Keep parser/help and CPU configuration tests usable on machines without
# faiss. Formal training still fails clearly through _require_baseline().
try:
    import _train_agreid_engine as baseline
    _BASELINE_IMPORT_ERROR = None
except ModuleNotFoundError as error:
    baseline = None
    _BASELINE_IMPORT_ERROR = error


def _require_baseline():
    if baseline is None:
        raise RuntimeError(
            "formal training requires train_agreid.py and its dependencies; "
            "current missing dependency: {}".format(_BASELINE_IMPORT_ERROR))
    return baseline


def _resolve_project_path(path):
    if not path or osp.isabs(path):
        return path
    return osp.join(osp.dirname(osp.abspath(__file__)), path)


def _resolve_sub_batch_sizes(args):
    if args.far_batch_size is None and args.near_batch_size is None:
        near_batch_size = args.batch_size // 2
        far_batch_size = args.batch_size - near_batch_size
    elif args.far_batch_size is not None and args.near_batch_size is not None:
        far_batch_size = args.far_batch_size
        near_batch_size = args.near_batch_size
    else:
        raise ValueError(
            "--far-batch-size and --near-batch-size must be provided together")
    if far_batch_size + near_batch_size != args.batch_size:
        raise ValueError(
            "far_batch_size + near_batch_size must equal --batch-size so the "
            "total aerial budget matches baseline; got {} + {} != {}".format(
                far_batch_size, near_batch_size, args.batch_size))
    if far_batch_size < args.num_instances or near_batch_size < args.num_instances:
        raise ValueError(
            "Far/Near batch sizes must each be >= --num-instances; got {}/{}/{}".format(
                far_batch_size, near_batch_size, args.num_instances))
    return far_batch_size, near_batch_size


def _validate_args(args):
    if args.channel_adap_gray != "off":
        raise ValueError("Channel augmentation is fixed off in this release")
    if not 0.0 < args.view_near_ratio < 1.0:
        raise ValueError("--view-near-ratio must be in (0, 1)")
    if args.chain_cross_weight < 0.0:
        raise ValueError("--chain-cross-weight must be non-negative")
    if not 0.0 <= args.agva_ir_prob <= 1.0:
        raise ValueError("--agva-ir-prob must be in [0, 1]")
    for name, value in (("--agva-far-probability",
                         args.agva_far_probability),
                        ("--agva-near-probability",
                         args.agva_near_probability)):
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError("{} must be in [0, 1]".format(name))
    if args.use_agva_ir:
        if not osp.isdir(args.agva_orig_root):
            raise FileNotFoundError(
                "AGVA original root not found: {}".format(args.agva_orig_root))
        if not osp.isdir(args.agva_ir_root):
            raise FileNotFoundError(
                "AGVA root not found: {}".format(args.agva_ir_root))
    if args.stage2_aerial_root and not osp.isdir(args.stage2_aerial_root):
        raise FileNotFoundError(
            "Stage 2 enhanced aerial root not found: {}".format(
                args.stage2_aerial_root))
    if not 0.0 <= args.ground_global_downsample_prob <= 1.0:
        raise ValueError(
            "--ground-global-downsample-prob must be in [0, 1]")
    if not (0.0 < args.ground_global_min_scale <=
            args.ground_global_max_scale <= 1.0):
        raise ValueError(
            "global scales must satisfy 0 < min <= max <= 1")
    if args.ground_global_min_low_size < 1:
        raise ValueError(
            "--ground-global-min-low-size must be at least 1")
    if args.cluster_n_jobs < 1:
        raise ValueError("--cluster-n-jobs must be at least 1")
    _resolve_sub_batch_sizes(args)


def _load_trusted_checkpoint(path):
    base = _require_baseline()
    path = osp.abspath(osp.expanduser(path))
    if not osp.isfile(path):
        raise FileNotFoundError("Stage 1 checkpoint not found: {}".format(path))
    try:
        checkpoint = base.load_checkpoint(path)
    except Exception as error:
        if "Weights only load failed" not in str(error):
            raise
        print("PyTorch weights_only compatibility retry for trusted Stage 1 checkpoint")
        try:
            checkpoint = torch.load(
                path, map_location=torch.device("cpu"), weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=torch.device("cpu"))
    print("Stage 1 checkpoint loaded from: {}".format(path))
    return checkpoint


def _prepare_view_groups(args, dataset_ir):
    score_path = _resolve_project_path(args.view_score_file)
    score_map, csv_stats = load_view_score_map(
        score_path, score_column=args.view_score_column,
        near_ratio=args.view_near_ratio, weight_mode="hard_top",
        path_mode=args.view_score_path_mode, relative_root=args.data_dir)
    aerial_train = sorted(dataset_ir.train)
    matched_map, match_stats = match_view_score_map(
        [sample[0] for sample in aerial_train], score_map,
        path_mode=args.view_score_path_mode, relative_root=args.data_dir,
        image_root=dataset_ir.images_dir)
    if match_stats["missing"]:
        raise ValueError(
            "AG-ReID view-score CSV must match every aerial training image; "
            "matched={}, missing={}, first_missing={}".format(
                match_stats["matched"], match_stats["missing"],
                match_stats["missing_paths"][0]))
    far_train, near_train = split_aerial_trainset(aerial_train, matched_map)
    print("========== Three-domain view split ==========")
    print("View score file: {}".format(score_path))
    print("View score column: {}".format(args.view_score_column))
    print("View score path mode: {}".format(args.view_score_path_mode))
    print("Per-camera near ratio: {:.6f}".format(args.view_near_ratio))
    print("CSV rows: {}".format(csv_stats["csv_rows"]))
    print("Matched aerial images: {}".format(match_stats["matched"]))
    print("A_far images: {}".format(len(far_train)))
    print("A_near images: {}".format(len(near_train)))
    print("=============================================")
    return far_train, near_train, matched_map


def _adaptive_cluster(clusterer, distances, name, max_attempts=10):
    attempts = 0
    while attempts < max_attempts:
        labels = clusterer.fit_predict(distances)
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if num_clusters > 0:
            return labels, num_clusters
        clusterer.eps += 0.05
        print("Stage2 {}: no clusters found, increasing eps to {:.3f}".format(
            name, clusterer.eps))
        attempts += 1
    print("WARNING: Stage2 {} clustering failed; assigning cluster 0".format(name))
    return np.zeros(len(distances), dtype=np.int32), 1


def _extract_ordered_features(model, dataset, trainset, args, mode):
    loader = baseline.get_test_loader(
        dataset, args.height, args.width, args.test_batch, args.workers,
        testset=trainset)
    feature_map, _ = extract_features(
        model, loader, print_freq=50, mode=mode)
    return torch.cat([
        feature_map[fname].unsqueeze(0) for fname, _, _ in trainset
    ], dim=0)


def _create_hard_memory(model, args, labels, features, ema_features):
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    centers = generate_cluster_features(labels, features)
    samples = generate_random_features(
        labels, ema_features, num_clusters, args.num_instances)
    memory = ClusterMemory(
        model.module.num_features, num_clusters, temp=args.temp,
        momentum=args.momentum, mode="CMhard", smooth=args.smooth,
        num_instances=args.num_instances).cuda()
    memory.features = F.normalize(centers.repeat(2, 1), dim=1).cuda()
    memory.features_ema = F.normalize(samples, dim=1).cuda()
    return memory, centers, num_clusters


def _build_agva_candidate_map(aerial_train, args):
    if not args.use_agva_ir:
        return {sample[0]: sample[0] for sample in aerial_train}

    def candidate_builder(fname):
        relative_path = osp.relpath(fname, args.agva_orig_root)
        candidates = []
        if (relative_path != ".." and
                not relative_path.startswith(".." + osp.sep)):
            candidates.append(osp.join(args.agva_ir_root, relative_path))
        candidates.extend([
            osp.join(args.agva_ir_root, "aerial_modify", str(args.trial),
                     "bounding_box_train", osp.basename(fname)),
            osp.join(args.agva_ir_root, "aerial_modify",
                     "bounding_box_train", osp.basename(fname)),
        ])
        for candidate in candidates:
            if osp.isfile(candidate):
                return candidate
        return candidates[0]

    candidate_map, missing = build_candidate_map(
        aerial_train, candidate_builder, require_complete=True)
    print("AGVA coverage: {}/{} aerial images".format(
        len(candidate_map) - len(missing), len(candidate_map)))
    return candidate_map


def _map_agva_dataset(samples, labels, selected_paths):
    output = []
    augmented = original = 0
    for (fname, _, cid), raw_label in zip(samples, labels):
        label = int(raw_label)
        if label == -1:
            continue
        selected = selected_paths.get(fname, fname)
        output.append((selected, label, cid))
        augmented += int(selected != fname)
        original += int(selected == fname)
    return output, (augmented, original, 0)


def _build_transforms(args):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ground = T.Compose([
        T.Resize((args.height, args.width), interpolation=3), T.Pad(10),
        T.RandomCrop((args.height, args.width)),
        T.RandomHorizontalFlip(p=0.5), T.ToTensor(), normalizer,
        ChannelRandomErasing(probability=0.5)])
    ground_aug_ops = [
        T.Resize((args.height, args.width), interpolation=3), T.Pad(10),
        T.RandomCrop((args.height, args.width)),
        T.RandomHorizontalFlip(p=0.5),
    ]
    if args.use_ground_clean_global_stage2:
        ground_aug_ops.append(RandomGlobalResolutionDegradation(
            probability=args.ground_global_downsample_prob,
            min_scale=args.ground_global_min_scale,
            max_scale=args.ground_global_max_scale,
            min_low_size=args.ground_global_min_low_size))
    ground_aug_ops.extend([
        T.ToTensor(), normalizer, ChannelRandomErasing(probability=0.5),
    ])
    ground_aug = T.Compose(ground_aug_ops)
    aerial = T.Compose([
        T.Resize((args.height, args.width), interpolation=3), T.Pad(10),
        T.RandomCrop((args.height, args.width)), T.RandomHorizontalFlip(),
        T.ToTensor(), normalizer, ChannelRandomErasing(probability=0.5)])
    return ground, ground_aug, aerial


def _require_nonempty(name, dataset):
    if not dataset:
        raise RuntimeError("{} pseudo-labeled dataset is empty".format(name))


def _evaluate(model_ema, args, data_dir, epoch, best_r1, best_map,
              best_epoch):
    args.test_batch = 64
    args.img_w = args.width
    args.img_h = args.height
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    transform_test = T.Compose([
        T.ToPILImage(), T.Resize((args.img_h, args.img_w)),
        T.ToTensor(), normalize])
    query_img, query_label = baseline.process_test_regdb(
        data_dir, trial=args.trial, modal="visible")
    gallery_img, gallery_label = baseline.process_test_regdb(
        data_dir, trial=args.trial, modal="thermal")
    gallery_set = baseline.TestData(
        gallery_img, gallery_label, transform=transform_test,
        img_size=(args.img_w, args.img_h))
    gallery_loader = data.DataLoader(
        gallery_set, batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers)
    query_set = baseline.TestData(
        query_img, query_label, transform=transform_test,
        img_size=(args.img_w, args.img_h))
    query_loader = data.DataLoader(
        query_set, batch_size=args.test_batch, shuffle=False, num_workers=4)
    query_features = baseline.extract_query_feat(
        model_ema, query_loader, len(query_label))
    gallery_features = baseline.extract_gall_feat(
        model_ema, gallery_loader, len(gallery_label))
    distances = np.matmul(query_features, np.transpose(gallery_features))
    cmc, mean_ap, minp = baseline.eval_regdb(
        -distances, query_label, gallery_label)
    print("Test Trial: {}".format(args.trial))
    print("FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| "
          "Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}".format(
              cmc[0], cmc[4], cmc[9], cmc[19], mean_ap, minp))
    is_best = cmc[0] > best_r1
    if is_best:
        best_r1 = cmc[0]
        best_map = mean_ap
        best_epoch = epoch
    save_checkpoint({
        "state_dict": model_ema.state_dict(), "epoch": epoch + 1,
        "best_mAP": best_map,
    }, is_best, fpath=osp.join(args.logs_dir, "checkpoint.pth.tar"))
    print("\n * Finished epoch {:3d}   model R1: {:5.1%}  model mAP: "
          "{:5.1%}   best R1: {:5.1%}   best mAP: {:5.1%}"
          "(best_epoch:{})\n".format(
              epoch, cmc[0], mean_ap, best_r1, best_map, best_epoch))
    return best_r1, best_map, best_epoch


def main_worker_stage2(args):
    _require_baseline()
    start_time = time.monotonic()
    data_dir = args.data_dir
    args.memorybank = "CMhard"
    args.logs_dir = get_stage2_output_directory(
        args.stage2_log_name, args.trial, logs_root=args.logs_root)
    cudnn.benchmark = True
    sys.stdout = Logger(osp.join(args.logs_dir, "{}log.txt".format(args.trial)))
    print("==========\nArgs:{}\n==========".format(args))
    print("Stage 2 channel conversion augmentation enabled: False")
    print("Stage 2 ChannelRandomErasing enabled: True")

    print("==> Load unlabeled dataset")
    dataset_ir = baseline.get_data(
        "agreid_ir", args.data_dir, trial=args.trial)
    dataset_ground = baseline.get_data(
        "agreid_rgb", args.data_dir, trial=args.trial)
    if args.stage2_aerial_root:
        original_count = len(dataset_ir.train)
        dataset_ir.train = remap_samples_to_enhanced_root(
            dataset_ir.train, args.stage2_aerial_root)
        print("Stage 2 aerial domain: ENHANCED")
        print("Stage 2 enhanced aerial root: {}".format(args.stage2_aerial_root))
        print("Stage 2 enhanced aerial coverage: {}/{}".format(
            len(dataset_ir.train), original_count))
    aerial_train = sorted(dataset_ir.train)
    agva_candidate_map = _build_agva_candidate_map(aerial_train, args)
    ground_train = sorted(dataset_ground.train)
    far_train, near_train, score_map = _prepare_view_groups(args, dataset_ir)
    far_paths = {sample[0] for sample in far_train}
    near_paths = {sample[0] for sample in near_train}
    far_indexes = torch.tensor([
        index for index, sample in enumerate(aerial_train)
        if sample[0] in far_paths], dtype=torch.long)
    near_indexes = torch.tensor([
        index for index, sample in enumerate(aerial_train)
        if sample[0] in near_paths], dtype=torch.long)

    far_batch_size, near_batch_size = _resolve_sub_batch_sizes(args)
    dataset_far_ratio = len(far_train) / float(len(aerial_train))
    far_weight = (dataset_far_ratio
                  if args.aerial_loss_weight_mode == "dataset" else 0.5)
    print("Far batch size: {}".format(far_batch_size))
    print("Near batch size: {}".format(near_batch_size))
    print("Ground batch size: {}".format(args.batch_size))
    print("Aerial samples/iteration: {} (baseline={})".format(
        far_batch_size + near_batch_size, args.batch_size))
    print("Aerial loss weight mode: {}".format(
        args.aerial_loss_weight_mode))
    print("Far/near loss weights: {:.6f}/{:.6f}".format(
        far_weight, 1.0 - far_weight))

    model, model_ema = baseline.create_model(args)
    checkpoint = _load_trusted_checkpoint(args.stage1_checkpoint)
    model.load_state_dict(checkpoint["state_dict"])
    model_ema.load_state_dict(checkpoint["state_dict"])
    optimizer = torch.optim.Adam(
        [{"params": [value]} for _, value in model.named_parameters()
         if value.requires_grad], lr=args.lr,
        weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=0.1)
    trainer = ClusterContrastTrainer_PCLMP_ThreeDomain(
        model, model_ema, far_weight=far_weight,
        chain_cross_weight=args.chain_cross_weight)

    cluster_far = DBSCAN(eps=0.3, min_samples=4,
                         metric="precomputed", n_jobs=args.cluster_n_jobs)
    cluster_near = DBSCAN(eps=0.3, min_samples=4,
                          metric="precomputed", n_jobs=args.cluster_n_jobs)
    cluster_ground = DBSCAN(
        eps=0.3, min_samples=4, metric="precomputed",
        n_jobs=args.cluster_n_jobs)
    cluster_all = DBSCAN(eps=0.3, min_samples=4,
                         metric="precomputed", n_jobs=args.cluster_n_jobs)
    print("Far/Near/Ground/ALL clustering initial eps: 0.300")

    best_r1 = 0.0
    best_map = 0.0
    best_epoch = -1
    iters = args.iters if args.iters > 0 else None

    for epoch in range(args.epochs):
        with torch.no_grad():
            print("==> Extract Ground features")
            features_ground_ema = _extract_ordered_features(
                model_ema, dataset_ground, ground_train, args, mode=1)
            features_ground = _extract_ordered_features(
                model, dataset_ground, ground_train, args, mode=1)
            print("==> Extract Aerial features once, then split Far/Near")
            features_aerial_ema = _extract_ordered_features(
                model_ema, dataset_ir, aerial_train, args, mode=2)
            features_aerial = _extract_ordered_features(
                model, dataset_ir, aerial_train, args, mode=2)
            device = features_aerial.device
            epoch_far_indexes = far_indexes.to(device)
            epoch_near_indexes = near_indexes.to(device)
            features_far = features_aerial.index_select(
                0, epoch_far_indexes)
            features_near = features_aerial.index_select(
                0, epoch_near_indexes)
            features_far_ema = features_aerial_ema.index_select(
                0, epoch_far_indexes)
            features_near_ema = features_aerial_ema.index_select(
                0, epoch_near_indexes)

            distance_far = baseline.compute_jaccard_distance(
                features_far, k1=args.k1, k2=args.k2, search_option=3)
            labels_far, num_far = _adaptive_cluster(
                cluster_far, distance_far, "A_far")
            del distance_far

            distance_near = baseline.compute_jaccard_distance(
                features_near, k1=args.k1, k2=args.k2, search_option=3)
            labels_near, num_near = _adaptive_cluster(
                cluster_near, distance_near, "A_near")
            del distance_near

            distance_ground = baseline.compute_jaccard_distance(
                features_ground, k1=args.k1, k2=args.k2, search_option=3)
            labels_ground, num_ground = _adaptive_cluster(
                cluster_ground, distance_ground, "Ground")
            del distance_ground

            all_train = ground_train + aerial_train
            features_all = torch.cat(
                (features_ground, features_aerial), dim=0)
            distance_all = baseline.compute_modal_invariant_jaccard_distance(
                features_all, k1=40, k2=32, file=all_train,
                search_option=3)
            labels_all, num_all = _adaptive_cluster(
                cluster_all, distance_all, "ALL")
            del distance_all

        memory_far, centers_far, num_far = _create_hard_memory(
            model, args, labels_far, features_far, features_far_ema)
        memory_near, centers_near, num_near = _create_hard_memory(
            model, args, labels_near, features_near, features_near_ema)
        memory_ground, centers_ground, num_ground = _create_hard_memory(
            model, args, labels_ground, features_ground,
            features_ground_ema)
        modality_is_aerial = ([False] * len(ground_train)
                              + [True] * len(aerial_train))
        centers_all = generate_joint_cluster_features(
            labels_all, num_all, features_all, modality_is_aerial)
        memory_all = ClusterMemory(
            model.module.num_features, num_all, temp=args.temp,
            momentum=args.momentum, mode="CMhybrid", smooth=args.smooth,
            num_instances=args.num_instances).cuda()
        memory_all.features = F.normalize(
            centers_all.repeat(2, 1), dim=1).cuda()
        trainer.memory_far = memory_far
        trainer.memory_near = memory_near
        trainer.memory_ground = memory_ground
        trainer.memory_all = memory_all

        print("==> Clusters epoch {}: Far={}, Near={}, Ground={}, ALL={}".format(
            epoch, num_far, num_near, num_ground, num_all))
        print("Start chain Bipartite Graph Matching")
        far_to_near, near_to_far, fn_stats = build_bidirectional_bgm(
            centers_far, centers_near)
        near_to_ground, ground_to_near, ng_stats = build_bidirectional_bgm(
            centers_near, centers_ground)
        print("Far<->Near BGM: {}".format(fn_stats))
        print("Near<->Ground BGM: {}".format(ng_stats))
        print("Direct Far<->Ground BGM: disabled")

        if args.use_agva_ir and args.agva_selection_mode == "camera_group":
            selected_agva_paths, epoch_agva = \
                select_stratified_epoch_paths(
                    aerial_train, agva_candidate_map, far_paths, near_paths,
                    args.agva_ir_prob, args.seed + epoch,
                    group_probabilities={
                        "Far": (args.agva_ir_prob if
                                args.agva_far_probability is None else
                                args.agva_far_probability),
                        "Near": (args.agva_ir_prob if
                                 args.agva_near_probability is None else
                                 args.agva_near_probability),
                    })
        else:
            selected_agva_paths, epoch_agva = select_epoch_paths(
                agva_candidate_map,
                args.agva_ir_prob if args.use_agva_ir else 0.0,
                args.seed + epoch)
        if args.stage2_aerial_root:
            print("Epoch aerial source: fixed enhanced domain ({} images)".format(
                len(selected_agva_paths)))
        else:
            print("Epoch AGVA shared selection: {}".format(epoch_agva))
        pseudo_far, far_agva = _map_agva_dataset(
            far_train, labels_far, selected_agva_paths)
        pseudo_near, near_agva = _map_agva_dataset(
            near_train, labels_near, selected_agva_paths)
        pseudo_ground = [
            (fname, int(label), cid)
            for (fname, _, cid), label in zip(ground_train, labels_ground)
            if int(label) != -1]
        print("Independent AGVA Far augmented/original/fallback: {}".format(
            far_agva))
        print("Independent AGVA Near augmented/original/fallback: {}".format(
            near_agva))

        valid_joint_clusters = joint_cross_modal_flags(
            labels_all, modality_is_aerial)
        print("ALL cross-modal clusters: {}/{}".format(
            len(valid_joint_clusters), num_all))
        joint_far = []
        joint_near = []
        joint_ground = []
        for (fname, _, cid), raw_label in zip(all_train, labels_all):
            label = int(raw_label)
            if label not in valid_joint_clusters:
                continue
            if fname in far_paths:
                selected = selected_agva_paths.get(fname, fname)
                joint_far.append((selected, label, cid))
            elif fname in near_paths:
                selected = selected_agva_paths.get(fname, fname)
                joint_near.append((selected, label, cid))
            else:
                joint_ground.append((fname, label, cid))

        for name, pseudo_set in (
                ("Far", pseudo_far), ("Near", pseudo_near),
                ("Ground", pseudo_ground), ("ALL Far", joint_far),
                ("ALL Near", joint_near),
                ("ALL Ground", joint_ground)):
            _require_nonempty(name, pseudo_set)

        ground_transform, ground_aug_transform, aerial_transform = \
            _build_transforms(args)
        loader_far = baseline.get_train_loader_ir(
            args, dataset_ir, args.height, args.width, far_batch_size,
            args.workers, args.num_instances, iters,
            trainset=pseudo_far, no_cam=args.no_cam,
            train_transformer=aerial_transform)
        loader_near = baseline.get_train_loader_ir(
            args, dataset_ir, args.height, args.width, near_batch_size,
            args.workers, args.num_instances, iters,
            trainset=pseudo_near, no_cam=args.no_cam,
            train_transformer=aerial_transform)
        loader_ground = baseline.get_train_loader_color(
            args, dataset_ground, args.height, args.width, args.batch_size,
            args.workers, args.num_instances, iters,
            trainset=pseudo_ground, no_cam=args.no_cam,
            train_transformer=ground_transform,
            train_transformer1=ground_aug_transform)
        loader_all_far = baseline.get_train_loader_ir(
            args, dataset_ir, args.height, args.width, far_batch_size,
            args.workers, args.num_instances, iters,
            trainset=joint_far, no_cam=args.no_cam,
            train_transformer=aerial_transform)
        loader_all_near = baseline.get_train_loader_ir(
            args, dataset_ir, args.height, args.width, near_batch_size,
            args.workers, args.num_instances, iters,
            trainset=joint_near, no_cam=args.no_cam,
            train_transformer=aerial_transform)
        loader_all_ground = baseline.get_train_loader_color(
            args, dataset_ground, args.height, args.width, args.batch_size,
            args.workers, args.num_instances, iters,
            trainset=joint_ground, no_cam=args.no_cam,
            train_transformer=ground_transform,
            train_transformer1=ground_aug_transform)
        loaders = (loader_far, loader_near, loader_ground,
                   loader_all_far, loader_all_near, loader_all_ground)
        for loader in loaders:
            loader.new_epoch()

        trainer.train(
            epoch, *loaders, optimizer,
            print_freq=args.print_freq, train_iters=len(loader_far),
            far_to_near=far_to_near, near_to_far=near_to_far,
            near_to_ground=near_to_ground,
            ground_to_near=ground_to_near)
        best_r1, best_map, best_epoch = _evaluate(
            model_ema, args, data_dir, epoch,
            best_r1, best_map, best_epoch)
        lr_scheduler.step()

    print("Total running time: {}".format(
        timedelta(seconds=time.monotonic() - start_time)))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Internal AG-ReID three-domain Stage 2 (no EMA loss)")
    parser.add_argument("-d", "--dataset", type=str, default="agreid_ir")
    parser.add_argument("-b", "--batch-size", type=int, default=64)
    parser.add_argument("--test-batch", type=int, default=64)
    parser.add_argument("-j", "--workers", type=int, default=8)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--width", type=int, default=144)
    parser.add_argument("--num-instances", type=int, default=16)
    parser.add_argument("--eps", type=float, default=0.6)
    parser.add_argument("--eps-gap", type=float, default=0.02)
    parser.add_argument("--k1", type=int, default=30)
    parser.add_argument("--k2", type=int, default=6)
    parser.add_argument("-a", "--arch", type=str, default="agw")
    parser.add_argument("--features", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--momentum", type=float, default=0.2)
    parser.add_argument("-mb", "--memorybank", type=str, default="CMhard")
    parser.add_argument("--smooth", type=float, default=0)
    parser.add_argument("--lr", type=float, default=0.00035)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--iters", type=int, default=400)
    parser.add_argument("--step-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--print-freq", type=int, default=10)
    parser.add_argument("--eval-step", type=int, default=1)
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--temp", type=float, default=0.05)
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument("--data-dir", type=str,
                        default=osp.join(working_dir, "data"))
    parser.add_argument("--logs-dir", type=str,
                        default=osp.join(working_dir, "logs"))
    parser.add_argument("--pooling-type", type=str, default="gem")
    parser.add_argument("--use-hard", action="store_true")
    parser.add_argument("--no-cam", action="store_true")
    parser.add_argument("--stage2-only", action="store_true",
                        help="compatibility flag; this entry is Stage 2 only")
    parser.add_argument("--stage1-checkpoint", type=str, required=True)
    parser.add_argument("--stage2-log-name", type=str,
                        default=DEFAULT_STAGE2_LOG_NAME)
    parser.add_argument("--allow-log-overwrite", action="store_true")
    parser.add_argument(
        "--channel-adap-gray", choices=("off",), default="off",
        help="Fixed to off; ChannelAdapGray/ChannelExchange/Gray are disabled")
    parser.add_argument("--view-score-file", type=str, required=True,
                        help="OOF AG-ReID aerial view-score CSV")
    parser.add_argument("--view-score-column", type=str,
                        default="p_ground_foreground")
    parser.add_argument("--view-score-path-mode", choices=VIEW_PATH_MODES,
                        default="basename")
    parser.add_argument("--view-near-ratio", type=float, default=0.50,
                        help="top ratio selected independently per camera")
    parser.add_argument("--far-batch-size", type=int, default=None)
    parser.add_argument("--near-batch-size", type=int, default=None)
    parser.add_argument("--aerial-loss-weight-mode",
                        choices=("dataset", "balanced"), default="dataset")
    parser.add_argument("--chain-cross-weight", type=float, default=0.25)
    parser.add_argument("--cluster-n-jobs", type=int, default=1)
    parser.add_argument(
        "--use-ground-clean-global-stage2", action="store_true")
    parser.add_argument(
        "--ground-global-downsample-prob", type=float, default=0.50)
    parser.add_argument(
        "--ground-global-min-scale", type=float, default=0.60)
    parser.add_argument(
        "--ground-global-max-scale", type=float, default=0.85)
    parser.add_argument(
        "--ground-global-min-low-size", type=int, default=8)
    # Accepted three-domain Stage 2 always uses original aerial images.
    parser.set_defaults(
        use_agva_ir=False,
        agva_orig_root='',
        agva_ir_root='',
        agva_ir_prob=0.0,
        agva_far_probability=None,
        agva_near_probability=None,
        agva_selection_mode='global',
        stage2_aerial_root='')
    return parser


def main(argv=None):
    args = build_parser().parse_args(
        sys.argv[1:] if argv is None else argv)
    _validate_args(args)
    args.logs_root = osp.abspath(osp.expanduser(args.logs_dir))
    output_dir = get_stage2_output_directory(
        args.stage2_log_name, args.trial, logs_root=args.logs_root)
    print("Experiment: THREE-DOMAIN STAGE 2 (NO EMA LOSS)")
    print("Ground Clean-Global Stage 2: {}".format(
        args.use_ground_clean_global_stage2))
    if args.use_ground_clean_global_stage2:
        print("Ground Clean-Global: p={}, scale={}..{}, min_low={}".format(
            args.ground_global_downsample_prob,
            args.ground_global_min_scale,
            args.ground_global_max_scale,
            args.ground_global_min_low_size))
    print("Low-memory clustering: sequential matrices, n_jobs={}".format(
        args.cluster_n_jobs))
    print("Stage 2 AGVA/offline replacement: disabled")
    print("Stage 1: skipped; explicit checkpoint required")
    print("Stage 2 log name: {}".format(args.stage2_log_name))
    print("Stage 2 output directory: {}".format(output_dir))
    check_stage2_output_directory(
        output_dir, allow_overwrite=args.allow_log_overwrite)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    main_worker_stage2(args)


if __name__ == "__main__":
    main()


