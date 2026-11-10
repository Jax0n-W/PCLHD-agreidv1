# -*- coding: utf-8 -*-
"""Controlled AG-ReID.v1 D-AGVA/three-domain ablation launcher.

The release intentionally exposes only a 2x2 experiment matrix. Channel
conversion augmentation is fixed off in both stages for every experiment.
"""
from __future__ import absolute_import, print_function

import argparse
import os.path as osp
import shlex
import subprocess
import sys


PROJECT_ROOT = osp.dirname(osp.abspath(__file__))

EXPERIMENTS = {
    "baseline": {"dynamic_agva": False, "three_domain": False},
    "dynamic": {"dynamic_agva": True, "three_domain": False},
    "three_domain": {"dynamic_agva": False, "three_domain": True},
    "full": {"dynamic_agva": True, "three_domain": True},
}

DEFAULT_DATA_DIR = "/home/lab338/Jaxon/dataset/AGreid_train"
DEFAULT_MASK_DIR = (
    "/home/lab338/Jaxon/dataset/AGreid_train_SCHP/"
    "aerial_modify/1/bounding_box_train"
)
DEFAULT_IMAGE_ROOT = (
    "/home/lab338/Jaxon/dataset/AGreid_train/"
    "aerial_modify/1/bounding_box_train"
)
DEFAULT_VIEW_SCORE = osp.join(
    PROJECT_ROOT, "view_scores", "agreid_aerial_view_scores_foreground.csv"
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "AG-ReID.v1 2x2 ablation: D-AGVA (Stage 1) x "
            "three-domain matching (Stage 2), channel augmentation fixed off"
        )
    )
    parser.add_argument(
        "--experiment", choices=tuple(EXPERIMENTS), default="full")
    parser.add_argument(
        "--run-mode", choices=("full", "stage1", "stage2"), default="full")
    parser.add_argument(
        "--stage1-checkpoint", default="",
        help="Required in stage2 mode; model_best.pth.tar is the default policy")
    parser.add_argument(
        "--checkpoint-policy", choices=("best", "final", "legacy-best"),
        default="best",
        help=("Stage-2 initialization in full mode: best uses the Stage-1 "
              "model_best.pth.tar; final uses checkpoint.pth.tar. "
              "legacy-best is a deprecated alias of best."))

    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--logs-dir", default=osp.join(PROJECT_ROOT, "logs_agreid_dt"))
    parser.add_argument("--stage1-log-name", default="")
    parser.add_argument("--stage2-log-name", default="")
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--iters", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--width", type=int, default=144)
    parser.add_argument("--num-instances", type=int, default=16)
    parser.add_argument("--k1", type=int, default=30)
    parser.add_argument("--k2", type=int, default=6)
    parser.add_argument("--eps", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.00035)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--step-size", type=int, default=20)
    parser.add_argument("--temp", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--print-freq", type=int, default=50)

    parser.add_argument("--dynamic-agva-mask-dir", default=DEFAULT_MASK_DIR)
    parser.add_argument("--dynamic-agva-image-root", default=DEFAULT_IMAGE_ROOT)
    parser.add_argument(
        "--dynamic-agva-slots", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--dynamic-agva-min-displacement", type=float, default=0.030)
    parser.add_argument(
        "--dynamic-agva-max-displacement", type=float, default=0.040)
    parser.add_argument("--dynamic-agva-strict-masks", action="store_true")

    parser.add_argument("--view-score-file", default=DEFAULT_VIEW_SCORE)
    parser.add_argument(
        "--view-score-column", default="p_ground_foreground")
    parser.add_argument(
        "--view-score-path-mode",
        choices=("absolute", "basename", "relative"), default="basename")
    parser.add_argument("--view-near-ratio", type=float, default=0.50)
    parser.add_argument("--far-batch-size", type=int, default=32)
    parser.add_argument("--near-batch-size", type=int, default=32)
    parser.add_argument(
        "--aerial-loss-weight-mode", choices=("dataset", "balanced"),
        default="dataset")
    parser.add_argument("--chain-cross-weight", type=float, default=0.25)
    parser.add_argument("--cluster-n-jobs", type=int, default=1)

    parser.add_argument(
        "--debug-nonfinite", action="store_true", default=True)
    parser.add_argument(
        "--no-debug-nonfinite", action="store_false",
        dest="debug_nonfinite")
    parser.add_argument("--allow-log-overwrite", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate configuration and print commands without executing")
    return parser


def get_experiment_spec(name):
    try:
        return dict(EXPERIMENTS[name])
    except KeyError:
        raise ValueError("Unknown experiment: {}".format(name))


def default_log_names(args, spec=None):
    spec = spec or get_experiment_spec(args.experiment)
    d_flag = int(spec["dynamic_agva"])
    t_flag = int(spec["three_domain"])
    stage1 = args.stage1_log_name or (
        "agreid_dt_s1_d{}_channel_off_seed{}".format(d_flag, args.seed))
    stage2 = args.stage2_log_name or (
        "agreid_dt_{}_d{}_t{}_channel_off_s2_seed{}".format(
            args.experiment, d_flag, t_flag, args.seed))
    return stage1, stage2


def _path(value):
    return osp.abspath(osp.expanduser(value))


def _base_common_args(args, stage1_log_name, stage2_log_name):
    result = [
        "--dataset", "agreid_ir",
        "--trial", str(args.trial),
        "--epochs", str(args.epochs),
        "--iters", str(args.iters),
        "--step-size", str(args.step_size),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--batch-size", str(args.batch_size),
        "--height", str(args.height),
        "--width", str(args.width),
        "--num-instances", str(args.num_instances),
        "--eps", str(args.eps),
        "--k1", str(args.k1),
        "--k2", str(args.k2),
        "--temp", str(args.temp),
        "--momentum", str(args.momentum),
        "--seed", str(args.seed),
        "--print-freq", str(args.print_freq),
        "--pooling-type", "gem",
        "--data-dir", _path(args.data_dir),
        "--logs-dir", _path(args.logs_dir),
        "--stage1-log-name", stage1_log_name,
        "--stage2-log-name", stage2_log_name,
        "--workers", str(args.workers),
        "--channel-adap-gray", "off",
    ]
    if args.debug_nonfinite:
        result.append("--debug-nonfinite")
    if args.allow_log_overwrite:
        result.append("--allow-log-overwrite")
    return result


def _dynamic_args(args):
    result = [
        "--use-dynamic-agva",
        "--dynamic-agva-mask-dir", _path(args.dynamic_agva_mask_dir),
        "--dynamic-agva-image-root", _path(args.dynamic_agva_image_root),
        "--dynamic-agva-slots", str(args.dynamic_agva_slots),
        "--dynamic-agva-min-displacement",
        str(args.dynamic_agva_min_displacement),
        "--dynamic-agva-max-displacement",
        str(args.dynamic_agva_max_displacement),
    ]
    if args.dynamic_agva_strict_masks:
        result.append("--dynamic-agva-strict-masks")
    return result


def build_stage1_command(args, stage1_log_name=None, stage2_log_name=None):
    spec = get_experiment_spec(args.experiment)
    default_s1, default_s2 = default_log_names(args, spec)
    stage1_log_name = stage1_log_name or default_s1
    stage2_log_name = stage2_log_name or default_s2
    command = [sys.executable, osp.join(PROJECT_ROOT, "_train_agreid_engine.py")]
    command += _base_common_args(args, stage1_log_name, stage2_log_name)
    command += ["--stage1-only"]
    if spec["dynamic_agva"]:
        command += _dynamic_args(args)
    return command


def build_stage2_command(args, checkpoint, stage1_log_name=None,
                         stage2_log_name=None):
    spec = get_experiment_spec(args.experiment)
    default_s1, default_s2 = default_log_names(args, spec)
    stage1_log_name = stage1_log_name or default_s1
    stage2_log_name = stage2_log_name or default_s2
    checkpoint = _path(checkpoint)
    if not spec["three_domain"]:
        command = [
            sys.executable, osp.join(PROJECT_ROOT, "_train_agreid_engine.py")]
        command += _base_common_args(args, stage1_log_name, stage2_log_name)
        command += [
            "--stage2-only", "--stage1-checkpoint", checkpoint,
            "--stage2-memorybank", "CMhard", "--no-ema-loss",
        ]
        return command

    command = [
        sys.executable,
        osp.join(PROJECT_ROOT, "_train_agreid_three_domain_engine.py"),
        "--stage2-only",
        "--stage1-checkpoint", checkpoint,
        "--stage2-log-name", stage2_log_name,
        "--dataset", "agreid_ir",
        "--trial", str(args.trial),
        "--epochs", str(args.epochs),
        "--iters", str(args.iters),
        "--step-size", str(args.step_size),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--batch-size", str(args.batch_size),
        "--test-batch", str(args.test_batch),
        "--height", str(args.height),
        "--width", str(args.width),
        "--num-instances", str(args.num_instances),
        "--eps", str(args.eps),
        "--k1", str(args.k1),
        "--k2", str(args.k2),
        "--temp", str(args.temp),
        "--momentum", str(args.momentum),
        "--seed", str(args.seed),
        "--print-freq", str(args.print_freq),
        "--eval-step", "1",
        "--pooling-type", "gem",
        "--data-dir", _path(args.data_dir),
        "--logs-dir", _path(args.logs_dir),
        "--workers", str(args.workers),
        "--view-score-file", _path(args.view_score_file),
        "--view-score-column", args.view_score_column,
        "--view-score-path-mode", args.view_score_path_mode,
        "--view-near-ratio", str(args.view_near_ratio),
        "--far-batch-size", str(args.far_batch_size),
        "--near-batch-size", str(args.near_batch_size),
        "--aerial-loss-weight-mode", args.aerial_loss_weight_mode,
        "--chain-cross-weight", str(args.chain_cross_weight),
        "--cluster-n-jobs", str(args.cluster_n_jobs),
        "--channel-adap-gray", "off",
    ]
    if args.allow_log_overwrite:
        command.append("--allow-log-overwrite")
    return command


def checkpoint_path(args, stage1_log_name):
    filename = ("checkpoint.pth.tar"
                if args.checkpoint_policy == "final"
                else "model_best.pth.tar")
    return osp.join(
        _path(args.logs_dir), stage1_log_name, str(args.trial), filename)


def validate_args(args):
    spec = get_experiment_spec(args.experiment)
    if args.run_mode == "stage2" and not args.stage1_checkpoint:
        raise ValueError("--run-mode stage2 requires --stage1-checkpoint")
    if args.run_mode != "stage2" and args.stage1_checkpoint:
        raise ValueError(
            "--stage1-checkpoint is only valid with --run-mode stage2")
    if args.far_batch_size + args.near_batch_size != args.batch_size:
        raise ValueError(
            "far-batch-size + near-batch-size must equal batch-size")
    if not 0.0 < args.view_near_ratio < 1.0:
        raise ValueError("--view-near-ratio must be in (0, 1)")
    if not (0.0 < args.dynamic_agva_min_displacement <=
            args.dynamic_agva_max_displacement < 0.20):
        raise ValueError("invalid D-AGVA displacement range")
    if args.cluster_n_jobs < 1:
        raise ValueError("--cluster-n-jobs must be at least 1")

    if args.dry_run:
        return
    if not osp.isdir(_path(args.data_dir)):
        raise FileNotFoundError("dataset root not found: {}".format(
            args.data_dir))
    runs_stage1 = args.run_mode in ("full", "stage1")
    if runs_stage1 and spec["dynamic_agva"]:
        if not osp.isdir(_path(args.dynamic_agva_mask_dir)):
            raise FileNotFoundError("D-AGVA mask directory not found: {}".format(
                args.dynamic_agva_mask_dir))
        if not osp.isdir(_path(args.dynamic_agva_image_root)):
            raise FileNotFoundError("D-AGVA image root not found: {}".format(
                args.dynamic_agva_image_root))
    if args.run_mode == "stage2" and not osp.isfile(
            _path(args.stage1_checkpoint)):
        raise FileNotFoundError("Stage 1 checkpoint not found: {}".format(
            args.stage1_checkpoint))
    runs_stage2 = args.run_mode in ("full", "stage2")
    if runs_stage2 and spec["three_domain"] and not osp.isfile(
            _path(args.view_score_file)):
        raise FileNotFoundError("view-score file not found: {}".format(
            args.view_score_file))


def _run(command, dry_run):
    print("\n$ {}".format(shlex.join(command)))
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _print_config(args, spec, stage1_log_name, stage2_log_name):
    print("========== AG-ReID.v1 Controlled Ablation ==========")
    print("Experiment: {}".format(args.experiment))
    print("Run mode: {}".format(args.run_mode))
    print("D-AGVA (Stage 1 only): {}".format(spec["dynamic_agva"]))
    print("Three-domain matching (Stage 2): {}".format(
        spec["three_domain"]))
    print("ChannelAdapGray: False (Stage 1 and Stage 2)")
    print("ChannelExchange/Gray: False (Stage 1 and Stage 2)")
    print("ChannelRandomErasing: True (spatial augmentation)")
    print("Stage 2 memory: CMhard")
    print("Stage 2 EMA loss included: False")
    print("Stage 2 EMA encoder update enabled: True")
    print("Stage 2 ALL-memory optimization enabled: True")
    print("Checkpoint policy: {}".format(args.checkpoint_policy))
    print("Stage 1 log: {}".format(stage1_log_name))
    print("Stage 2 log: {}".format(stage2_log_name))
    print("=====================================================")


def run_experiment(dataset_key="agreid", argv=None):
    if dataset_key != "agreid":
        raise ValueError("This release supports AG-ReID.v1 only")
    args = build_parser().parse_args(argv)
    validate_args(args)
    spec = get_experiment_spec(args.experiment)
    stage1_log_name, stage2_log_name = default_log_names(args, spec)
    _print_config(args, spec, stage1_log_name, stage2_log_name)

    checkpoint = ""
    if args.run_mode in ("full", "stage1"):
        _run(build_stage1_command(
            args, stage1_log_name, stage2_log_name), args.dry_run)
        checkpoint = checkpoint_path(args, stage1_log_name)
        if not args.dry_run and not osp.isfile(checkpoint):
            raise FileNotFoundError(
                "Stage 1 completed but checkpoint was not found: {}".format(
                    checkpoint))
    if args.run_mode == "stage1":
        return 0
    if args.run_mode == "stage2":
        checkpoint = _path(args.stage1_checkpoint)

    _run(build_stage2_command(
        args, checkpoint, stage1_log_name, stage2_log_name), args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(run_experiment("agreid"))
