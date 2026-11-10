#!/usr/bin/env python3
"""Run the controlled AG-ReID.v1 2x2 D-AGVA/three-domain ablation.

Only two Stage-1 jobs are trained (D=0 and D=1); their final checkpoints are
then reused by the four Stage-2 branches.
"""
from __future__ import print_function

import argparse
import os
import os.path as osp
import subprocess
import sys


ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
ENTRY = osp.join(ROOT, "train_agreid.py")
ALL_EXPERIMENTS = ("baseline", "dynamic", "three_domain", "full")
DYNAMIC_EXPERIMENTS = {"dynamic", "full"}


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run channel-off AG-ReID.v1 D/T ablations")
    parser.add_argument("--gpu", default="0")
    parser.add_argument(
        "--experiments", nargs="+", choices=ALL_EXPERIMENTS,
        default=list(ALL_EXPERIMENTS))
    parser.add_argument(
        "--data-dir", default="/home/lab338/Jaxon/dataset/AGreid_train")
    parser.add_argument(
        "--logs-dir", default=osp.join(ROOT, "logs_agreid_dt"))
    parser.add_argument(
        "--mask-dir", default=(
            "/home/lab338/Jaxon/dataset/AGreid_train_SCHP/"
            "aerial_modify/1/bounding_box_train"))
    parser.add_argument(
        "--aerial-image-root", default=(
            "/home/lab338/Jaxon/dataset/AGreid_train/"
            "aerial_modify/1/bounding_box_train"))
    parser.add_argument(
        "--view-score-file",
        default=osp.join(
            ROOT, "view_scores", "agreid_aerial_view_scores_foreground.csv"))
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--iters", type=int, default=400)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--print-freq", type=int, default=50)
    parser.add_argument("--dynamic-agva-strict-masks", action="store_true")
    parser.add_argument("--allow-log-overwrite", action="store_true")
    parser.add_argument("--rerun-stage1", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def common_args(args):
    result = [
        "--data-dir", osp.abspath(osp.expanduser(args.data_dir)),
        "--logs-dir", osp.abspath(osp.expanduser(args.logs_dir)),
        "--trial", str(args.trial),
        "--seed", str(args.seed),
        "--epochs", str(args.epochs),
        "--iters", str(args.iters),
        "--workers", str(args.workers),
        "--print-freq", str(args.print_freq),
        "--checkpoint-policy", "final",
        "--dynamic-agva-mask-dir", osp.abspath(osp.expanduser(args.mask_dir)),
        "--dynamic-agva-image-root",
        osp.abspath(osp.expanduser(args.aerial_image_root)),
        "--view-score-file", osp.abspath(osp.expanduser(args.view_score_file)),
    ]
    if args.dynamic_agva_strict_masks:
        result.append("--dynamic-agva-strict-masks")
    if args.allow_log_overwrite:
        result.append("--allow-log-overwrite")
    if args.dry_run:
        result.append("--dry-run")
    return result


def run(command, env, dry_run):
    print("\n$ {}".format(" ".join(command)))
    if not dry_run:
        subprocess.run(command, cwd=ROOT, env=env, check=True)


def stage1_checkpoint(args, dynamic):
    log_name = "agreid_dt_s1_d{}_channel_off_seed{}".format(
        int(dynamic), args.seed)
    return osp.join(
        osp.abspath(osp.expanduser(args.logs_dir)), log_name,
        str(args.trial), "checkpoint.pth.tar")


def main(argv=None):
    args = build_parser().parse_args(argv)
    selected = list(dict.fromkeys(args.experiments))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    shared = common_args(args)

    needed_d = sorted({name in DYNAMIC_EXPERIMENTS for name in selected})
    for dynamic in needed_d:
        checkpoint = stage1_checkpoint(args, dynamic)
        if osp.isfile(checkpoint) and not args.rerun_stage1 and not args.dry_run:
            print("\nReuse Stage-1 final checkpoint: {}".format(checkpoint))
            continue
        stage1_experiment = "dynamic" if dynamic else "baseline"
        command = [
            sys.executable, ENTRY,
            "--experiment", stage1_experiment,
            "--run-mode", "stage1",
        ] + shared
        run(command, env, args.dry_run)
        if not args.dry_run and not osp.isfile(checkpoint):
            raise FileNotFoundError(
                "Expected Stage-1 final checkpoint not found: {}".format(
                    checkpoint))

    for experiment in selected:
        dynamic = experiment in DYNAMIC_EXPERIMENTS
        checkpoint = stage1_checkpoint(args, dynamic)
        command = [
            sys.executable, ENTRY,
            "--experiment", experiment,
            "--run-mode", "stage2",
            "--stage1-checkpoint", checkpoint,
        ] + shared
        run(command, env, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
