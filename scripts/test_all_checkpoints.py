#!/usr/bin/env python3
"""Evaluate the four Stage-2 final checkpoints with detailed run headers."""
from __future__ import print_function

import argparse
import os
import os.path as osp
import subprocess
import sys


ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
TEST_ENTRY = osp.join(ROOT, "examples", "PCLHD_AGReID_g2ag.py")
EXPERIMENTS = {
    "baseline": (0, 0),
    "dynamic": (1, 0),
    "three_domain": (0, 1),
    "full": (1, 1),
}


def parser():
    p = argparse.ArgumentParser(description="Batch-test AG-ReID checkpoints")
    p.add_argument("--gpu", default="0")
    p.add_argument("--test-data-dir", required=True)
    p.add_argument("--logs-dir", required=True)
    p.add_argument("--output-dir", default="")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--trial", type=int, default=1)
    p.add_argument("--checkpoint-name", default="checkpoint.pth.tar")
    p.add_argument(
        "--experiments", nargs="+", choices=tuple(EXPERIMENTS),
        default=list(EXPERIMENTS))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--dry-run", action="store_true")
    return p


def log_name(experiment, seed):
    d_flag, t_flag = EXPERIMENTS[experiment]
    return "agreid_dt_{}_d{}_t{}_channel_off_s2_seed{}".format(
        experiment, d_flag, t_flag, seed)


def main(argv=None):
    args = parser().parse_args(argv)
    logs_dir = osp.abspath(osp.expanduser(args.logs_dir))
    output_dir = args.output_dir or osp.join(logs_dir, "evaluation_logs")
    output_dir = osp.abspath(osp.expanduser(output_dir))
    if not args.dry_run:
        os.makedirs(output_dir, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    for experiment in args.experiments:
        d_flag, t_flag = EXPERIMENTS[experiment]
        checkpoint = osp.join(
            logs_dir, log_name(experiment, args.seed), str(args.trial),
            args.checkpoint_name)
        note = (
            "experiment={} | D-AGVA(Stage1)={} | three-domain(Stage2)={} | "
            "channel augmentation Stage1/Stage2=OFF | checkpoint={}"
        ).format(experiment, bool(d_flag), bool(t_flag), checkpoint)
        command = [
            sys.executable, TEST_ENTRY,
            "--data-dir", osp.abspath(osp.expanduser(args.test_data_dir)),
            "--resume", checkpoint,
            "--gpu", str(args.gpu),
            "--batch-size", str(args.batch_size),
            "--workers", str(args.workers),
        ]
        print("\n" + "=" * 88)
        print("TEST NOTE: " + note)
        print("$ " + " ".join(command))
        print("=" * 88)
        if args.dry_run:
            continue
        if not osp.isfile(checkpoint):
            raise FileNotFoundError(checkpoint)
        log_path = osp.join(output_dir, experiment + ".log")
        with open(log_path, "w", encoding="utf-8") as handle:
            handle.write("TEST NOTE: " + note + "\n")
            handle.flush()
            process = subprocess.Popen(
                command, cwd=ROOT, env=env, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                bufsize=1)
            for line in process.stdout:
                sys.stdout.write(line)
                handle.write(line)
            return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)
        print("Saved evaluation log: {}".format(log_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
