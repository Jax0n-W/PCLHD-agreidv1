#!/usr/bin/env python3
"""Static verification for the channel-off AG-ReID release package."""
from __future__ import print_function

import os
import os.path as osp
import hashlib
import json
import sys


ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
REQUIRED = (
    "agreid_identity.py",
    "train_agreid.py",
    "experiment_runner.py",
    "_train_agreid_engine.py",
    "_train_agreid_three_domain_engine.py",
    "scripts/run_all_ablations.py",
    "scripts/test_all_checkpoints.py",
    "tests/test_release_configuration.py",
    "tests/test_agreid_identity.py",
    "view_scores/agreid_aerial_view_scores_foreground.csv",
)
FORBIDDEN_SUFFIXES = (
    ".pth", ".pth.tar", ".pt", ".ckpt", ".pyc", ".pyo")
FORBIDDEN_CALLS = ("ChannelAdapGray(", "ChannelExchange(", "Gray(")


def fail(message):
    print("[FAIL] " + message)
    return False


def main():
    ok = True
    for relative in REQUIRED:
        if not osp.isfile(osp.join(ROOT, relative)):
            ok = fail("missing required file: {}".format(relative)) and ok

    for filename in (
            "_train_agreid_engine.py",
            "_train_agreid_three_domain_engine.py"):
        path = osp.join(ROOT, filename)
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()
        for call in FORBIDDEN_CALLS:
            if call in source:
                ok = fail("active channel conversion call in {}: {}".format(
                    filename, call)) and ok
        if "ChannelRandomErasing(" not in source:
            ok = fail("spatial random erasing missing from {}".format(
                filename)) and ok

    identity_consumers = {
        "_train_agreid_engine.py": "parse_agreid_name",
        "clustercontrast/datasets/agreid_ir.py": "parse_agreid_name",
        "clustercontrast/datasets/agreid_rgb.py": "parse_agreid_name",
        "examples/PCLHD_AGReID_g2ag.py": "parse_agreid_name",
        "tools/generate_evaluation_protocols.py": "parse_agreid_name",
    }
    for relative, symbol in identity_consumers.items():
        with open(osp.join(ROOT, *relative.split("/")), "r",
                  encoding="utf-8") as handle:
            if symbol not in handle.read():
                ok = fail("official identity parser missing from {}".format(
                    relative)) and ok

    artifacts = []
    for directory, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d != ".git"]
        for name in filenames:
            lower = name.lower()
            if any(lower.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES):
                artifacts.append(osp.relpath(osp.join(directory, name), ROOT))
        if "__pycache__" in dirnames:
            artifacts.append(osp.relpath(
                osp.join(directory, "__pycache__"), ROOT) + os.sep)
    if artifacts:
        ok = fail("forbidden release artifacts: {}".format(
            ", ".join(sorted(artifacts)))) and ok

    info_path = osp.join(ROOT, "RELEASE_INFO.json")
    with open(info_path, "r", encoding="utf-8") as handle:
        info = json.load(handle)
    fixed = info["fixed_settings"]
    for key in (
            "channel_adap_gray_stage1", "channel_adap_gray_stage2",
            "channel_exchange_stage1", "channel_exchange_stage2"):
        if fixed[key] is not False:
            ok = fail("RELEASE_INFO does not fix {} to false".format(key)) and ok

    manifest_path = osp.join(ROOT, "MANIFEST_SHA256.txt")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest_lines = [line.strip() for line in handle if line.strip()]
    for line in manifest_lines:
        expected, relative = line.split(None, 1)
        path = osp.join(ROOT, *relative.replace("\\", "/").split("/"))
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            ok = fail("SHA256 mismatch: {}".format(relative)) and ok

    if ok:
        print("RELEASE_VERIFICATION_PASSED")
        print("Channel conversion augmentation: OFF in both stages")
        print("Controlled experiments: baseline, dynamic, three_domain, full")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
