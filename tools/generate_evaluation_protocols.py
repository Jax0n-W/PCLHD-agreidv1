#!/usr/bin/env python3
"""Generate deterministic AG-ReID and LAGPeR evaluation protocol files.

Two output formats are produced:

1. Explicit query/gallery manifests using ``<relative-path> <label>``.  This
   mirrors the ``test_ground_1.txt`` / ``test_aerial_1.txt`` files used by the
   training engines while avoiding ambiguous direction-dependent names.
2. For LAGPeR, SeCap-compatible path-only files ``exp3_A2G+.txt``,
   ``exp4_G2A+.txt`` and ``exp5_G2A+G.txt``.

The script never reads identity labels from annotations.  It extracts the raw
test identity from the PID directory or filename and creates one deterministic
label map shared by every generated protocol for the dataset.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agreid_identity import (
    AGREID_TEST_IDENTITIES, parse_agreid_name)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class Sample:
    path: Path
    relative_path: str
    raw_pid: int
    top_folder: str


@dataclass(frozen=True)
class Protocol:
    name: str
    query: Tuple[Sample, ...]
    gallery: Tuple[Sample, ...]


DATASET_LAYOUTS = {
    "agreid": {
        "query_aerial": (
            "query_all_aerial", "query_all_c0", "query_aerial"),
        "query_ground": (
            "query_all_ground", "query_all_c3", "query_ground"),
        "gallery_aerial": (
            "bounding_box_test_aerial", "bounding_box_test_all_c0"),
        "gallery_ground": (
            "bounding_box_test_ground", "bounding_box_test_all_c3"),
    },
    "lagper": {
        "query_aerial": ("query_aerial", "query_all_aerial"),
        "query_ground": ("query_ground", "query_all_ground"),
        "gallery_aerial": ("bounding_box_test_aerial",),
        "gallery_ground": ("bounding_box_test_ground",),
        "extra_gallery": ("bounding_box_gallery",),
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate A2G/G2A/G2A+G evaluation TXT files")
    parser.add_argument(
        "--dataset", required=True, choices=("agreid", "lagper"))
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument(
        "--lag-extra-gallery", choices=("auto", "include", "exclude"),
        default="auto",
        help=("How LAGPeR bounding_box_gallery is handled. In auto mode, "
              "existing official exp3/exp4/exp5 files are used to infer its "
              "per-protocol inclusion. Strict mode rejects unresolved cases."))
    parser.add_argument(
        "--strict", action="store_true",
        help="Fail on ambiguous layout, missing positives, or unresolved extra gallery")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Allow replacing generated files in output-dir")
    parser.add_argument(
        "--inspect-only", action="store_true",
        help="Inspect layout and validation statistics without writing files")
    return parser


def find_top_directory(root: Path, candidates: Sequence[str], role: str,
                       required: bool = True) -> Optional[Path]:
    found = [root / name for name in candidates if (root / name).is_dir()]
    if len(found) > 1:
        raise RuntimeError(
            "Ambiguous directories for {}: {}".format(
                role, ", ".join(str(path) for path in found)))
    if not found:
        if required:
            raise FileNotFoundError(
                "No directory found for {}. Tried: {}".format(
                    role, ", ".join(str(root / name) for name in candidates)))
        return None
    return found[0]


def parse_pid_from_component(component: str) -> Optional[int]:
    match = re.fullmatch(r"[Pp]?(-?\d+)", component)
    return int(match.group(1)) if match else None


def parse_pid(path: Path, role_root: Path, dataset: str) -> int:
    if dataset == "agreid":
        return parse_agreid_name(path.name).pid

    # Nested test sets conventionally use one PID directory per identity.
    relative = path.relative_to(role_root)
    for component in reversed(relative.parts[:-1]):
        parsed = parse_pid_from_component(component)
        if parsed is not None:
            return parsed

    stem = path.stem
    patterns = (
        r"^[Pp](-?\d+)(?:[_TtCc]|$)",
        r"^(-?\d+)_[cC]-?\d+",
        r"^(-?\d+)(?:[_-]|$)",
    )
    for pattern in patterns:
        match = re.match(pattern, stem)
        if match:
            return int(match.group(1))
    raise ValueError("Cannot parse PID from image path: {}".format(path))


def collect_samples(root: Path, role_root: Path,
                    dataset: str) -> Tuple[Sample, ...]:
    paths = sorted(
        (path for path in role_root.rglob("*")
         if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: path.relative_to(root).as_posix().lower(),
    )
    if not paths:
        raise RuntimeError("No images found under {}".format(role_root))

    samples = []
    seen_relative = set()
    for path in paths:
        relative = path.relative_to(root).as_posix()
        if relative in seen_relative:
            raise RuntimeError("Duplicate relative image path: {}".format(relative))
        seen_relative.add(relative)
        samples.append(Sample(
            path=path,
            relative_path=relative,
            raw_pid=parse_pid(path, role_root, dataset),
            top_folder=relative.split("/", 1)[0],
        ))
    return tuple(samples)


def merge_samples(*groups: Iterable[Sample]) -> Tuple[Sample, ...]:
    merged: Dict[str, Sample] = {}
    for group in groups:
        for sample in group:
            previous = merged.get(sample.relative_path)
            if previous is not None and previous.raw_pid != sample.raw_pid:
                raise RuntimeError(
                    "Conflicting PID for {}: {} vs {}".format(
                        sample.relative_path, previous.raw_pid, sample.raw_pid))
            merged[sample.relative_path] = sample
    return tuple(merged[key] for key in sorted(merged, key=str.lower))


def locate_existing_protocol(root: Path, filename: str) -> Optional[Path]:
    candidates = (root / filename, root / "idx" / filename)
    found = [path for path in candidates if path.is_file()]
    if len(found) > 1:
        raise RuntimeError(
            "Multiple existing protocol files found for {}: {}".format(
                filename, ", ".join(str(path) for path in found)))
    return found[0] if found else None


def existing_protocol_uses_extra(root: Path, filename: str,
                                 extra_name: str) -> Optional[bool]:
    protocol = locate_existing_protocol(root, filename)
    if protocol is None:
        return None
    extra_prefix = extra_name.rstrip("/") + "/"
    with protocol.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip().replace("\\", "/")
            if stripped and stripped.split()[0].startswith(extra_prefix):
                return True
    return False


def resolve_lag_extra_inclusion(
        args: argparse.Namespace, extra_dir: Optional[Path],
        protocol_filename: str) -> bool:
    if extra_dir is None:
        return False
    if args.lag_extra_gallery == "include":
        return True
    if args.lag_extra_gallery == "exclude":
        return False

    inferred = existing_protocol_uses_extra(
        args.data_root, protocol_filename, extra_dir.name)
    if inferred is not None:
        return inferred
    message = (
        "Found {} but cannot infer whether it belongs to {} because no "
        "existing official protocol file was found".format(
            extra_dir, protocol_filename))
    if args.strict:
        raise RuntimeError(
            message + ". Pass --lag-extra-gallery include or exclude explicitly.")
    print("WARNING: {}. It will be excluded.".format(message), file=sys.stderr)
    return False


def build_protocols(args: argparse.Namespace) -> Tuple[
        Mapping[str, Protocol], Mapping[str, Tuple[Sample, ...]], Dict[str, str]]:
    layout = DATASET_LAYOUTS[args.dataset]
    resolved_dirs: Dict[str, Path] = {}
    for role in ("query_aerial", "query_ground",
                 "gallery_aerial", "gallery_ground"):
        resolved_dirs[role] = find_top_directory(
            args.data_root, layout[role], role, required=True)

    samples = {
        role: collect_samples(args.data_root, directory, args.dataset)
        for role, directory in resolved_dirs.items()
    }

    extra_dir = None
    extra_samples: Tuple[Sample, ...] = tuple()
    if args.dataset == "lagper":
        extra_dir = find_top_directory(
            args.data_root, layout["extra_gallery"], "extra_gallery",
            required=False)
        if extra_dir is not None:
            extra_samples = collect_samples(
                args.data_root, extra_dir, args.dataset)

    extra_by_protocol = {"A2G": False, "G2A": False, "G2AG": False}
    if args.dataset == "lagper" and extra_dir is not None:
        filenames = {
            "A2G": "exp3_A2G+.txt",
            "G2A": "exp4_G2A+.txt",
            "G2AG": "exp5_G2A+G.txt",
        }
        for name, filename in filenames.items():
            extra_by_protocol[name] = resolve_lag_extra_inclusion(
                args, extra_dir, filename)

    a2g_gallery = merge_samples(
        samples["gallery_ground"],
        extra_samples if extra_by_protocol["A2G"] else tuple())
    g2a_gallery = merge_samples(
        samples["gallery_aerial"],
        extra_samples if extra_by_protocol["G2A"] else tuple())

    protocols: Dict[str, Protocol] = {
        "A2G": Protocol(
            "A2G", samples["query_aerial"], a2g_gallery),
        "G2A": Protocol(
            "G2A", samples["query_ground"], g2a_gallery),
    }
    if args.dataset == "lagper":
        g2ag_gallery = merge_samples(
            samples["gallery_aerial"], samples["gallery_ground"],
            extra_samples if extra_by_protocol["G2AG"] else tuple())
        protocols["G2AG"] = Protocol(
            "G2AG", samples["query_ground"], g2ag_gallery)

    directory_report = {
        role: directory.relative_to(args.data_root).as_posix()
        for role, directory in resolved_dirs.items()
    }
    if extra_dir is not None:
        directory_report["extra_gallery"] = extra_dir.relative_to(
            args.data_root).as_posix()
    directory_report.update({
        "extra_in_{}".format(name): str(enabled)
        for name, enabled in extra_by_protocol.items()
    })
    return protocols, samples, directory_report


def build_label_map(protocols: Mapping[str, Protocol]) -> Dict[int, int]:
    raw_pids = sorted({
        sample.raw_pid
        for protocol in protocols.values()
        for sample in protocol.query + protocol.gallery
        if sample.raw_pid >= 0
    })
    return {raw_pid: label for label, raw_pid in enumerate(raw_pids)}


def validate_protocol(protocol: Protocol, strict: bool) -> Dict[str, object]:
    query_paths = [sample.relative_path for sample in protocol.query]
    gallery_paths = [sample.relative_path for sample in protocol.gallery]
    duplicate_query = len(query_paths) - len(set(query_paths))
    duplicate_gallery = len(gallery_paths) - len(set(gallery_paths))
    overlap = sorted(set(query_paths) & set(gallery_paths))

    gallery_pids = {sample.raw_pid for sample in protocol.gallery
                    if sample.raw_pid >= 0}
    query_pids = {sample.raw_pid for sample in protocol.query
                  if sample.raw_pid >= 0}
    missing_positive_pids = sorted(query_pids - gallery_pids)
    query_distractors = sum(sample.raw_pid < 0 for sample in protocol.query)
    gallery_distractors = sum(sample.raw_pid < 0 for sample in protocol.gallery)

    problems = []
    if duplicate_query:
        problems.append("{} duplicate query entries".format(duplicate_query))
    if duplicate_gallery:
        problems.append("{} duplicate gallery entries".format(duplicate_gallery))
    if overlap:
        problems.append("{} query/gallery path overlaps".format(len(overlap)))
    if missing_positive_pids:
        problems.append("{} query PIDs have no gallery positive".format(
            len(missing_positive_pids)))
    if query_distractors:
        problems.append("{} distractors appear in query".format(query_distractors))

    if problems and strict:
        raise RuntimeError(
            "{} validation failed: {}".format(
                protocol.name, "; ".join(problems)))

    return {
        "query_images": len(protocol.query),
        "gallery_images": len(protocol.gallery),
        "query_pids": len(query_pids),
        "gallery_pids": len(gallery_pids),
        "query_distractors": query_distractors,
        "gallery_distractors": gallery_distractors,
        "duplicate_query": duplicate_query,
        "duplicate_gallery": duplicate_gallery,
        "query_gallery_overlap": len(overlap),
        "missing_positive_pids": missing_positive_pids,
        "query_top_folders": sorted({s.top_folder for s in protocol.query}),
        "gallery_top_folders": sorted({s.top_folder for s in protocol.gallery}),
        "warnings": problems,
    }


def indexed_lines(samples: Sequence[Sample],
                  label_map: Mapping[int, int]) -> List[str]:
    return [
        "{} {}".format(
            sample.relative_path,
            -1 if sample.raw_pid < 0 else label_map[sample.raw_pid])
        for sample in samples
    ]


def path_only_lines(protocol: Protocol) -> List[str]:
    return [sample.relative_path for sample in protocol.query + protocol.gallery]


def ensure_writable(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            "Output already exists: {} (pass --overwrite to replace it)".format(path))


def write_lines(path: Path, lines: Sequence[str], overwrite: bool) -> None:
    ensure_writable(path, overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(lines) + "\n"
    path.write_text(payload, encoding="utf-8", newline="\n")


def planned_output_files(args: argparse.Namespace,
                         protocols: Mapping[str, Protocol],
                         label_map: Mapping[int, int]) -> Dict[Path, List[str]]:
    output: Dict[Path, List[str]] = {}
    dataset_folder = "AGReID" if args.dataset == "agreid" else "LAGPeR"
    base = args.output_dir / dataset_folder

    for name, protocol in protocols.items():
        protocol_dir = base / name
        output[protocol_dir / "query_{}_{}.txt".format(
            "aerial" if name == "A2G" else "ground", args.trial)] = indexed_lines(
                protocol.query, label_map)
        gallery_name = {
            "A2G": "gallery_ground",
            "G2A": "gallery_aerial",
            "G2AG": "gallery_aerial_ground",
        }[name]
        output[protocol_dir / "{}_{}.txt".format(
            gallery_name, args.trial)] = indexed_lines(protocol.gallery, label_map)

    # Legacy pair used by the original G->A evaluation engines.
    g2a = protocols["G2A"]
    output[base / "idx" / "test_ground_{}.txt".format(
        args.trial)] = indexed_lines(g2a.query, label_map)
    output[base / "idx" / "test_aerial_{}.txt".format(
        args.trial)] = indexed_lines(g2a.gallery, label_map)

    if args.dataset == "agreid":
        output[base / "idx" / "test_visible_{}.txt".format(
            args.trial)] = indexed_lines(g2a.query, label_map)
        output[base / "idx" / "test_thermal_{}.txt".format(
            args.trial)] = indexed_lines(g2a.gallery, label_map)
    else:
        output[base / "exp3_A2G+.txt"] = path_only_lines(protocols["A2G"])
        output[base / "exp4_G2A+.txt"] = path_only_lines(protocols["G2A"])
        output[base / "exp5_G2A+G.txt"] = path_only_lines(protocols["G2AG"])
    return output


def make_text_report(report: Mapping[str, object]) -> str:
    lines = [
        "Evaluation protocol generation report",
        "dataset: {}".format(report["dataset"]),
        "data_root: {}".format(report["data_root"]),
        "trial: {}".format(report["trial"]),
        "global_nonnegative_pid_count: {}".format(
            report["global_nonnegative_pid_count"]),
        "directories: {}".format(json.dumps(
            report["directories"], ensure_ascii=False, sort_keys=True)),
        "",
    ]
    for name, stats in report["protocols"].items():
        lines.extend([
            "[{}]".format(name),
            "query_images: {}".format(stats["query_images"]),
            "gallery_images: {}".format(stats["gallery_images"]),
            "query_pids: {}".format(stats["query_pids"]),
            "gallery_pids: {}".format(stats["gallery_pids"]),
            "query_distractors: {}".format(stats["query_distractors"]),
            "gallery_distractors: {}".format(stats["gallery_distractors"]),
            "query_top_folders: {}".format(
                ", ".join(stats["query_top_folders"])),
            "gallery_top_folders: {}".format(
                ", ".join(stats["gallery_top_folders"])),
            "missing_positive_pids: {}".format(
                stats["missing_positive_pids"]),
            "warnings: {}".format(stats["warnings"]),
            "",
        ])
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    args.data_root = args.data_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.data_root.is_dir():
        raise NotADirectoryError("Dataset root not found: {}".format(args.data_root))
    if args.trial < 1:
        raise ValueError("--trial must be a positive integer")

    protocols, _, directories = build_protocols(args)
    label_map = build_label_map(protocols)
    if (args.dataset == "agreid" and
            len(label_map) != AGREID_TEST_IDENTITIES):
        raise RuntimeError(
            "AG-ReID test identity count mismatch: expected {}, found {}. "
            "Verify that labels are parsed from P/T/A rather than P alone."
            .format(AGREID_TEST_IDENTITIES, len(label_map)))
    protocol_reports = {
        name: validate_protocol(protocol, strict=args.strict)
        for name, protocol in protocols.items()
    }
    report = {
        "dataset": args.dataset,
        "data_root": str(args.data_root),
        "output_dir": str(args.output_dir),
        "trial": args.trial,
        "global_nonnegative_pid_count": len(label_map),
        "directories": directories,
        "protocols": protocol_reports,
        "label_map": {str(raw_pid): label
                      for raw_pid, label in label_map.items()},
    }

    print(make_text_report(report))
    if args.inspect_only:
        print("Inspect-only mode: no files were written.")
        return 0

    outputs = planned_output_files(args, protocols, label_map)
    report_json = args.output_dir / "protocol_generation_report_{}.json".format(
        args.dataset)
    report_txt = args.output_dir / "protocol_generation_report_{}.txt".format(
        args.dataset)
    for path in list(outputs) + [report_json, report_txt]:
        ensure_writable(path, args.overwrite)

    for path, lines in outputs.items():
        write_lines(path, lines, overwrite=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n")
    report_txt.write_text(
        make_text_report(report) + "\n", encoding="utf-8", newline="\n")

    print("Generated {} protocol files under {}".format(
        len(outputs), args.output_dir))
    for path in sorted(outputs, key=lambda value: str(value).lower()):
        print("  {}".format(path))
    print("  {}".format(report_json))
    print("  {}".format(report_txt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
