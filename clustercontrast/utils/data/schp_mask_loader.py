"""Robust loader for precomputed SCHP binary masks used by AGVA."""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


class SCHPMaskLoader:
    def __init__(
        self,
        mask_dir: Optional[str],
        image_root: Optional[str] = None,
        resize_mismatch: bool = True,
        strict: bool = False,
    ) -> None:
        self.mask_dir = Path(mask_dir).resolve() if mask_dir else None
        self.image_root = Path(image_root).resolve() if image_root else None
        self.resize_mismatch = bool(resize_mismatch)
        self.strict = bool(strict)
        self.by_name: Dict[str, List[Path]] = defaultdict(list)
        self.loaded = 0
        self.missing = 0
        self.ambiguous = 0
        self.resized = 0

        if self.mask_dir is not None:
            if not self.mask_dir.is_dir():
                raise NotADirectoryError(f"SCHP mask directory not found: {self.mask_dir}")
            for path in self.mask_dir.rglob("*.png"):
                self.by_name[path.name.lower()].append(path)
            print(f"[*] SCHP masks indexed: {sum(map(len, self.by_name.values()))}")

    def _relative_candidate(self, img_path: Path) -> Optional[Path]:
        if self.mask_dir is None or self.image_root is None:
            return None
        try:
            relative = img_path.resolve().relative_to(self.image_root)
        except ValueError:
            return None
        return (self.mask_dir / relative).with_suffix(".png")

    def _find(self, img_path: Path) -> Optional[Path]:
        relative = self._relative_candidate(img_path)
        if relative is not None and relative.is_file():
            return relative

        candidates = self.by_name.get(f"{img_path.stem}.png".lower(), [])
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            self.ambiguous += 1
            if self.strict:
                raise RuntimeError(
                    f"Ambiguous mask name for {img_path}: {len(candidates)} candidates. "
                    "Pass image_root so mirrored relative paths can be used."
                )
        return None

    def load(self, img_path: str, img_shape: Optional[Tuple[int, int]] = None) -> Image.Image:
        if self.mask_dir is None:
            return self._dummy(img_shape)
        path = self._find(Path(img_path))
        if path is None:
            self.missing += 1
            if self.strict:
                raise FileNotFoundError(f"No unique SCHP mask found for: {img_path}")
            return self._dummy(img_shape)

        try:
            mask = Image.open(path).convert("L")
        except Exception:
            self.missing += 1
            if self.strict:
                raise
            return self._dummy(img_shape)

        if img_shape is not None:
            h, w = img_shape
            if mask.size != (w, h):
                if not self.resize_mismatch:
                    if self.strict:
                        raise ValueError(
                            f"Mask size {mask.size} != image size {(w, h)} for {img_path}"
                        )
                    return self._dummy(img_shape)
                mask = mask.resize((w, h), Image.Resampling.NEAREST)
                self.resized += 1
        self.loaded += 1
        # Force true binary values even if a palette/grayscale parsing map was supplied.
        arr = (np.asarray(mask) > 0).astype(np.uint8) * 255
        return Image.fromarray(arr, mode="L")

    @staticmethod
    def _dummy(img_shape: Optional[Tuple[int, int]]) -> Image.Image:
        if img_shape is None:
            return Image.fromarray(np.zeros((1, 1), dtype=np.uint8), mode="L")
        h, w = img_shape
        return Image.fromarray(np.zeros((h, w), dtype=np.uint8), mode="L")

    def summary(self) -> str:
        return (
            f"loaded={self.loaded}, missing={self.missing}, "
            f"ambiguous={self.ambiguous}, resized={self.resized}"
        )
