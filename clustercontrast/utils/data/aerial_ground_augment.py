"""
Aerial-to-Ground View Augmentation (AGVA)

A lightweight, identity-preserving geometric augmentation for aerial-ground
person re-identification. It does NOT reconstruct a real frontal view. Instead,
it mildly reduces lower-body foreshortening by moving internal body landmarks
upward while keeping the person's top and bottom anchors fixed.

OpenCV remap convention:
    output(y) = input(map_y(y))
Therefore map_y > y makes an input landmark appear at a smaller output y,
which increases its distance to the fixed bottom anchor.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

ArrayLike = Union[np.ndarray, Image.Image]


@dataclass(frozen=True)
class AGVAInfo:
    applied: bool
    reason: str
    strength: float = 0.0
    gamma: float = 0.0
    bbox: Optional[Tuple[int, int, int, int]] = None  # top, bottom, left, right
    foreground_ratio: float = 0.0
    max_displacement_px: float = 0.0


class AerialGroundViewAugment:
    """Mild aerial-to-ground body proportion augmentation.

    Parameters
    ----------
    p:
        Probability of applying the augmentation to an aerial image.
    strength_range:
        Range of the remap coefficient. For gamma around 2.0--3.0, values
        0.10--0.28 usually produce a maximum displacement of roughly 1--3.5%
        of body height. Start conservatively.
    gamma_range:
        Controls where the deformation peaks. Larger gamma concentrates the
        change lower on the body.
    max_displacement_ratio:
        Hard safety limit for the maximum vertical displacement relative to
        body-box height. The sampled deformation is rescaled if it exceeds it.
    min_box_size:
        Skip masks whose person bounding box is too small.
    min_foreground_ratio / max_foreground_ratio:
        Mask quality safeguards measured inside the whole image.
    keep_largest_component:
        Removes small disconnected SCHP false positives.
    use_distance_weight:
        Reduces deformation near mask boundaries to avoid tearing artifacts.
    debug:
        When True, return ``(image, AGVAInfo)``. Training should keep it False.
    """

    def __init__(
        self,
        p: float = 0.35,
        strength_range: Tuple[float, float] = (0.10, 0.28),
        gamma_range: Tuple[float, float] = (2.0, 3.0),
        max_displacement_ratio: float = 0.035,
        min_box_size: int = 24,
        min_foreground_ratio: float = 0.015,
        max_foreground_ratio: float = 0.90,
        blur_kernel: int = 5,
        morphology_kernel: int = 3,
        keep_largest_component: bool = True,
        use_distance_weight: bool = True,
        debug: bool = False,
    ) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        if strength_range[0] < 0 or strength_range[0] > strength_range[1]:
            raise ValueError(f"invalid strength_range: {strength_range}")
        if gamma_range[0] <= 0 or gamma_range[0] > gamma_range[1]:
            raise ValueError(f"invalid gamma_range: {gamma_range}")
        if not 0.0 < max_displacement_ratio < 0.20:
            raise ValueError("max_displacement_ratio should be in (0, 0.20)")
        if blur_kernel < 1:
            raise ValueError("blur_kernel must be >= 1")

        self.p = float(p)
        self.strength_range = tuple(map(float, strength_range))
        self.gamma_range = tuple(map(float, gamma_range))
        self.max_displacement_ratio = float(max_displacement_ratio)
        self.min_box_size = int(min_box_size)
        self.min_foreground_ratio = float(min_foreground_ratio)
        self.max_foreground_ratio = float(max_foreground_ratio)
        self.blur_kernel = self._odd(blur_kernel)
        self.morphology_kernel = max(1, int(morphology_kernel))
        self.keep_largest_component = bool(keep_largest_component)
        self.use_distance_weight = bool(use_distance_weight)
        self.debug = bool(debug)

    @staticmethod
    def _odd(value: int) -> int:
        value = int(value)
        return value if value % 2 == 1 else value + 1

    def __call__(self, img: ArrayLike, mask: Optional[ArrayLike]):
        original = img
        if random.random() > self.p:
            return self._return(original, AGVAInfo(False, "probability_skip"))
        if mask is None:
            return self._return(original, AGVAInfo(False, "mask_is_none"))

        is_pil = isinstance(img, Image.Image)
        img_np = np.asarray(img).copy() if is_pil else np.asarray(img).copy()
        mask_np = np.asarray(mask)

        if img_np.ndim != 3 or img_np.shape[2] != 3:
            return self._return(original, AGVAInfo(False, "invalid_image_shape"))
        if mask_np.ndim == 3:
            mask_np = mask_np[..., 0]
        if mask_np.shape[:2] != img_np.shape[:2]:
            return self._return(original, AGVAInfo(False, "mask_shape_mismatch"))

        binary = (mask_np > 0).astype(np.uint8)
        binary = self._clean_mask(binary)
        h_img, w_img = binary.shape
        foreground_ratio = float(binary.mean())

        if foreground_ratio < self.min_foreground_ratio:
            info = AGVAInfo(False, "mask_too_small", foreground_ratio=foreground_ratio)
            return self._return(original, info)
        if foreground_ratio > self.max_foreground_ratio:
            info = AGVAInfo(False, "mask_too_large", foreground_ratio=foreground_ratio)
            return self._return(original, info)

        ys, xs = np.where(binary > 0)
        if ys.size == 0:
            return self._return(original, AGVAInfo(False, "empty_mask"))

        top, bottom = int(ys.min()), int(ys.max())
        left, right = int(xs.min()), int(xs.max())
        h_box = bottom - top + 1
        w_box = right - left + 1
        bbox = (top, bottom, left, right)

        if h_box < self.min_box_size or w_box < self.min_box_size:
            info = AGVAInfo(False, "bbox_too_small", bbox=bbox,
                            foreground_ratio=foreground_ratio)
            return self._return(original, info)

        crop_img = img_np[top:bottom + 1, left:right + 1]
        crop_mask = binary[top:bottom + 1, left:right + 1].astype(np.float32)

        strength = random.uniform(*self.strength_range)
        gamma = random.uniform(*self.gamma_range)

        x_coords = np.arange(w_box, dtype=np.float32)
        y_coords = np.arange(h_box, dtype=np.float32)
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)
        yn = y_grid / max(float(h_box - 1), 1.0)

        spatial_weight = self._spatial_weight(crop_mask)
        base_curve = (yn ** gamma) * (1.0 - yn)
        displacement_norm = strength * spatial_weight * base_curve

        # Safety clamp by actual maximum relative displacement.
        current_max = float(displacement_norm.max())
        if current_max > self.max_displacement_ratio:
            displacement_norm *= self.max_displacement_ratio / (current_max + 1e-12)

        # IMPORTANT: cv2.remap is inverse sampling. A positive source offset
        # makes internal landmarks appear upward in the output, expanding the
        # lower-body interval relative to the fixed bottom anchor.
        source_yn = np.clip(yn + displacement_norm, 0.0, 1.0)
        map_y = (source_yn * (h_box - 1.0)).astype(np.float32)
        map_x = x_grid.astype(np.float32)

        warped_crop = cv2.remap(
            crop_img,
            map_x,
            map_y,
            interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT101,
        )

        blend = cv2.GaussianBlur(
            crop_mask,
            (self.blur_kernel, self.blur_kernel),
            0,
        )
        blend = np.clip(blend, 0.0, 1.0)[..., None]

        output = img_np.copy().astype(np.float32)
        mixed = warped_crop.astype(np.float32) * blend + crop_img.astype(np.float32) * (1.0 - blend)
        output[top:bottom + 1, left:right + 1] = mixed
        output = np.clip(output, 0, 255).astype(np.uint8)

        max_disp_px = float(displacement_norm.max() * max(h_box - 1, 1))
        info = AGVAInfo(
            True,
            "applied",
            strength=strength,
            gamma=gamma,
            bbox=bbox,
            foreground_ratio=foreground_ratio,
            max_displacement_px=max_disp_px,
        )
        result = Image.fromarray(output) if is_pil else output
        return self._return(result, info)

    def _clean_mask(self, binary: np.ndarray) -> np.ndarray:
        if self.morphology_kernel > 1:
            kernel = np.ones((self.morphology_kernel, self.morphology_kernel), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        if self.keep_largest_component and np.any(binary):
            count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            if count > 1:
                largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                binary = (labels == largest).astype(np.uint8)
        return binary

    def _spatial_weight(self, crop_mask: np.ndarray) -> np.ndarray:
        if not self.use_distance_weight:
            return crop_mask

        # Smoothly suppress deformation at uncertain mask boundaries while
        # preserving full strength inside the person.
        distance = cv2.distanceTransform(crop_mask.astype(np.uint8), cv2.DIST_L2, 3)
        if float(distance.max()) <= 0:
            return crop_mask
        distance = distance / float(distance.max())
        weight = np.sqrt(np.clip(distance, 0.0, 1.0)) * crop_mask
        weight = cv2.GaussianBlur(weight, (self.blur_kernel, self.blur_kernel), 0)
        return np.clip(weight, 0.0, 1.0)

    def _return(self, image: ArrayLike, info: AGVAInfo):
        return (image, info) if self.debug else image
