# -*- coding: utf-8 -*-
"""Stage 2 Ground 图像的动态非重叠局部分辨率退化。"""
from __future__ import absolute_import, print_function

import random

from PIL import Image


def _validate_probability(value):
    if not 0.0 <= value <= 1.0:
        raise ValueError("probability must be in [0, 1], got {}".format(value))


def _validate_ratio(name, value):
    if not 0.0 < value <= 1.0:
        raise ValueError("{} must be in (0, 1], got {}".format(name, value))


class DynamicNonOverlappingLocalDownsample(object):
    """随机降低多个互不重叠矩形的有效空间分辨率。

    每次调用都重新采样是否执行、矩形数量/位置/尺寸及每块 scale。因此
    ``Preprocessor_color`` 对同一图片生成的两个训练视图会独立采样。
    """

    def __init__(
            self, probability=0.5, min_regions=2, max_regions=4,
            min_width_ratio=0.15, max_width_ratio=0.35,
            min_height_ratio=0.08, max_height_ratio=0.22,
            min_scale=0.25, max_scale=0.60,
            min_area_ratio=0.20, max_area_ratio=0.35,
            region_gap=2, min_patch_size=4,
            max_attempts_per_region=50):
        self.probability = float(probability)
        self.min_regions = int(min_regions)
        self.max_regions = int(max_regions)
        self.min_width_ratio = float(min_width_ratio)
        self.max_width_ratio = float(max_width_ratio)
        self.min_height_ratio = float(min_height_ratio)
        self.max_height_ratio = float(max_height_ratio)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.min_area_ratio = float(min_area_ratio)
        self.max_area_ratio = float(max_area_ratio)
        self.region_gap = int(region_gap)
        self.min_patch_size = int(min_patch_size)
        self.max_attempts_per_region = int(max_attempts_per_region)
        self._validate_parameters()

    def _validate_parameters(self):
        _validate_probability(self.probability)
        if self.min_regions < 1:
            raise ValueError("min_regions must be >= 1")
        if self.max_regions < self.min_regions:
            raise ValueError("max_regions must be >= min_regions")

        _validate_ratio("min_width_ratio", self.min_width_ratio)
        _validate_ratio("max_width_ratio", self.max_width_ratio)
        if self.min_width_ratio > self.max_width_ratio:
            raise ValueError(
                "min_width_ratio must be <= max_width_ratio")

        _validate_ratio("min_height_ratio", self.min_height_ratio)
        _validate_ratio("max_height_ratio", self.max_height_ratio)
        if self.min_height_ratio > self.max_height_ratio:
            raise ValueError(
                "min_height_ratio must be <= max_height_ratio")

        _validate_ratio("min_scale", self.min_scale)
        _validate_ratio("max_scale", self.max_scale)
        if self.min_scale > self.max_scale:
            raise ValueError("min_scale must be <= max_scale")

        _validate_ratio("min_area_ratio", self.min_area_ratio)
        _validate_ratio("max_area_ratio", self.max_area_ratio)
        if self.min_area_ratio > self.max_area_ratio:
            raise ValueError(
                "min_area_ratio must be <= max_area_ratio")
        if self.region_gap < 0:
            raise ValueError("region_gap must be >= 0")
        if self.min_patch_size < 1:
            raise ValueError("min_patch_size must be >= 1")
        if self.max_attempts_per_region < 1:
            raise ValueError("max_attempts_per_region must be >= 1")

    @staticmethod
    def _rectangles_overlap_with_gap(rect_a, rect_b, gap):
        """完整判断右开区间矩形相交，并把最小间隔纳入碰撞范围。"""
        ax1, ay1, ax2, ay2 = rect_a
        bx1, by1, bx2, by2 = rect_b
        separated = (
            ax2 + gap <= bx1 or bx2 + gap <= ax1 or
            ay2 + gap <= by1 or by2 + gap <= ay1)
        return not separated

    def _sample_rectangle(self, image_width, image_height):
        """按当前图像尺寸采样一个边界内矩形，不假设固定输入分辨率。"""
        rectangle_width = int(round(
            image_width * random.uniform(
                self.min_width_ratio, self.max_width_ratio)))
        rectangle_height = int(round(
            image_height * random.uniform(
                self.min_height_ratio, self.max_height_ratio)))
        rectangle_width = min(image_width, max(1, rectangle_width))
        rectangle_height = min(image_height, max(1, rectangle_height))
        x1 = random.randint(0, image_width - rectangle_width)
        y1 = random.randint(0, image_height - rectangle_height)
        return (
            x1, y1, x1 + rectangle_width, y1 + rectangle_height)

    @staticmethod
    def _empty_info(image):
        return {
            "applied": False,
            "requested_regions": 0,
            "accepted_regions": 0,
            "target_area_ratio": 0.0,
            "actual_area_ratio": 0.0,
            "rectangles": [],
            "placement_failures": 0,
            "original_size": image.size,
        }

    def apply_with_info(self, image):
        if not isinstance(image, Image.Image):
            raise TypeError(
                "DynamicNonOverlappingLocalDownsample expects PIL.Image, "
                "got {}".format(type(image).__name__))

        info = self._empty_info(image)
        output = image.copy()
        if random.random() >= self.probability:
            return output, info

        image_width, image_height = image.size
        if image_width < 1 or image_height < 1:
            return output, info

        requested_regions = random.randint(
            self.min_regions, self.max_regions)
        # min/max_area_ratio 采样的是本次面积预算上限；它不保证实际覆盖率
        # 达到 min_area_ratio。actual_area_ratio 才是最终真实覆盖率。
        target_area_ratio = random.uniform(
            self.min_area_ratio, self.max_area_ratio)
        image_area = image_width * image_height
        target_area = target_area_ratio * image_area
        maximum_area = self.max_area_ratio * image_area

        info["requested_regions"] = requested_regions
        info["target_area_ratio"] = target_area_ratio
        accepted_rectangles = []
        current_area = 0

        for _ in range(requested_regions):
            accepted = None
            for _attempt in range(self.max_attempts_per_region):
                rect = self._sample_rectangle(image_width, image_height)
                x1, y1, x2, y2 = rect
                rectangle_area = (x2 - x1) * (y2 - y1)
                if current_area + rectangle_area > target_area:
                    continue
                if current_area + rectangle_area > maximum_area:
                    continue
                if any(self._rectangles_overlap_with_gap(
                        rect, previous, self.region_gap)
                       for previous in accepted_rectangles):
                    continue
                accepted = rect
                break

            if accepted is None:
                info["placement_failures"] += 1
                continue

            x1, y1, x2, y2 = accepted
            patch_width = x2 - x1
            patch_height = y2 - y1
            scale = random.uniform(self.min_scale, self.max_scale)
            low_width = min(
                patch_width,
                max(self.min_patch_size,
                    int(round(patch_width * scale))))
            low_height = min(
                patch_height,
                max(self.min_patch_size,
                    int(round(patch_height * scale))))

            if hasattr(Image, "Resampling"):
                bicubic = Image.Resampling.BICUBIC
                bilinear = Image.Resampling.BILINEAR
            else:
                bicubic = Image.BICUBIC
                bilinear = Image.BILINEAR

            # 始终从未修改原图裁剪，各非重叠区域互不传递增强结果。
            patch = image.crop(accepted)
            low_patch = patch.resize((low_width, low_height), bicubic)
            degraded_patch = low_patch.resize(
                (patch_width, patch_height), bilinear)
            output.paste(degraded_patch, (x1, y1))

            rectangle_area = patch_width * patch_height
            current_area += rectangle_area
            accepted_rectangles.append(accepted)
            info["rectangles"].append({
                "rect": accepted,
                "width": patch_width,
                "height": patch_height,
                "area": rectangle_area,
                "scale": scale,
                "low_size": (low_width, low_height),
            })

        info["accepted_regions"] = len(info["rectangles"])
        info["actual_area_ratio"] = (
            float(current_area) / image_area if image_area else 0.0)
        info["applied"] = info["accepted_regions"] > 0
        return output, info

    def __call__(self, image):
        output, _ = self.apply_with_info(image)
        return output

    def __repr__(self):
        return (
            "{}(probability={}, regions={}-{}, width_ratio={}-{}, "
            "height_ratio={}-{}, scale={}-{}, area_ratio={}-{}, "
            "region_gap={}, min_patch_size={}, "
            "max_attempts_per_region={})".format(
                self.__class__.__name__, self.probability,
                self.min_regions, self.max_regions,
                self.min_width_ratio, self.max_width_ratio,
                self.min_height_ratio, self.max_height_ratio,
                self.min_scale, self.max_scale,
                self.min_area_ratio, self.max_area_ratio,
                self.region_gap, self.min_patch_size,
                self.max_attempts_per_region))
