# -*- coding: utf-8 -*-
"""Clean-anchor global resolution degradation for Ground ReID views."""
from __future__ import absolute_import, print_function

import random

from PIL import Image


class RandomGlobalResolutionDegradation(object):
    """Downsample and restore the full PIL image with bounded probability."""

    def __init__(self, probability=0.5, min_scale=0.50, max_scale=0.80,
                 min_low_size=8):
        self.probability = float(probability)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.min_low_size = int(min_low_size)
        self._validate_parameters()

    def _validate_parameters(self):
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be in [0, 1]")
        if not 0.0 < self.min_scale <= self.max_scale <= 1.0:
            raise ValueError(
                "scales must satisfy 0 < min_scale <= max_scale <= 1")
        if self.min_low_size < 1:
            raise ValueError("min_low_size must be >= 1")

    @staticmethod
    def _resampling_modes():
        if hasattr(Image, "Resampling"):
            return Image.Resampling.BICUBIC, Image.Resampling.BILINEAR
        return Image.BICUBIC, Image.BILINEAR

    @staticmethod
    def _empty_info(image):
        return {
            "applied": False,
            "scale": 1.0,
            "original_size": image.size,
            "low_size": image.size,
        }

    def apply_with_info(self, image):
        if not isinstance(image, Image.Image):
            raise TypeError(
                "RandomGlobalResolutionDegradation expects PIL.Image, "
                "got {}".format(type(image).__name__))

        output = image.copy()
        info = self._empty_info(image)
        if random.random() >= self.probability:
            return output, info

        width, height = image.size
        if width < 1 or height < 1:
            return output, info

        scale = random.uniform(self.min_scale, self.max_scale)
        low_width = min(
            width, max(self.min_low_size, int(round(width * scale))))
        low_height = min(
            height, max(self.min_low_size, int(round(height * scale))))
        info["scale"] = scale
        info["low_size"] = (low_width, low_height)

        if (low_width, low_height) == (width, height):
            return output, info

        down_mode, up_mode = self._resampling_modes()
        low_image = image.resize((low_width, low_height), down_mode)
        output = low_image.resize((width, height), up_mode)
        info["applied"] = True
        return output, info

    def __call__(self, image):
        output, _ = self.apply_with_info(image)
        return output

    def __repr__(self):
        return (
            "{}(probability={}, scale={}-{}, min_low_size={})"
            .format(self.__class__.__name__, self.probability,
                    self.min_scale, self.max_scale, self.min_low_size))
