"""Dynamic, online AGVA process for aerial training images."""

from __future__ import absolute_import

import os.path as osp
import random

from .aerial_ground_augment import AGVAInfo, AerialGroundViewAugment
from .schp_mask_loader import SCHPMaskLoader


CHANNEL_ADAP_GRAY_MODES = ("off",)


def channel_adap_gray_enabled(args):
    """Return False: channel conversion is fixed off in this release."""
    mode = getattr(args, "channel_adap_gray", "off")
    if mode not in CHANNEL_ADAP_GRAY_MODES:
        raise ValueError(
            "channel_adap_gray is fixed to 'off' in this release; got {}".format(
                mode))
    return False


class DynamicAGVAProcess(object):
    """Apply AGVA when a random integer in [1, 3] is selected.

    ``apply_slots=1`` means 1/3 probability and ``apply_slots=2`` means 2/3.
    The maximum body-height displacement is sampled independently for every
    applied image, centered on the original validated value of 0.035.
    """

    def __init__(self, mask_dir, image_root=None, apply_slots=1,
                 displacement_ratio_range=(0.030, 0.040),
                 strict_masks=False, debug=False):
        if apply_slots not in (1, 2):
            raise ValueError("apply_slots must be 1 or 2, got {}".format(apply_slots))
        lower, upper = map(float, displacement_ratio_range)
        if not (0.0 < lower <= upper < 0.20):
            raise ValueError(
                "displacement_ratio_range must be ordered and inside (0, 0.20)"
            )
        self.apply_slots = int(apply_slots)
        self.displacement_ratio_range = (lower, upper)
        self.debug = bool(debug)
        self.mask_loader = SCHPMaskLoader(
            mask_dir=mask_dir,
            image_root=image_root,
            resize_mismatch=True,
            strict=strict_masks,
        )
        self.augment = AerialGroundViewAugment(p=1.0, debug=debug)

    @property
    def probability(self):
        return self.apply_slots / 3.0

    def __call__(self, image, image_path):
        draw = random.randint(1, 3)
        if draw > self.apply_slots:
            if self.debug:
                return image, AGVAInfo(False, "probability_skip")
            return image

        # A small per-image fluctuation around the original 0.035 cap.
        self.augment.max_displacement_ratio = random.uniform(
            *self.displacement_ratio_range
        )
        mask = self.mask_loader.load(
            image_path, img_shape=(image.height, image.width)
        )
        return self.augment(image, mask)


def get_dynamic_agva_process(args, image_root):
    """Build once per training program and reuse in all aerial loss loaders."""
    if not getattr(args, "use_dynamic_agva", False):
        return None

    cached = getattr(args, "_dynamic_agva_process", None)
    cached_root = getattr(args, "_dynamic_agva_image_root", None)
    effective_root = getattr(args, "dynamic_agva_image_root", "") or image_root
    effective_root = osp.abspath(osp.expanduser(effective_root))
    if cached is not None and cached_root == effective_root:
        return cached

    process = DynamicAGVAProcess(
        mask_dir=args.dynamic_agva_mask_dir,
        image_root=effective_root,
        apply_slots=args.dynamic_agva_slots,
        displacement_ratio_range=(
            args.dynamic_agva_min_displacement,
            args.dynamic_agva_max_displacement,
        ),
        strict_masks=args.dynamic_agva_strict_masks,
    )
    args._dynamic_agva_process = process
    args._dynamic_agva_image_root = effective_root
    return process


def validate_dynamic_agva_args(args):
    if not getattr(args, "use_dynamic_agva", False):
        return
    if getattr(args, "use_agva_ir", False):
        raise ValueError(
            "--use-dynamic-agva and offline --use-agva-ir cannot be enabled together"
        )
    if args.dynamic_agva_slots not in (1, 2):
        raise ValueError("--dynamic-agva-slots must be 1 or 2")
    lower = args.dynamic_agva_min_displacement
    upper = args.dynamic_agva_max_displacement
    if not (0.0 < lower <= upper < 0.20):
        raise ValueError(
            "dynamic AGVA displacement range must be ordered and inside (0, 0.20)"
        )
    if not osp.isdir(args.dynamic_agva_mask_dir):
        raise FileNotFoundError(
            "Dynamic AGVA SCHP mask directory not found: {}".format(
                args.dynamic_agva_mask_dir
            )
        )


def print_dynamic_agva_config(args):
    enabled = getattr(args, "use_dynamic_agva", False)
    print("========== Dynamic AGVA Configuration ==========")
    print("Enabled:", enabled)
    if enabled:
        print("Mask directory:", args.dynamic_agva_mask_dir)
        print("Image root:", args.dynamic_agva_image_root or "dataset image root")
        print("Apply slots: {}/3".format(args.dynamic_agva_slots))
        print("Apply probability: {:.6f}".format(args.dynamic_agva_slots / 3.0))
        print(
            "Max displacement ratio range: [{:.4f}, {:.4f}]".format(
                args.dynamic_agva_min_displacement,
                args.dynamic_agva_max_displacement,
            )
        )
    print("Channel conversion augmentation: disabled")
    print("IR ChannelAdapGray enabled:", channel_adap_gray_enabled(args))
    print("Ground ChannelExchange/Gray enabled: False")
    print("================================================")


def add_dynamic_agva_arguments(parser, default_mask_dir, default_image_root=""):
    parser.add_argument(
        "--use-dynamic-agva", action="store_true",
        help="Apply mask-guided AGVA online in aerial loss-training loaders",
    )
    parser.add_argument(
        "--dynamic-agva-mask-dir", type=str, default=default_mask_dir,
        help="Directory containing SCHP foreground PNG masks",
    )
    parser.add_argument(
        "--dynamic-agva-image-root", type=str, default=default_image_root,
        help="Optional original aerial image root for mirrored mask lookup",
    )
    parser.add_argument(
        "--dynamic-agva-slots", type=int, choices=(1, 2), default=1,
        help="Use 1 for 1/3 probability or 2 for 2/3 probability",
    )
    parser.add_argument(
        "--dynamic-agva-min-displacement", type=float, default=0.030,
        help="Minimum per-image maximum displacement ratio",
    )
    parser.add_argument(
        "--dynamic-agva-max-displacement", type=float, default=0.040,
        help="Maximum per-image maximum displacement ratio",
    )
    parser.add_argument(
        "--dynamic-agva-strict-masks", action="store_true",
        help="Fail instead of falling back to the original image when a mask is missing",
    )
    parser.add_argument(
        "--channel-adap-gray", choices=CHANNEL_ADAP_GRAY_MODES,
        default="off",
        help="Fixed to off; retained only as an explicit reproducibility guard.",
    )
