"""Pure-PyTorch compatible SCHP/CE2P inference for LIP-20 human parsing.

This file reproduces the network and inference geometry used by the official
Self-Correction-Human-Parsing `simple_extractor.py`, but replaces the legacy
compiled InPlaceABNSync extension with a state-dict-compatible PyTorch module.
This avoids the common Windows/modern-PyTorch build failure while preserving
checkpoint parameter names.

Original project:
    Peike Li et al., Self-Correction for Human Parsing
    https://github.com/GoGoDuck912/Self-Correction-Human-Parsing
The original project is MIT licensed. This compatibility wrapper retains the
same model topology for inference and adds strict checkpoint validation.
"""

from __future__ import annotations

import functools
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


LIP_LABELS: Tuple[str, ...] = (
    "Background", "Hat", "Hair", "Glove", "Sunglasses", "Upper-clothes",
    "Dress", "Coat", "Socks", "Pants", "Jumpsuits", "Scarf", "Skirt",
    "Face", "Left-arm", "Right-arm", "Left-leg", "Right-leg",
    "Left-shoe", "Right-shoe",
)


class InPlaceABNCompat(nn.BatchNorm2d):
    """Inference-compatible replacement for legacy InPlaceABNSync.

    The old SCHP checkpoint stores ordinary BN affine/running-stat tensors.
    Subclassing ``nn.BatchNorm2d`` preserves those state-dict key names. The
    only normally missing keys are ``num_batches_tracked`` from newer PyTorch.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        activation: str = "leaky_relu",
        slope: float = 0.01,
        **_: object,
    ) -> None:
        super().__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=True,
        )
        self.activation = activation
        self.slope = float(slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        if self.activation in {"none", "identity", None}:  # type: ignore[arg-type]
            return x
        if self.activation == "relu":
            return F.relu(x, inplace=False)
        if self.activation == "elu":
            return F.elu(x, inplace=False)
        if self.activation == "leaky_relu":
            return F.leaky_relu(x, negative_slope=self.slope, inplace=False)
        raise ValueError(f"Unsupported ABN activation: {self.activation}")


BatchNorm2d = functools.partial(InPlaceABNCompat, activation="none")
affine_par = True


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: Optional[nn.Module] = None,
        fist_dilation: int = 1,  # kept for checkpoint/topology compatibility
        multi_grid: int = 1,
    ) -> None:
        super().__init__()
        del fist_dilation
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation * multi_grid,
            dilation=dilation * multi_grid,
            bias=False,
        )
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.relu_inplace(out + residual)
        return out


class PSPModule(nn.Module):
    def __init__(
        self,
        features: int,
        out_features: int = 512,
        sizes: Sequence[int] = (1, 2, 3, 6),
    ) -> None:
        super().__init__()
        self.stages = nn.ModuleList(
            [self._make_stage(features, out_features, size) for size in sizes]
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                features + len(sizes) * out_features,
                out_features,
                kernel_size=3,
                padding=1,
                dilation=1,
                bias=False,
            ),
            InPlaceABNCompat(out_features),
        )

    @staticmethod
    def _make_stage(features: int, out_features: int, size: int) -> nn.Sequential:
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(size, size)),
            nn.Conv2d(features, out_features, kernel_size=1, bias=False),
            InPlaceABNCompat(out_features),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        h, w = feats.shape[2:]
        priors = [
            F.interpolate(stage(feats), size=(h, w), mode="bilinear", align_corners=True)
            for stage in self.stages
        ]
        priors.append(feats)
        return self.bottleneck(torch.cat(priors, dim=1))


class EdgeModule(nn.Module):
    def __init__(
        self,
        in_fea: Sequence[int] = (256, 512, 1024),
        mid_fea: int = 256,
        out_fea: int = 2,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_fea[0], mid_fea, 1, bias=False),
            InPlaceABNCompat(mid_fea),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_fea[1], mid_fea, 1, bias=False),
            InPlaceABNCompat(mid_fea),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_fea[2], mid_fea, 1, bias=False),
            InPlaceABNCompat(mid_fea),
        )
        self.conv4 = nn.Conv2d(mid_fea, out_fea, 3, padding=1, bias=True)
        self.conv5 = nn.Conv2d(out_fea * 3, out_fea, 1, bias=True)

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h, w = x1.shape[2:]
        edge1_fea = self.conv1(x1)
        edge1 = self.conv4(edge1_fea)
        edge2_fea = self.conv2(x2)
        edge2 = self.conv4(edge2_fea)
        edge3_fea = self.conv3(x3)
        edge3 = self.conv4(edge3_fea)

        edge2_fea = F.interpolate(edge2_fea, size=(h, w), mode="bilinear", align_corners=True)
        edge3_fea = F.interpolate(edge3_fea, size=(h, w), mode="bilinear", align_corners=True)
        edge2 = F.interpolate(edge2, size=(h, w), mode="bilinear", align_corners=True)
        edge3 = F.interpolate(edge3, size=(h, w), mode="bilinear", align_corners=True)

        edge = self.conv5(torch.cat([edge1, edge2, edge3], dim=1))
        edge_fea = torch.cat([edge1_fea, edge2_fea, edge3_fea], dim=1)
        return edge, edge_fea


class DecoderModule(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, 1, bias=False),
            InPlaceABNCompat(256),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            InPlaceABNCompat(48),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(304, 256, 1, bias=False),
            InPlaceABNCompat(256),
            nn.Conv2d(256, 256, 1, bias=False),
            InPlaceABNCompat(256),
        )
        self.conv4 = nn.Conv2d(256, num_classes, 1, bias=True)

    def forward(
        self, xt: torch.Tensor, xl: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h, w = xl.shape[2:]
        xt = F.interpolate(self.conv1(xt), size=(h, w), mode="bilinear", align_corners=True)
        xl = self.conv2(xl)
        x = self.conv3(torch.cat([xt, xl], dim=1))
        return self.conv4(x), x


class SCHPResNet101(nn.Module):
    def __init__(self, num_classes: int = 20) -> None:
        super().__init__()
        self.inplanes = 128
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 23, stride=2)
        self.layer4 = self._make_layer(
            Bottleneck, 512, 3, stride=1, dilation=2, multi_grid=(1, 1, 1)
        )
        self.context_encoding = PSPModule(2048, 512)
        self.edge = EdgeModule()
        self.decoder = DecoderModule(num_classes)
        # Keep official misspelling because checkpoint keys use `fushion`.
        self.fushion = nn.Sequential(
            nn.Conv2d(1024, 256, 1, bias=False),
            InPlaceABNCompat(256),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, 1, bias=True),
        )

    def _make_layer(
        self,
        block: type[Bottleneck],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilation: int = 1,
        multi_grid: Sequence[int] | int = 1,
    ) -> nn.Sequential:
        downsample: Optional[nn.Module] = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                BatchNorm2d(planes * block.expansion, affine=affine_par),
            )

        def grid(index: int) -> int:
            if isinstance(multi_grid, (tuple, list)):
                return int(multi_grid[index % len(multi_grid)])
            return 1

        layers: List[nn.Module] = [
            block(
                self.inplanes,
                planes,
                stride,
                dilation=dilation,
                downsample=downsample,
                multi_grid=grid(0),
            )
        ]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    dilation=dilation,
                    multi_grid=grid(i),
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        context = self.context_encoding(x5)
        parsing_result, parsing_fea = self.decoder(context, x2)
        edge_result, edge_fea = self.edge(x2, x3, x4)
        fusion_result = self.fushion(torch.cat([parsing_fea, edge_fea], dim=1))
        return [[parsing_result, fusion_result], [edge_result]]


@dataclass(frozen=True)
class CheckpointReport:
    parameter_keys: int
    missing_keys: Tuple[str, ...]
    unexpected_keys: Tuple[str, ...]


@dataclass(frozen=True)
class ParsingResult:
    parsing: np.ndarray  # uint8 class-id map, HxW
    binary_mask: np.ndarray  # uint8 0/255, HxW
    foreground_ratio: float
    bbox: Optional[Tuple[int, int, int, int]]  # top,bottom,left,right
    component_count: int
    touches_border: bool


def load_schp_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> Tuple[SCHPResNet101, CheckpointReport]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SCHP checkpoint not found: {checkpoint_path}")

    model = SCHPResNet101(num_classes=20)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before weights_only argument
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state = checkpoint
    else:
        raise RuntimeError("Unsupported checkpoint format: expected a state_dict dictionary")

    cleaned: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        new_key = key[7:] if key.startswith("module.") else key
        cleaned[new_key] = value

    incompatible = model.load_state_dict(cleaned, strict=False)
    missing = tuple(incompatible.missing_keys)
    unexpected = tuple(incompatible.unexpected_keys)
    serious_missing = tuple(k for k in missing if not k.endswith("num_batches_tracked"))
    if serious_missing or unexpected:
        details = [
            "SCHP checkpoint does not match the compatibility model.",
            f"Serious missing keys ({len(serious_missing)}): {serious_missing[:20]}",
            f"Unexpected keys ({len(unexpected)}): {unexpected[:20]}",
            "Do not continue with a partially loaded/random model.",
        ]
        raise RuntimeError("\n".join(details))

    model.to(device)
    model.eval()
    return model, CheckpointReport(len(cleaned), missing, unexpected)


def get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point: Sequence[float], rot_rad: float) -> np.ndarray:
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    return np.array(
        [src_point[0] * cs - src_point[1] * sn,
         src_point[0] * sn + src_point[1] * cs],
        dtype=np.float32,
    )


def get_affine_transform(
    center: np.ndarray,
    scale: np.ndarray,
    rot: float,
    output_size: Sequence[int],
    inv: bool = False,
) -> np.ndarray:
    scale_tmp = np.asarray(scale, dtype=np.float32)
    src_w = float(scale_tmp[0])
    dst_w = float(output_size[1])
    dst_h = float(output_size[0])
    rot_rad = np.pi * rot / 180.0
    src_dir = get_dir([0.0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0.0, (dst_w - 1.0) * -0.5], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0] = center
    src[1] = center + src_dir
    dst[0] = [(dst_w - 1.0) * 0.5, (dst_h - 1.0) * 0.5]
    dst[1] = dst[0] + dst_dir
    src[2] = get_3rd_point(src[0], src[1])
    dst[2] = get_3rd_point(dst[0], dst[1])
    return cv2.getAffineTransform(dst, src) if inv else cv2.getAffineTransform(src, dst)


def image_center_scale(width: int, height: int, input_size: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    box_w = float(max(width - 1, 1))
    box_h = float(max(height - 1, 1))
    center = np.array([box_w * 0.5, box_h * 0.5], dtype=np.float32)
    aspect_ratio = float(input_size[1]) / float(input_size[0])
    if box_w > aspect_ratio * box_h:
        box_h = box_w / aspect_ratio
    elif box_w < aspect_ratio * box_h:
        box_w = box_h * aspect_ratio
    return center, np.array([box_w, box_h], dtype=np.float32)


def transform_logits_to_original(
    logits: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    width: int,
    height: int,
    input_size: Sequence[int],
) -> np.ndarray:
    trans = get_affine_transform(center, scale, 0, input_size, inv=True)
    channels = [
        cv2.warpAffine(
            logits[:, :, index],
            trans,
            (int(width), int(height)),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        for index in range(logits.shape[2])
    ]
    return np.stack(channels, axis=2)


def clean_binary_mask(
    mask: np.ndarray,
    morphology_kernel: int = 3,
    keep_largest_component: bool = True,
    min_component_area: int = 0,
) -> Tuple[np.ndarray, int]:
    binary = (mask > 0).astype(np.uint8)
    if morphology_kernel > 1:
        k = int(morphology_kernel)
        if k % 2 == 0:
            k += 1
        kernel = np.ones((k, k), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    component_count = max(count - 1, 0)
    if count <= 1:
        return binary * 255, component_count

    valid = []
    for label_id in range(1, count):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area >= int(min_component_area):
            valid.append((area, label_id))

    if not valid:
        return np.zeros_like(binary, dtype=np.uint8), component_count
    if keep_largest_component:
        _, label_id = max(valid)
        binary = (labels == label_id).astype(np.uint8)
    else:
        valid_ids = np.array([label_id for _, label_id in valid], dtype=labels.dtype)
        binary = np.isin(labels, valid_ids).astype(np.uint8)
    return binary * 255, component_count


class SCHPParser:
    """Single-image SCHP parser with official LIP preprocessing/postprocessing."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "auto",
        input_size: Tuple[int, int] = (473, 473),
        morphology_kernel: int = 3,
        keep_largest_component: bool = True,
        min_component_area: int = 0,
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
        self.device = torch.device(device)
        self.input_size = tuple(map(int, input_size))
        self.morphology_kernel = int(morphology_kernel)
        self.keep_largest_component = bool(keep_largest_component)
        self.min_component_area = int(min_component_area)
        self.model, self.checkpoint_report = load_schp_checkpoint(
            checkpoint_path, device=self.device
        )
        self.mean = torch.tensor([0.406, 0.456, 0.485], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor([0.225, 0.224, 0.229], dtype=torch.float32).view(3, 1, 1)

    def parse(self, image: str | Path | Image.Image | np.ndarray) -> ParsingResult:
        if isinstance(image, (str, Path)):
            image_bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise ValueError(f"Cannot read image: {image}")
        elif isinstance(image, Image.Image):
            image_bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        else:
            arr = np.asarray(image)
            if arr.ndim != 3 or arr.shape[2] != 3:
                raise ValueError(f"Expected HxWx3 image, got {arr.shape}")
            image_bgr = arr.copy()

        height, width = image_bgr.shape[:2]
        center, scale = image_center_scale(width, height, self.input_size)
        trans = get_affine_transform(center, scale, 0, self.input_size, inv=False)
        network_input = cv2.warpAffine(
            image_bgr,
            trans,
            (int(self.input_size[1]), int(self.input_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        tensor = torch.from_numpy(network_input.transpose(2, 0, 1)).float() / 255.0
        tensor = ((tensor - self.mean) / self.std).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            output = self.model(tensor)
            fusion = output[0][-1]
            fusion = F.interpolate(
                fusion,
                size=self.input_size,
                mode="bilinear",
                align_corners=True,
            )
        logits = fusion[0].permute(1, 2, 0).float().cpu().numpy()
        logits = transform_logits_to_original(
            logits, center, scale, width, height, self.input_size
        )
        parsing = np.argmax(logits, axis=2).astype(np.uint8)
        raw_binary = (parsing > 0).astype(np.uint8) * 255
        binary, component_count = clean_binary_mask(
            raw_binary,
            morphology_kernel=self.morphology_kernel,
            keep_largest_component=self.keep_largest_component,
            min_component_area=self.min_component_area,
        )

        ys, xs = np.where(binary > 0)
        if ys.size:
            bbox: Optional[Tuple[int, int, int, int]] = (
                int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
            )
            touches_border = bool(
                ys.min() == 0 or xs.min() == 0 or ys.max() == height - 1 or xs.max() == width - 1
            )
        else:
            bbox = None
            touches_border = False
        return ParsingResult(
            parsing=parsing,
            binary_mask=binary,
            foreground_ratio=float(np.mean(binary > 0)),
            bbox=bbox,
            component_count=component_count,
            touches_border=touches_border,
        )
