from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler

try:
    from .common import ViewRecord
except ImportError:
    from common import ViewRecord  # type: ignore


class ForegroundRenderer:
    def __init__(
        self,
        source_roots: Dict[str, Path],
        mask_roots: Dict[str, Path],
        background_gray: int = 128,
    ) -> None:
        self.source_roots = {key: value.resolve() for key, value in source_roots.items()}
        self.mask_roots = {key: value.resolve() for key, value in mask_roots.items()}
        self.background_gray = background_gray
        for domain in ("aerial", "ground"):
            if domain not in self.source_roots or domain not in self.mask_roots:
                raise ValueError("foreground mode requires source and mask roots for both domains")

    def mask_path(self, record: ViewRecord) -> Path:
        image_path = Path(record.image_path).resolve()
        try:
            relative = image_path.relative_to(self.source_roots[record.domain])
        except ValueError:
            raise ValueError(
                "Image {} is not under configured {} source root {}".format(
                    image_path, record.domain, self.source_roots[record.domain]
                )
            )
        candidate = self.mask_roots[record.domain] / relative
        if candidate.is_file():
            return candidate
        png_candidate = candidate.with_suffix(".png")
        if png_candidate.is_file():
            return png_candidate
        raise FileNotFoundError(
            "No SCHP mask for {} (tried {} and {})".format(
                image_path, candidate, png_candidate
            )
        )

    def __call__(self, image: Image.Image, record: ViewRecord) -> Image.Image:
        mask = Image.open(str(self.mask_path(record))).convert("L")
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.NEAREST)
        binary_mask = mask.point(lambda value: 255 if value > 0 else 0, mode="L")
        background = Image.new(
            "RGB", image.size, (self.background_gray,) * 3
        )
        return Image.composite(image, background, binary_mask)


class ViewDataset(Dataset):
    def __init__(
        self,
        records: Sequence[ViewRecord],
        transform=None,
        foreground_renderer: Optional[ForegroundRenderer] = None,
    ) -> None:
        self.records = list(records)
        self.transform = transform
        self.foreground_renderer = foreground_renderer

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        with Image.open(record.image_path) as handle:
            image = handle.convert("RGB")
        if self.foreground_renderer is not None:
            image = self.foreground_renderer(image, record)
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "target": torch.tensor(float(record.target), dtype=torch.float32),
            "image_path": record.image_path,
            "pid": record.pid,
            "camera": record.camera,
            "domain": record.domain,
            "fold": record.fold,
        }


class BalancedDomainBatchSampler(Sampler[List[int]]):
    """Yield half aerial and half ground samples, reshuffling on wraparound."""

    def __init__(
        self, records: Sequence[ViewRecord], batch_size: int, seed: int = 1
    ) -> None:
        if batch_size < 2 or batch_size % 2:
            raise ValueError("batch_size must be a positive even number")
        self.aerial = [index for index, row in enumerate(records) if row.domain == "aerial"]
        self.ground = [index for index, row in enumerate(records) if row.domain == "ground"]
        if not self.aerial or not self.ground:
            raise ValueError("balanced batches require both aerial and ground records")
        self.batch_size = batch_size
        self.half = batch_size // 2
        self.seed = seed
        self.epoch = 0
        self.num_batches = max(
            math.ceil(len(self.aerial) / self.half),
            math.ceil(len(self.ground) / self.half),
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    @staticmethod
    def _draw(pool: List[int], count: int, rng: random.Random) -> Iterator[int]:
        values = list(pool)
        rng.shuffle(values)
        cursor = 0
        while True:
            if cursor == len(values):
                rng.shuffle(values)
                cursor = 0
            yield values[cursor]
            cursor += 1

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        aerial = self._draw(self.aerial, self.half, rng)
        ground = self._draw(self.ground, self.half, rng)
        for _ in range(self.num_batches):
            batch = [next(aerial) for _ in range(self.half)]
            batch.extend(next(ground) for _ in range(self.half))
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.num_batches

