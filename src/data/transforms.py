"""Augmentation pipelines for MOT detection tasks without heavy dependencies."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter


TransformFn = Callable[[Image.Image, dict], Tuple[Image.Image, dict]]
TensorTransformFn = Callable[[torch.Tensor, dict], Tuple[torch.Tensor, dict]]


@dataclass
class Compose:
    transforms: Sequence[Callable]

    def __call__(self, image, target):
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target


class RandomColorJitter:
    def __init__(self, jitter: float = 0.2):
        self.jitter = jitter

    def __call__(self, image: Image.Image, target: dict):
        if self.jitter <= 0:
            return image, target
        factors = [
            1.0 + random.uniform(-self.jitter, self.jitter),
            1.0 + random.uniform(-self.jitter, self.jitter),
            1.0 + random.uniform(-self.jitter, self.jitter),
        ]
        brightness, contrast, color = factors
        image = ImageEnhance.Brightness(image).enhance(brightness)
        image = ImageEnhance.Contrast(image).enhance(contrast)
        image = ImageEnhance.Color(image).enhance(color)
        return image, target


class RandomGaussianBlur:
    def __init__(self, probability: float = 0.2, radius: Tuple[float, float] = (0.1, 2.0)):
        self.probability = probability
        self.radius = radius

    def __call__(self, image: Image.Image, target: dict):
        if random.random() < self.probability:
            radius = random.uniform(*self.radius)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        return image, target


class RandomHorizontalFlip:
    def __init__(self, probability: float = 0.5):
        self.probability = probability

    def __call__(self, image: Image.Image, target: dict):
        if random.random() >= self.probability:
            return image, target
        width = image.width
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        boxes = target.get("boxes")
        if boxes is not None and boxes.numel() > 0:
            flipped = boxes.clone()
            flipped[:, 0] = width - boxes[:, 2]
            flipped[:, 2] = width - boxes[:, 0]
            target["boxes"] = flipped
        return image, target


class ResizeShortestEdge:
    def __init__(self, min_size: int = 800, max_size: int = 1333):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, image: Image.Image, target: dict):
        width, height = image.size
        scale = self._compute_scale(width, height)
        if scale != 1.0:
            new_width = int(round(width * scale))
            new_height = int(round(height * scale))
            image = image.resize((new_width, new_height), resample=Image.BILINEAR)
            boxes = target.get("boxes")
            if boxes is not None and boxes.numel() > 0:
                target["boxes"] = boxes * scale
            target["size"] = torch.tensor([new_height, new_width], dtype=torch.int32)
        else:
            target.setdefault("size", torch.tensor([height, width], dtype=torch.int32))
        return image, target

    def _compute_scale(self, width: int, height: int) -> float:
        min_side = min(width, height)
        max_side = max(width, height)
        scale = self.min_size / min_side
        if max_side * scale > self.max_size:
            scale = self.max_size / max_side
        return scale


class ToTensor:
    def __call__(self, image: Image.Image, target: dict):
        array = np.array(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return tensor, target


class Normalize:
    def __init__(self, mean: Iterable[float], std: Iterable[float]):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, tensor: torch.Tensor, target: dict):
        normalized = (tensor - self.mean) / self.std
        return normalized, target


def build_detection_transforms(
    *,
    train: bool,
    image_min_side: int = 800,
    image_max_side: int = 1333,
    color_jitter: float = 0.2,
    blur_prob: float = 0.2,
    hflip_prob: float = 0.5,
):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    stages: List[Callable] = []
    if train:
        stages.extend(
            [
                RandomColorJitter(jitter=color_jitter),
                RandomGaussianBlur(probability=blur_prob),
                RandomHorizontalFlip(probability=hflip_prob),
            ]
        )
    stages.append(ResizeShortestEdge(min_size=image_min_side, max_size=image_max_side))
    stages.append(ToTensor())
    stages.append(Normalize(mean, std))

    return Compose(stages)


def prepare_target_for_transforms(boxes: torch.Tensor, *, image_size: Tuple[int, int]) -> dict:
    target = {"boxes": boxes.clone() if boxes.numel() > 0 else boxes}
    target["size"] = torch.tensor([image_size[0], image_size[1]], dtype=torch.int32)
    return target


def finalize_target_after_transforms(target: dict) -> dict:
    return target


__all__ = [
    "build_detection_transforms",
    "prepare_target_for_transforms",
    "finalize_target_after_transforms",
]
