"""Torch datasets and dataloaders for MOT sequences."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms.functional import to_tensor

from .transforms import (
    build_detection_transforms,
    finalize_target_after_transforms,
    prepare_target_for_transforms,
)

TargetTransforms = Callable[[Image.Image, dict], Tuple[torch.Tensor, dict]]


@dataclass(frozen=True)
class FrameRecord:
    dataset_sequence: str
    short_sequence: str
    frame_id: int
    image_path: Path
    annotation_path: Path


class MOTDetectionDataset(Dataset):
    """Detection-ready dataset for MOT sequences."""

    def __init__(
        self,
        images_root: Path | str,
        annotations_root: Path | str,
        *,
        sequences: Optional[Sequence[str]] = None,
        transforms: Optional[TargetTransforms] = None,
        min_visibility: float = 0.0,
        label_default: int = 1,
    ) -> None:
        self.images_root = Path(images_root)
        self.dataset_root = self.images_root.parent
        self.annotations_root = Path(annotations_root)
        self.transforms = transforms
        self.min_visibility = min_visibility
        self.label_default = label_default

        if sequences is not None:
            normalized = set()
            for seq in sequences:
                seq_norm = seq.strip("/")
                normalized.add(seq_norm)
                if not seq_norm.startswith("train") and not seq_norm.startswith("test"):
                    normalized.add(f"train/{seq_norm}")
                    normalized.add(f"test/{seq_norm}")
            self.sequence_filter = normalized
        else:
            self.sequence_filter = None

        self.frame_records: List[FrameRecord] = []
        self._index_frames()

        if not self.frame_records:
            raise ValueError(
                "No frames discovered. Check images_root, annotations_root, and sequences filters."
            )

    def _index_frames(self) -> None:
        for img_dir in sorted(self.images_root.glob("**/img1")):
            short_sequence = str(img_dir.parent.relative_to(self.images_root)).replace("\\", "/")
            dataset_sequence = str(img_dir.parent.relative_to(self.dataset_root)).replace("\\", "/")

            if self.sequence_filter is not None and (
                short_sequence not in self.sequence_filter
                and dataset_sequence not in self.sequence_filter
            ):
                continue

            frames_dir = self.annotations_root / dataset_sequence / "frames"
            if not frames_dir.exists():
                continue

            for json_file in sorted(frames_dir.glob("*.json")):
                frame_id = int(json_file.stem)
                image_path = img_dir / f"{frame_id:06d}.jpg"
                if not image_path.exists():
                    continue
                self.frame_records.append(
                    FrameRecord(
                        dataset_sequence=dataset_sequence,
                        short_sequence=short_sequence,
                        frame_id=frame_id,
                        image_path=image_path,
                        annotation_path=json_file,
                    )
                )

    def __len__(self) -> int:  # pragma: no cover - simple getter
        return len(self.frame_records)

    def _load_annotations(self, record: FrameRecord) -> dict:
        with record.annotation_path.open("r", encoding="utf-8") as handle:
            entries = json.load(handle)

        boxes: List[Tuple[float, float, float, float]] = []
        labels: List[int] = []
        track_ids: List[int] = []
        areas: List[float] = []
        crowd: List[int] = []

        for entry in entries:
            visibility = entry.get("visibility")
            if visibility is not None and visibility < self.min_visibility:
                continue
            left, top, width, height = (
                float(entry["bb_left"]),
                float(entry["bb_top"]),
                float(entry["bb_width"]),
                float(entry["bb_height"]),
            )
            x_max = left + width
            y_max = top + height
            boxes.append((left, top, x_max, y_max))

            label = entry.get("class_id", self.label_default)
            if label is None or label == -1:
                label = self.label_default
            labels.append(int(label))
            track_ids.append(int(entry["object_id"]))
            areas.append(width * height)
            crowd.append(0)

        return {
            "boxes": torch.tensor(boxes, dtype=torch.float32)
            if boxes
            else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
            if labels
            else torch.zeros((0,), dtype=torch.int64),
            "track_ids": torch.tensor(track_ids, dtype=torch.int64)
            if track_ids
            else torch.zeros((0,), dtype=torch.int64),
            "areas": torch.tensor(areas, dtype=torch.float32)
            if areas
            else torch.zeros((0,), dtype=torch.float32),
            "iscrowd": torch.tensor(crowd, dtype=torch.int64)
            if crowd
            else torch.zeros((0,), dtype=torch.int64),
        }

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        record = self.frame_records[idx]
        image = Image.open(record.image_path).convert("RGB")
        width, height = image.size
        targets = self._load_annotations(record)
        targets["frame_id"] = record.frame_id
        targets["sequence"] = record.dataset_sequence
        targets["orig_size"] = torch.tensor([height, width], dtype=torch.int32)

        if self.transforms is not None:
            prepared = prepare_target_for_transforms(
                targets["boxes"], image_size=(height, width)
            )
            prepared.update({k: v for k, v in targets.items() if k not in prepared})
            image, prepared = self.transforms(image, prepared)
            targets.update(prepared)
            targets = finalize_target_after_transforms(targets)
        else:
            image = to_tensor(image)

        return image, targets


def detection_collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def create_detection_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
    )


def split_train_val(
    dataset: MOTDetectionDataset,
    *,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[Subset, Subset]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)
    val_size = int(len(dataset) * val_fraction)
    val_indices = indices[:val_size].tolist()
    train_indices = indices[val_size:].tolist()
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def build_detection_dataloaders(
    *,
    images_root: Path | str,
    annotations_root: Path | str,
    train_sequences: Optional[Sequence[str]] = None,
    val_sequences: Optional[Sequence[str]] = None,
    batch_size: int = 2,
    num_workers: int = 0,
    min_visibility: float = 0.0,
    label_default: int = 1,
    val_fraction: Optional[float] = None,
    transform_kwargs: Optional[dict] = None,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    transform_kwargs = transform_kwargs or {}

    train_transforms = build_detection_transforms(train=True, **transform_kwargs)
    val_transforms = build_detection_transforms(train=False, **transform_kwargs)

    train_dataset = MOTDetectionDataset(
        images_root,
        annotations_root,
        sequences=train_sequences,
        transforms=train_transforms,
        min_visibility=min_visibility,
        label_default=label_default,
    )

    val_loader = None

    if val_sequences is not None:
        val_dataset = MOTDetectionDataset(
            images_root,
            annotations_root,
            sequences=val_sequences,
            transforms=val_transforms,
            min_visibility=min_visibility,
            label_default=label_default,
        )
        val_loader = create_detection_dataloader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
    elif val_fraction is not None:
        train_subset, val_subset = split_train_val(
            train_dataset, val_fraction=val_fraction
        )
        val_dataset = val_subset
        train_dataset = train_subset
        val_loader = create_detection_dataloader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    train_loader = create_detection_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    return train_loader, val_loader


__all__ = [
    "MOTDetectionDataset",
    "detection_collate_fn",
    "create_detection_dataloader",
    "split_train_val",
    "build_detection_dataloaders",
]
