"""Datasets for training person Re-ID models from MOT-style annotations."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import json
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .mots_parser import MotAnnotation


@dataclass(frozen=True)
class PatchRecord:
    """Metadata describing a single cropped identity patch."""

    sequence: str
    frame_id: int
    object_id: int
    image_path: Path
    bbox: Tuple[float, float, float, float]  # (left, top, right, bottom)

    @property
    def identity_key(self) -> Tuple[str, int]:
        return self.sequence, self.object_id


class MOTReIDPatchDataset(Dataset):
    """Loads cropped person patches for Re-ID training on MOT-style datasets."""

    def __init__(
        self,
        images_root: Path | str,
        annotations_root: Path | str,
        *,
        sequences: Optional[Sequence[str]] = None,
        transform=None,
        context_scale: float = 1.2,
        min_visibility: float = 0.0,
        records: Optional[List[PatchRecord]] = None,
        identity_to_indices: Optional[Dict[Tuple[str, int], List[int]]] = None,
        keep_identities: Optional[Iterable[Tuple[str, int]]] = None,
    ) -> None:
        self.images_root = Path(images_root)
        self.dataset_root = self.images_root.parent
        self.annotations_root = Path(annotations_root)
        self.transform = transform
        self.context_scale = max(1.0, float(context_scale))
        self.min_visibility = float(min_visibility)

        if records is None or identity_to_indices is None:
            self.records, self.identity_to_indices = self._discover_records(
                sequences=sequences,
            )
        else:
            self.records = list(records)
            self.identity_to_indices = {
                key: list(indices) for key, indices in identity_to_indices.items()
            }

        if keep_identities is not None:
            allowed = {tuple(item) for item in keep_identities}
            filtered_indices: List[int] = []
            for key, indices in self.identity_to_indices.items():
                if key in allowed:
                    filtered_indices.extend(indices)
            filtered_indices.sort()
            self.records = [self.records[i] for i in filtered_indices]
            self.identity_to_indices = {}
            for new_idx, record in enumerate(self.records):
                self.identity_to_indices.setdefault(record.identity_key, []).append(new_idx)

        self._validate_identities()

    def _validate_identities(self) -> None:
        identities = [
            key for key, indices in self.identity_to_indices.items() if len(indices) >= 1
        ]
        if not identities:
            raise ValueError(
                "No identities discovered. Check dataset paths or filters provided."
            )

    def spawn_subset(self, identities: Iterable[Tuple[str, int]]) -> "MOTReIDPatchDataset":
        """Create a filtered clone limited to the specified identities."""

        allowed = {tuple(item) for item in identities}
        indices: List[int] = []
        for key in allowed:
            for idx in self.identity_to_indices.get(key, []):
                indices.append(idx)
        if not indices:
            raise ValueError("Requested subset contains no samples")
        indices.sort()
        subset_records = [self.records[i] for i in indices]
        new_mapping: Dict[Tuple[str, int], List[int]] = {}
        for idx, record in enumerate(subset_records):
            new_mapping.setdefault(record.identity_key, []).append(idx)
        return MOTReIDPatchDataset(
            images_root=self.images_root,
            annotations_root=self.annotations_root,
            transform=self.transform,
            context_scale=self.context_scale,
            min_visibility=self.min_visibility,
            records=subset_records,
            identity_to_indices=new_mapping,
        )

    def _discover_records(
        self,
        *,
        sequences: Optional[Sequence[str]] = None,
    ) -> Tuple[List[PatchRecord], Dict[Tuple[str, int], List[int]]]:
        sequence_filter = None
        if sequences is not None:
            sequence_filter = set()
            for seq in sequences:
                normalized = seq.strip("/")
                sequence_filter.add(normalized)
                if not normalized.startswith("train") and not normalized.startswith("test"):
                    sequence_filter.add(f"train/{normalized}")
                    sequence_filter.add(f"test/{normalized}")

        records: List[PatchRecord] = []
        identity_to_indices: Dict[Tuple[str, int], List[int]] = {}

        for gt_file in sorted(self.annotations_root.glob("**/frames/*.json")):
            frame_id = int(gt_file.stem)
            relative_sequence = str(
                gt_file.parent.parent.relative_to(self.annotations_root)
            ).replace("\\", "/")
            if sequence_filter is not None and relative_sequence not in sequence_filter:
                continue

            image_dir = self.images_root / relative_sequence / "img1"
            if not image_dir.exists():
                image_dir = self.dataset_root / relative_sequence / "img1"
            image_path = image_dir / f"{frame_id:06d}.jpg"
            if not image_path.exists():
                continue

            annotations = _load_annotations(gt_file)
            for annotation in annotations:
                if annotation.visibility is not None and annotation.visibility < self.min_visibility:
                    continue
                left, top, width, height = annotation.bbox
                x2 = left + width
                y2 = top + height
                record = PatchRecord(
                    sequence=relative_sequence,
                    frame_id=annotation.frame_id,
                    object_id=annotation.object_id,
                    image_path=image_path,
                    bbox=(left, top, x2, y2),
                )
                identity_key = record.identity_key
                identity_to_indices.setdefault(identity_key, []).append(len(records))
                records.append(record)

        if not records:
            raise ValueError(
                "No patch records constructed. Ensure processed annotations are available."
            )

        return records, identity_to_indices

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = Image.open(record.image_path).convert("RGB")
        left, top, right, bottom = record.bbox
        crop = _crop_with_context(image, (left, top, right, bottom), self.context_scale)
        if self.transform is not None:
            crop = self.transform(crop)
        else:
            crop = _pil_to_tensor(crop)

        return crop, {
            "sequence": record.sequence,
            "frame_id": record.frame_id,
            "object_id": record.object_id,
        }


class ReIDPairDataset(Dataset):
    """Produces positive and negative pairs for Re-ID training."""

    def __init__(
        self,
        base_dataset: MOTReIDPatchDataset,
        *,
        length: int,
        positive_fraction: float = 0.5,
        seed: Optional[int] = None,
    ) -> None:
        if length <= 0:
            raise ValueError("length must be positive")
        self.base_dataset = base_dataset
        self.length = int(length)
        self.positive_fraction = max(0.0, min(float(positive_fraction), 1.0))
        self.rng = random.Random(seed)

        self.identities = [
            key for key, indices in base_dataset.identity_to_indices.items() if len(indices) >= 1
        ]
        self.identities_with_pairs = [
            key for key, indices in base_dataset.identity_to_indices.items() if len(indices) >= 2
        ]
        if not self.identities:
            raise ValueError("Base dataset contains no identities")
        if not self.identities_with_pairs:
            raise ValueError("Need at least one identity with two samples for positive pairs")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        want_positive = self.rng.random() < self.positive_fraction
        if want_positive:
            identity = self.rng.choice(self.identities_with_pairs)
            pool = self.base_dataset.identity_to_indices[identity]
            idx1, idx2 = self.rng.sample(pool, 2)
            label = 1
        else:
            identity1, identity2 = self.rng.sample(self.identities, 2)
            idx1 = self.rng.choice(self.base_dataset.identity_to_indices[identity1])
            idx2 = self.rng.choice(self.base_dataset.identity_to_indices[identity2])
            label = 0

        img1, meta1 = self.base_dataset[idx1]
        img2, meta2 = self.base_dataset[idx2]

        pair_metadata = {
            "meta1": meta1,
            "meta2": meta2,
            "same_identity": bool(label),
        }
        return img1, img2, torch.tensor(label, dtype=torch.float32), pair_metadata


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return tensor


def _crop_with_context(
    image: Image.Image,
    bbox: Tuple[float, float, float, float],
    scale: float,
) -> Image.Image:
    left, top, right, bottom = bbox
    width = right - left
    height = bottom - top
    cx = left + width / 2.0
    cy = top + height / 2.0
    half_w = width * scale / 2.0
    half_h = height * scale / 2.0

    new_left = max(0.0, cx - half_w)
    new_top = max(0.0, cy - half_h)
    new_right = min(float(image.width), cx + half_w)
    new_bottom = min(float(image.height), cy + half_h)

    return image.crop((new_left, new_top, new_right, new_bottom))


def _load_annotations(frame_json: Path) -> List[MotAnnotation]:
    with frame_json.open("r", encoding="utf-8") as handle:
        entries = json.load(handle)

    annotations: List[MotAnnotation] = []
    for entry in entries:
        annotation = MotAnnotation(
            frame_id=int(entry["frame_id"]),
            object_id=int(entry["object_id"]),
            bbox=(
                float(entry["bb_left"]),
                float(entry["bb_top"]),
                float(entry["bb_width"]),
                float(entry["bb_height"]),
            ),
            confidence=float(entry.get("confidence", 1.0)),
            class_id=(
                int(entry["class_id"]) if entry.get("class_id") not in (None, "") else None
            ),
            visibility=(
                float(entry["visibility"]) if entry.get("visibility") not in (None, "") else None
            ),
            mask_rle=entry.get("mask_rle"),
        )
        annotations.append(annotation)
    return annotations


__all__ = [
    "PatchRecord",
    "MOTReIDPatchDataset",
    "ReIDPairDataset",
]
