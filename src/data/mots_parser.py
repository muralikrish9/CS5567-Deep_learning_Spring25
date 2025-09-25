"""Utilities to parse MOT16/MOTS style ground-truth files and create quick visual checks."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class MotAnnotation:
    """Container for a single MOT-style annotation record."""

    frame_id: int
    object_id: int
    bbox: Tuple[float, float, float, float]
    confidence: float
    class_id: Optional[int] = None
    visibility: Optional[float] = None
    mask_rle: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize annotation to a dictionary."""
        return {
            "frame_id": self.frame_id,
            "object_id": self.object_id,
            "bb_left": self.bbox[0],
            "bb_top": self.bbox[1],
            "bb_width": self.bbox[2],
            "bb_height": self.bbox[3],
            "confidence": self.confidence,
            "class_id": self.class_id,
            "visibility": self.visibility,
            "mask_rle": self.mask_rle,
        }


def parse_mot_annotations(
    gt_file: Path | str,
    *,
    include_masks: bool = False,
    skip_invalid: bool = True,
) -> List[MotAnnotation]:
    """Parse a MOT16/MOTS style ground-truth file."""

    path = Path(gt_file)
    if not path.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {path}")

    annotations: List[MotAnnotation] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if "," in line:
                parts = [token.strip() for token in line.split(",")]
            else:
                parts = [token.strip() for token in line.split()]

            try:
                frame_id = int(parts[0])
                object_id = int(parts[1])
                bb_left = float(parts[2])
                bb_top = float(parts[3])
                bb_width = float(parts[4])
                bb_height = float(parts[5])
                confidence = float(parts[6]) if len(parts) > 6 else 1.0
                class_id = int(parts[7]) if len(parts) > 7 and parts[7] else None
                visibility = (
                    float(parts[8]) if len(parts) > 8 and parts[8] else None
                )
                mask_rle = parts[9] if len(parts) > 9 else None
            except (IndexError, ValueError) as exc:
                if skip_invalid:
                    continue
                raise ValueError(
                    f"Failed to parse line {line_number} in {path}: {raw_line!r}"
                ) from exc

            if not include_masks:
                mask_rle = None

            annotations.append(
                MotAnnotation(
                    frame_id=frame_id,
                    object_id=object_id,
                    bbox=(bb_left, bb_top, bb_width, bb_height),
                    confidence=confidence,
                    class_id=class_id,
                    visibility=visibility,
                    mask_rle=mask_rle,
                )
            )

    return annotations


def annotations_by_frame(
    annotations: Sequence[MotAnnotation],
) -> Dict[int, List[MotAnnotation]]:
    """Group annotations by frame id."""

    grouped: Dict[int, List[MotAnnotation]] = defaultdict(list)
    for annotation in annotations:
        grouped[annotation.frame_id].append(annotation)

    return {frame_id: grouped[frame_id] for frame_id in sorted(grouped)}


def decode_rle_mask(
    rle: str,
    height: int,
    width: int,
    *,
    dtype: np.dtype = np.uint8,
) -> np.ndarray:
    """Decode a MOTS-style run-length encoded mask."""

    counts = [int(value) for value in rle.strip().split() if value]
    if not counts:
        raise ValueError("RLE string is empty or malformed")

    total_pixels = height * width
    decoded = np.zeros(total_pixels, dtype=dtype)

    current_index = 0
    fill_value = 0

    for run_length in counts:
        next_index = current_index + run_length
        if next_index > total_pixels:
            raise ValueError("RLE run length exceeds mask dimensions")
        if fill_value == 1:
            decoded[current_index:next_index] = 1
        current_index = next_index
        fill_value = 1 - fill_value

    if current_index != total_pixels:
        decoded[current_index:] = 0

    return decoded.reshape((height, width), order="F")


def mot_annotations_to_dataframe(
    annotations: Sequence[MotAnnotation],
):
    """Convert annotations to a pandas DataFrame."""

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pandas is required for mot_annotations_to_dataframe; install pandas first"
        ) from exc

    records = [annotation.to_dict() for annotation in annotations]
    return pd.DataFrame.from_records(records)


def render_annotations_overlay(
    image_path: Path | str,
    annotations: Sequence[MotAnnotation],
    *,
    output_path: Optional[Path | str] = None,
    mask_alpha: float = 0.35,
    box_color: Tuple[int, int, int] = (0, 255, 0),
    text_color: Tuple[int, int, int] = (255, 255, 255),
):
    """Render bounding boxes (and masks when available) on a frame for sanity checks."""

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Pillow is required for render_annotations_overlay; install pillow first"
        ) from exc

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    base_image = Image.open(image_path).convert("RGBA")

    overlay_array = np.array(base_image, dtype=np.uint8)
    height, width = base_image.height, base_image.width

    clamped_alpha = max(0.0, min(float(mask_alpha), 1.0))
    mask_color = np.array(box_color, dtype=np.float32)

    for annotation in annotations:
        if annotation.mask_rle:
            mask = decode_rle_mask(annotation.mask_rle, height=height, width=width).astype(bool)
            if mask.any():
                source_pixels = overlay_array[..., :3][mask]
                blended = (1.0 - clamped_alpha) * source_pixels + clamped_alpha * mask_color
                overlay_array[..., :3][mask] = blended.astype(np.uint8)

    overlay_image = Image.fromarray(overlay_array, mode="RGBA")
    draw = ImageDraw.Draw(overlay_image)

    try:
        font = ImageFont.load_default()
    except OSError:  # pragma: no cover
        font = None

    for annotation in annotations:
        left, top, width_box, height_box = annotation.bbox
        right = left + width_box
        bottom = top + height_box
        draw.rectangle([left, top, right, bottom], outline=box_color, width=2)

        label = str(annotation.object_id)
        if font is not None:
            draw.text((left + 2, top + 2), label, fill=text_color, font=font)
        else:  # pragma: no cover
            draw.text((left + 2, top + 2), label, fill=text_color)

    result = overlay_image.convert("RGB")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path)

    return result


__all__ = [
    "MotAnnotation",
    "parse_mot_annotations",
    "annotations_by_frame",
    "decode_rle_mask",
    "mot_annotations_to_dataframe",
    "render_annotations_overlay",
]
