"""Utility to create overlay previews for MOT sequences."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

from .mots_parser import (
    MotAnnotation,
    annotations_by_frame,
    parse_mot_annotations,
    render_annotations_overlay,
)


def frame_ids_to_sample(total_frames: int, stride: int) -> Sequence[int]:
    """Return frame ids to sample given total number of frames and stride."""

    stride = max(1, int(stride))
    frames = list(range(1, total_frames + 1, stride))
    if frames[-1] != total_frames:
        frames.append(total_frames)
    return frames


def generate_sequence_overlays(
    sequence_dir: Path,
    output_dir: Path,
    *,
    stride: int = 50,
) -> int:
    """Generate overlays for a single MOT sequence.

    Parameters
    ----------
    sequence_dir:
        Directory containing ``img1`` and ``gt/gt.txt``.
    output_dir:
        Destination directory for overlay images.
    stride:
        Sample every ``stride`` frames (inclusive).

    Returns
    -------
    int
        Number of overlay images written.
    """

    sequence_dir = Path(sequence_dir)
    output_dir = Path(output_dir)

    gt_file = sequence_dir / "gt" / "gt.txt"
    img_dir = sequence_dir / "img1"
    if not gt_file.exists():
        raise FileNotFoundError(f"Missing gt.txt in {sequence_dir}")
    if not img_dir.exists():
        raise FileNotFoundError(f"Missing img1 directory in {sequence_dir}")

    annotations = parse_mot_annotations(gt_file)
    grouped = annotations_by_frame(annotations)
    frame_ids = frame_ids_to_sample(len(grouped), stride)

    written = 0
    output_dir.mkdir(parents=True, exist_ok=True)

    for frame_id in frame_ids:
        if frame_id not in grouped:
            continue
        image_path = img_dir / f"{frame_id:06d}.jpg"
        if not image_path.exists():
            continue
        destination = output_dir / f"{frame_id:06d}.jpg"
        render_annotations_overlay(image_path, grouped[frame_id], output_path=destination)
        written += 1

    return written


def generate_dataset_overlays(
    dataset_root: Path,
    output_root: Path,
    *,
    stride: int = 50,
) -> int:
    """Generate overlays for every sequence under a MOT dataset root."""

    dataset_root = Path(dataset_root)
    output_root = Path(output_root)

    sequences = list(dataset_root.glob("**/img1"))
    total_written = 0

    for img_dir in sequences:
        sequence_dir = img_dir.parent
        relative = sequence_dir.relative_to(dataset_root)
        sequence_output = output_root / relative
        written = generate_sequence_overlays(
            sequence_dir, sequence_output, stride=stride
        )
        total_written += written

    return total_written


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate overlay previews for MOT sequences")
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Path to the MOT dataset root (e.g. data/train)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/overlays"),
        help="Destination folder for overlay images",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=50,
        help="Sample every N frames (default: 50)",
    )
    args = parser.parse_args()

    written = generate_dataset_overlays(
        args.dataset_root, args.output, stride=args.stride
    )

    print(f"Generated {written} overlays into {args.output}")


if __name__ == "__main__":  # pragma: no cover
    main()
