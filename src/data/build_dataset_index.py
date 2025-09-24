"""Helpers to convert raw MOT/MOTS annotations into per-frame JSON records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

from .mots_parser import MotAnnotation, annotations_by_frame, parse_mot_annotations


def find_gt_files(dataset_root: Path) -> List[Path]:
    """Discover ``gt.txt`` files under a MOT-style directory tree."""

    dataset_root = Path(dataset_root)
    return sorted(dataset_root.glob("**/gt/gt.txt"))


def export_frame_annotations(
    frame_annotations: Iterable[MotAnnotation],
    destination: Path,
) -> None:
    """Write a single frame worth of annotations to JSON."""

    payload = [annotation.to_dict() for annotation in frame_annotations]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_dataset_index(
    dataset_root: Path,
    output_root: Path,
    *,
    include_masks: bool = False,
) -> Dict[str, Dict[str, int]]:
    """Parse every ``gt.txt`` file and emit per-frame JSON records."""

    dataset_root = Path(dataset_root)
    output_root = Path(output_root)

    summary: Dict[str, Dict[str, int]] = {}

    for gt_file in find_gt_files(dataset_root):
        annotations = parse_mot_annotations(gt_file, include_masks=include_masks)
        frames = annotations_by_frame(annotations)

        relative_sequence = gt_file.parent.parent.relative_to(dataset_root)
        sequence_output = output_root / relative_sequence / "frames"

        for frame_id, frame_annotations in frames.items():
            filename = f"{frame_id:06d}.json"
            export_frame_annotations(frame_annotations, sequence_output / filename)

        summary[str(relative_sequence).replace("\\", "/")] = {
            "frames": len(frames),
            "annotations": len(annotations),
        }

    if summary:
        output_root.mkdir(parents=True, exist_ok=True)
        summary_path = output_root / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-frame annotation JSON files")
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Path to the MOT16/MOTS root (e.g., data/MOT16)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed_annotations"),
        help="Directory where JSON files will be written",
    )
    parser.add_argument(
        "--include-masks",
        action="store_true",
        help="Preserve RLE mask strings when present (MOTS datasets)",
    )
    args = parser.parse_args()

    summary = build_dataset_index(
        args.dataset_root,
        args.output,
        include_masks=args.include_masks,
    )

    if not summary:
        print("No annotations processed; ensure dataset_root contains gt/gt.txt files")
    else:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
