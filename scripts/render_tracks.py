"""Render tracking overlays from JSON output onto video frames."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    import cv2
except ImportError as exc:  # pragma: no cover - user environment check
    raise ImportError(
        "OpenCV (cv2) is required. Install it with 'pip install opencv-python'."
    ) from exc

import numpy as np
from tqdm import tqdm


def load_tracks(path: Path) -> List[Dict]:
    data = json.loads(path.read_text())
    data.sort(key=lambda entry: (entry["frame"], entry["track_id"]))
    return data


def render_sequence(
    images_dir: Path,
    tracks_json: Path,
    output_path: Path,
    *,
    fps: float = 10.0,
    scale: float = 1.0,
) -> None:
    frames = sorted(images_dir.glob("*.jpg"))
    if not frames:
        raise ValueError(f"No frames found in {images_dir}")

    tracks = load_tracks(tracks_json)
    by_frame: Dict[int, List[Dict]] = {}
    for entry in tracks:
        by_frame.setdefault(entry["frame"], []).append(entry)

    sample_frame = cv2.imread(str(frames[0]))
    height, width = sample_frame.shape[:2]
    frame_size = (int(width * scale), int(height * scale))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        frame_size,
    )

    for frame_idx, frame_path in enumerate(tqdm(frames, desc="Rendering"), start=1):
        image = cv2.imread(str(frame_path))
        if scale != 1.0:
            image = cv2.resize(image, frame_size)

        for track in by_frame.get(frame_idx, []):
            left, top, width_box, height_box = track["bbox"]
            right = left + width_box
            bottom = top + height_box

            if scale != 1.0:
                left *= scale
                top *= scale
                right *= scale
                bottom *= scale

            color = track_color(track["track_id"])
            cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), color, 2)
            label = f"ID {track['track_id']}"
            cv2.putText(
                image,
                label,
                (int(left), int(top) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        writer.write(image)

    writer.release()
    print(f"Saved overlay video to {output_path}")


def track_color(track_id: int) -> tuple:
    np.random.seed(track_id)
    color = tuple(int(c) for c in np.random.randint(0, 255, size=3))
    return color


def parse_args():
    parser = argparse.ArgumentParser(description="Render tracked video from JSON tracks")
    parser.add_argument("images_dir", type=Path, help="Directory with img1/*.jpg frames")
    parser.add_argument("tracks_json", type=Path, help="Path to *_tracks.json output")
    parser.add_argument("output_video", type=Path, help="Where to write the mp4 result")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--scale", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    render_sequence(
        args.images_dir,
        args.tracks_json,
        args.output_video,
        fps=args.fps,
        scale=args.scale,
    )


if __name__ == "__main__":
    main()
