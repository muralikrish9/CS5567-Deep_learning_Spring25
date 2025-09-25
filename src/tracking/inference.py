"""Run detector + Re-ID embeddings to produce tracked trajectories."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.training.reid_trainer import SiameseNetwork
from src.tracking.association import associate_detections


try:
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "torchvision with detection models is required for tracking inference"
    ) from exc


@dataclass
class TrackerConfig:
    images_root: Path
    annotations_root: Path
    detector_checkpoint: Path
    reid_checkpoint: Path
    output_dir: Path
    sequences: Optional[List[str]] = None
    device: Optional[str] = None
    detection_threshold: float = 0.5
    max_track_age: int = 30
    max_distance: float = 0.5
    iou_weight: float = 0.3
    context_scale: float = 1.2
    crop_size: int = 128
    batch_size: int = 16
    num_workers: int = 2
    smoothing_alpha: float = 0.6


@dataclass
class Track:
    track_id: int
    embedding: np.ndarray
    bbox: np.ndarray
    last_frame: int
    hits: int = 1
    age: int = 0
    smoothed_bbox: Optional[np.ndarray] = None
    history: List[Dict[str, float]] = field(default_factory=list)


def _load_checkpoint(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device)
    except Exception as exc:  # Handle PyTorch 2.6 safe-loading defaults
        message = str(exc)
        if "Weights only load failed" not in message:
            raise

        try:
            from torch.serialization import add_safe_globals  # type: ignore

            import pathlib

            add_safe_globals([
                pathlib.Path,
                pathlib.PosixPath,
                pathlib.WindowsPath,
                pathlib.PurePosixPath,
                pathlib.PureWindowsPath,
            ])
        except Exception:
            pass

        try:
            return torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=device)


def load_detector(checkpoint_path: Path, device: torch.device):
    checkpoint = _load_checkpoint(checkpoint_path, device)
    state_dict = checkpoint.get("model_state", checkpoint)
    predictor_weight = state_dict["roi_heads.box_predictor.cls_score.weight"]
    num_classes = predictor_weight.shape[0]

    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"Warning: unexpected keys in detector checkpoint: {unexpected}")
    if missing:
        print(f"Warning: missing keys when loading detector checkpoint: {missing}")
    model.to(device)
    model.eval()
    return model


def load_reid_model(checkpoint_path: Path, device: torch.device) -> SiameseNetwork:
    checkpoint = _load_checkpoint(checkpoint_path, device)
    cfg_dict = checkpoint.get("config", {})
    model = SiameseNetwork()
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def preprocess_image(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    return transform(image)


def detections_to_numpy(detections: Dict[str, torch.Tensor], threshold: float):
    scores = detections["scores"].cpu().numpy()
    keep = scores >= threshold
    boxes = detections["boxes"][keep].cpu().numpy()
    scores = scores[keep]
    return boxes, scores


def crop_embeddings(
    reid_model: SiameseNetwork,
    image: Image.Image,
    boxes: np.ndarray,
    *,
    device: torch.device,
    crop_size: int,
    context_scale: float,
) -> np.ndarray:
    crops = []
    transform = transforms.Compose(
        [
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
        ]
    )
    for box in boxes:
        crop = _crop_with_context(image, box, context_scale)
        crops.append(transform(crop))
    if not crops:
        return np.empty((0, 128))
    batch = torch.stack(crops).to(device)
    with torch.no_grad():
        embeddings = reid_model.forward_once(batch)
    return embeddings.cpu().numpy()


def _crop_with_context(image: Image.Image, box: np.ndarray, scale: float) -> Image.Image:
    left, top, right, bottom = box
    width = right - left
    height = bottom - top
    cx = left + width / 2.0
    cy = top + height / 2.0
    half_w = width * scale / 2.0
    half_h = height * scale / 2.0
    new_left = max(0.0, cx - half_w)
    new_top = max(0.0, cy - half_h)
    new_right = min(image.width, cx + half_w)
    new_bottom = min(image.height, cy + half_h)
    return image.crop((new_left, new_top, new_right, new_bottom))


def run_tracker(cfg: TrackerConfig) -> None:
    device = torch.device(cfg.device) if cfg.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    detector = load_detector(cfg.detector_checkpoint, device)
    reid_model = load_reid_model(cfg.reid_checkpoint, device)

    sequences = discover_sequences(cfg.images_root, cfg.sequences)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    for sequence in sequences:
        run_sequence(detector, reid_model, cfg, sequence, device)


def smooth_bbox(previous: Optional[np.ndarray], current: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if previous is None or alpha >= 0.999:
        return current.copy()
    return alpha * current + (1.0 - alpha) * previous


def discover_sequences(root: Path, sequences: Optional[Sequence[str]]) -> List[Path]:
    root = Path(root)
    if not sequences:
        return sorted(p.parent for p in root.glob("**/img1"))

    resolved: List[Path] = []
    for seq in sequences:
        seq_path = Path(seq)
        candidates = [seq_path, root / seq_path, root.parent / seq_path]
        found = None
        for candidate in candidates:
            if candidate.exists():
                found = candidate
                break
        if found is None:
            raise FileNotFoundError(f"Could not locate sequence '{seq}' relative to {root}")
        resolved.append(found)
    return resolved


def run_sequence(
    detector,
    reid_model,
    cfg: TrackerConfig,
    sequence_dir: Path,
    device: torch.device,
):
    track_id_counter = 0
    active_tracks: List[Track] = []
    outputs: List[Dict[str, float]] = []

    frames = sorted((sequence_dir / "img1").glob("*.jpg"))
    for frame_idx, frame_path in enumerate(frames, start=1):
        image = Image.open(frame_path).convert("RGB")
        input_tensor = preprocess_image(image).to(device)
        with torch.no_grad():
            detections = detector([input_tensor])[0]
        boxes, scores = detections_to_numpy(detections, cfg.detection_threshold)
        embeddings = crop_embeddings(
            reid_model,
            image,
            boxes,
            device=device,
            crop_size=cfg.crop_size,
            context_scale=cfg.context_scale,
        )

        track_embeddings = np.array([track.embedding for track in active_tracks])
        track_boxes = np.array([track.bbox for track in active_tracks])
        iou_matrix = compute_iou_matrix(track_boxes, boxes)
        matches, unmatched_tracks, unmatched_detections = associate_detections(
            track_embeddings,
            embeddings,
            max_distance=cfg.max_distance,
            iou_matrix=iou_matrix,
            iou_weight=cfg.iou_weight,
        )

        for track_idx, det_idx in matches:
            track = active_tracks[track_idx]
            track.embedding = embeddings[det_idx]
            track.bbox = boxes[det_idx]
            track.last_frame = frame_idx
            track.hits += 1
            track.age = 0

            track.smoothed_bbox = smooth_bbox(
                track.smoothed_bbox,
                boxes[det_idx],
                cfg.smoothing_alpha,
            )

            outputs.append(
                make_output(
                    track.track_id,
                    frame_idx,
                    track.smoothed_bbox if track.smoothed_bbox is not None else boxes[det_idx],
                    scores[det_idx],
                )
            )

        for track_idx in unmatched_tracks:
            track = active_tracks[track_idx]
            track.age += 1

        active_tracks = [t for t in active_tracks if t.age <= cfg.max_track_age]

        for det_idx in unmatched_detections:
            track_id_counter += 1
            new_track = Track(
                track_id=track_id_counter,
                embedding=embeddings[det_idx],
                bbox=boxes[det_idx],
                last_frame=frame_idx,
                smoothed_bbox=boxes[det_idx].copy(),
            )
            active_tracks.append(new_track)
            outputs.append(
                make_output(
                    new_track.track_id,
                    frame_idx,
                    new_track.smoothed_bbox,
                    scores[det_idx],
                )
            )

    output_path = cfg.output_dir / f"{sequence_dir.name}_tracks.json"
    output_path.write_text(json.dumps(outputs, indent=2), encoding="utf-8")


def make_output(track_id: int, frame_idx: int, box: np.ndarray, score: float) -> Dict[str, float]:
    left, top, right, bottom = box
    width = right - left
    height = bottom - top
    return {
        "frame": frame_idx,
        "track_id": track_id,
        "bbox": [float(left), float(top), float(width), float(height)],
        "score": float(score),
    }


def compute_iou_matrix(tracks: np.ndarray, detections: np.ndarray) -> np.ndarray:
    if len(tracks) == 0 or len(detections) == 0:
        return np.empty((len(tracks), len(detections)))
    iou_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    for i, track_box in enumerate(tracks):
        for j, det_box in enumerate(detections):
            iou_matrix[i, j] = compute_iou(track_box, det_box)
    return iou_matrix


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def parse_args(argv: Optional[Sequence[str]] = None) -> TrackerConfig:
    parser = argparse.ArgumentParser(description="Run detector + Re-ID tracking")
    parser.add_argument("images_root", type=Path)
    parser.add_argument("annotations_root", type=Path)
    parser.add_argument("detector_checkpoint", type=Path)
    parser.add_argument("reid_checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tracks"))
    parser.add_argument("--sequences", nargs="*", default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--detection-threshold", type=float, default=0.5)
    parser.add_argument("--max-track-age", type=int, default=30)
    parser.add_argument("--max-distance", type=float, default=0.5)
    parser.add_argument("--iou-weight", type=float, default=0.3)
    parser.add_argument("--context-scale", type=float, default=1.2)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--smoothing-alpha", type=float, default=0.6)

    args = parser.parse_args(argv)
    return TrackerConfig(
        images_root=args.images_root,
        annotations_root=args.annotations_root,
        detector_checkpoint=args.detector_checkpoint,
        reid_checkpoint=args.reid_checkpoint,
        output_dir=args.output_dir,
        sequences=args.sequences,
        device=args.device,
        detection_threshold=args.detection_threshold,
        max_track_age=args.max_track_age,
        max_distance=args.max_distance,
        iou_weight=args.iou_weight,
        context_scale=args.context_scale,
        crop_size=args.crop_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        smoothing_alpha=args.smoothing_alpha,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg = parse_args(argv)
    run_tracker(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
