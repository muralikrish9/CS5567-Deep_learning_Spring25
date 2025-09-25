"""Fine-tune Faster R-CNN on MOT-style datasets."""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# Reduce the chance of MKL/OMP shared-memory crashes in constrained sandboxes
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torchvision
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.data.datasets import build_detection_dataloaders


def _create_grad_scaler():
    """Handle PyTorch version differences for AMP GradScaler."""

    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
        return torch.cuda.amp.GradScaler()
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler()
    return None


def _autocast(device: torch.device):
    if device.type == "cuda":
        if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
            return torch.cuda.amp.autocast()
        if hasattr(torch, "autocast"):
            return torch.autocast(device_type="cuda")
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast(device_type="cuda")
    return nullcontext()


@dataclass
class TrainingConfig:
    images_root: Path
    annotations_root: Path
    output_dir: Path
    device: Optional[str] = None
    train_sequences: Optional[List[str]] = None
    val_sequences: Optional[List[str]] = None
    epochs: int = 10
    batch_size: int = 2
    base_lr: float = 0.005
    weight_decay: float = 0.0005
    momentum: float = 0.9
    grad_clip: Optional[float] = 5.0
    amp: bool = True
    val_fraction: Optional[float] = None
    num_workers: int = 0
    min_visibility: float = 0.0
    label_default: int = 1
    image_min_side: int = 600
    image_max_side: int = 1024
    warmup_iters: int = 500
    warmup_factor: float = 0.001
    freeze_backbone: bool = True
    trainable_backbone_layers: int = 2


def create_model(num_classes: int, *, cfg: TrainingConfig) -> nn.Module:
    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=weights,
        trainable_backbone_layers=cfg.trainable_backbone_layers,
    )

    if cfg.freeze_backbone:
        for name, parameter in model.backbone.named_parameters():
            parameter.requires_grad_(False)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader,
    device: torch.device,
    epoch: int,
    scaler: Optional[torch.cuda.amp.GradScaler],
    cfg: TrainingConfig,
    print_freq: int = 20,
) -> Dict[str, float]:
    model.train()
    metric_logger = defaultdict(float)
    iteration = 0
    start = time.time()

    for images, targets in data_loader:
        iteration += 1
        images = [img.to(device) for img in images]
        targets = [
            {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in target.items()
            }
            for target in targets
        ]

        autocast_enabled = scaler is not None and cfg.amp and device.type == "cuda"
        context = _autocast(device) if autocast_enabled else nullcontext()

        with context:
            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())

        loss_value = losses.item()
        if not torch.isfinite(torch.tensor(loss_value)):
            raise RuntimeError(f"Non-finite loss detected: {loss_value}")

        optimizer.zero_grad()
        if scaler is not None and cfg.amp and device.type == "cuda":
            scaler.scale(losses).backward()
            if cfg.grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        for name, value in loss_dict.items():
            metric_logger[name] += float(value.item())
        metric_logger["loss"] += float(loss_value)

        if iteration % print_freq == 0:
            elapsed = time.time() - start
            avg_loss = metric_logger["loss"] / iteration
            print(
                f"Epoch {epoch} Iter {iteration}/{len(data_loader)} | "
                f"Loss: {avg_loss:.4f} | Time/iter: {elapsed/iteration:.3f}s"
            )

    for key in metric_logger:
        metric_logger[key] /= max(1, iteration)

    return metric_logger


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    data_loader,
    device: torch.device,
) -> Dict[str, float]:
    was_training = model.training
    model.train()
    loss_totals = defaultdict(float)
    count = 0

    for images, targets in data_loader:
        count += 1
        images = [img.to(device) for img in images]
        targets = [
            {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in target.items()
            }
            for target in targets
        ]
        loss_dict = model(images, targets)
        for name, value in loss_dict.items():
            loss_totals[name] += float(value.item())
        loss_totals["loss"] += float(sum(loss_dict.values()).item())

    if count > 0:
        for key in loss_totals:
            loss_totals[key] /= count

    if not was_training:
        model.eval()
    return loss_totals


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    cfg: TrainingConfig,
    scheduler=None,
    scaler=None,
    metrics: Optional[Dict[str, float]] = None,
) -> Path:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = cfg.output_dir / f"detector_epoch_{epoch:03d}.pth"
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": cfg.__dict__,
    }
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler_state"] = scaler.state_dict()
    if metrics is not None:
        payload["metrics"] = metrics
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def resolve_device(preferred: Optional[str] = None) -> torch.device:
    if preferred:
        preferred = preferred.lower()
        if preferred == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if preferred == "mps":
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                print("Warning: torchvision detection models have limited MPS support; falling back to CPU.")
            return torch.device("cpu")
        if preferred == "cpu":
            return torch.device("cpu")
        raise ValueError(f"Requested device '{preferred}' is not available")

    if torch.cuda.is_available():
        return torch.device("cuda")

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print(
            "MPS detected but torchvision detection ops are unstable on Metal; using CPU instead."
        )

    return torch.device("cpu")


def run_training(cfg: TrainingConfig) -> None:
    device = resolve_device(cfg.device)
    print(f"Using device: {device}")

    train_loader, val_loader = build_detection_dataloaders(
        images_root=cfg.images_root,
        annotations_root=cfg.annotations_root,
        train_sequences=cfg.train_sequences,
        val_sequences=cfg.val_sequences,
        val_fraction=cfg.val_fraction,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        min_visibility=cfg.min_visibility,
        label_default=cfg.label_default,
        transform_kwargs={
            "image_min_side": cfg.image_min_side,
            "image_max_side": cfg.image_max_side,
            "color_jitter": 0.3,
            "blur_prob": 0.2,
            "hflip_prob": 0.5,
        },
    )

    num_classes = 1 + 20  # MOT16 uses 1-based class IDs up to 20
    model = create_model(num_classes, cfg=cfg)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = SGD(
        params,
        lr=cfg.base_lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.base_lr * 0.1)
    scaler = None
    if cfg.amp and device.type == "cuda":
        scaler = _create_grad_scaler()
        if scaler is None:
            print("Warning: AMP GradScaler unavailable; continuing without mixed precision.")

    history: List[Dict[str, float]] = []

    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        if cfg.warmup_iters > 0 and epoch == 1:
            warmup_factor = cfg.warmup_factor
            warmup_iters = min(cfg.warmup_iters, len(train_loader) - 1)
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=warmup_factor,
                total_iters=warmup_iters,
            )
        else:
            warmup_scheduler = None

        model.train()
        metrics = train_one_epoch(
            model,
            optimizer,
            train_loader,
            device,
            epoch,
            scaler,
            cfg,
        )
        if warmup_scheduler is not None:
            warmup_scheduler.step()

        scheduler.step()

        val_metrics = None
        if val_loader is not None:
            val_metrics = evaluate_loss(model, val_loader, device)
            metrics = {**metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}

        history.append(metrics)
        scaler_to_save = None
        if scaler is not None:
            if not hasattr(scaler, "is_enabled") or scaler.is_enabled():
                scaler_to_save = scaler

        checkpoint_path = save_checkpoint(
            model,
            optimizer,
            epoch,
            cfg,
            scheduler=scheduler,
            scaler=scaler_to_save,
            metrics=metrics,
        )
        print(
            f"Epoch {epoch} complete. Training loss: {metrics['loss']:.4f}. "
            f"Checkpoint saved to {checkpoint_path}."
        )
        if val_metrics is not None:
            print(
                "Validation: "
                + ", ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
            )

    history_path = cfg.output_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Training history saved to {history_path}")


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Fine-tune Faster R-CNN on MOT data")
    parser.add_argument(
        "images_root",
        type=Path,
        help="Path to MOT images root (e.g. data/train)",
    )
    parser.add_argument(
        "annotations_root",
        type=Path,
        help="Path to processed annotations root (e.g. data/processed_annotations)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/detector"),
        help="Directory to store checkpoints and logs",
    )
    parser.add_argument(
        "--train-seqs",
        nargs="*",
        default=None,
        help="List of sequences to use for training (default: all)",
    )
    parser.add_argument(
        "--val-seqs",
        nargs="*",
        default=None,
        help="List of sequences to use for validation",
    )
    parser.add_argument("--val-fraction", type=float, default=None, help="Fraction for random val split if --val-seqs not provided")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision training")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--min-visibility", type=float, default=0.0)
    parser.add_argument("--image-min-side", type=int, default=600)
    parser.add_argument("--image-max-side", type=int, default=1024)
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default=None,
        help="Force a specific device (default auto-detect; mps falls back to CPU).",
    )
    parser.add_argument(
        "--no-freeze-backbone",
        action="store_true",
        help="Allow backbone layers to update (default freezes backbone)",
    )
    parser.add_argument("--trainable-backbone-layers", type=int, default=2)

    args = parser.parse_args()

    cfg = TrainingConfig(
        images_root=args.images_root,
        annotations_root=args.annotations_root,
        output_dir=args.output_dir,
        train_sequences=args.train_seqs,
        val_sequences=args.val_seqs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        base_lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        grad_clip=None if args.grad_clip <= 0 else args.grad_clip,
        amp=not args.no_amp,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        min_visibility=args.min_visibility,
        image_min_side=args.image_min_side,
        image_max_side=args.image_max_side,
        freeze_backbone=not args.no_freeze_backbone,
        trainable_backbone_layers=args.trainable_backbone_layers,
        device=args.device,
    )

    return cfg


def main() -> None:
    cfg = parse_args()
    run_training(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
