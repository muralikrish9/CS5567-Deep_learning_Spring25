"""Train a Siamese Re-ID network on MOT-style person crops."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.reid_dataset import MOTReIDPatchDataset, ReIDPairDataset


@dataclass
class ReIDTrainingConfig:
    images_root: Path
    annotations_root: Path
    output_dir: Path
    sequences: Optional[List[str]] = None
    context_scale: float = 1.2
    min_visibility: float = 0.0
    crop_size: int = 128
    positive_fraction: float = 0.5
    pairs_per_epoch: int = 10000
    val_pairs: int = 2000
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    margin: float = 1.0
    val_fraction: float = 0.2
    num_workers: int = 4
    device: Optional[str] = None
    seed: int = 42


class SiameseNetwork(nn.Module):
    """Simple convolutional backbone producing 128-d embeddings."""

    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embedding_dim),
        )

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        return self.embedding(feats)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_once(x1), self.forward_once(x2)


class ContrastiveLoss(nn.Module):
    """Classic contrastive loss for pairwise metric learning."""

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        distances = torch.nn.functional.pairwise_distance(embedding1, embedding2)
        positive_loss = label * distances.pow(2)
        negative_loss = (1.0 - label) * torch.clamp(self.margin - distances, min=0.0).pow(2)
        return (positive_loss + negative_loss).mean()


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_preference: Optional[str] = None) -> torch.device:
    if device_preference:
        requested = device_preference.lower()
        if requested == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if requested == "cpu":
            return torch.device("cpu")
        raise ValueError(f"Requested device '{device_preference}' is not available")

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_transforms(crop_size: int) -> transforms.Compose:
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return transforms.Compose(
        [
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def split_identities(
    dataset: MOTReIDPatchDataset,
    *,
    val_fraction: float,
    seed: int,
) -> Tuple[MOTReIDPatchDataset, Optional[MOTReIDPatchDataset]]:
    identities = list(dataset.identity_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(identities)
    if not identities:
        raise ValueError("Dataset contains no identities to split")

    if val_fraction <= 0.0 or len(identities) <= 1:
        return dataset, None

    val_count = int(math.floor(len(identities) * val_fraction))
    val_count = max(1, min(val_count, len(identities) - 1))
    val_identities = identities[:val_count]
    train_identities = identities[val_count:]
    train_dataset = dataset.spawn_subset(train_identities)
    val_dataset = dataset.spawn_subset(val_identities)
    return train_dataset, val_dataset


def create_pair_loader(
    dataset: MOTReIDPatchDataset,
    *,
    pairs: int,
    positive_fraction: float,
    batch_size: int,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    pair_dataset = ReIDPairDataset(
        dataset,
        length=pairs,
        positive_fraction=positive_fraction,
        seed=seed,
    )
    return DataLoader(
        pair_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    steps = 0
    start = time.time()

    for batch in loader:
        img1, img2, labels, _ = batch
        img1 = img1.to(device)
        img2 = img2.to(device)
        labels = labels.to(device)

        embedding1, embedding2 = model(img1, img2)
        loss = criterion(embedding1, embedding2, labels)
        loss_value = loss.detach().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss_value
        steps += 1

    elapsed = time.time() - start
    return {
        "loss": total_loss / max(1, steps),
        "time": elapsed,
        "steps": steps,
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    steps = 0
    pos_distances: List[float] = []
    neg_distances: List[float] = []

    with torch.no_grad():
        for img1, img2, labels, _ in loader:
            img1 = img1.to(device)
            img2 = img2.to(device)
            labels = labels.to(device)

            embedding1, embedding2 = model(img1, img2)
            loss = criterion(embedding1, embedding2, labels)
            total_loss += float(loss.item())
            steps += 1

            distances = torch.nn.functional.pairwise_distance(embedding1, embedding2)
            pos_mask = labels > 0.5
            neg_mask = labels <= 0.5
            if pos_mask.any():
                pos_distances.append(float(distances[pos_mask].mean().item()))
            if neg_mask.any():
                neg_distances.append(float(distances[neg_mask].mean().item()))

    return {
        "loss": total_loss / max(1, steps),
        "pos_distance": float(np.mean(pos_distances)) if pos_distances else float("nan"),
        "neg_distance": float(np.mean(neg_distances)) if neg_distances else float("nan"),
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: ReIDTrainingConfig,
    epoch: int,
    metrics: Dict[str, float],
    path: Path,
) -> None:
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": cfg.__dict__,
        "metrics": metrics,
    }
    torch.save(payload, path)


def run_training(cfg: ReIDTrainingConfig) -> None:
    set_determinism(cfg.seed)
    device = resolve_device(cfg.device)
    print(f"Using device: {device}")

    transform = build_transforms(cfg.crop_size)
    base_dataset = MOTReIDPatchDataset(
        images_root=cfg.images_root,
        annotations_root=cfg.annotations_root,
        sequences=cfg.sequences,
        transform=transform,
        context_scale=cfg.context_scale,
        min_visibility=cfg.min_visibility,
    )

    train_dataset, val_dataset = split_identities(
        base_dataset,
        val_fraction=cfg.val_fraction,
        seed=cfg.seed,
    )

    train_loader = create_pair_loader(
        train_dataset,
        pairs=cfg.pairs_per_epoch,
        positive_fraction=cfg.positive_fraction,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        pin_memory=device.type == "cuda",
    )
    val_loader = None
    if val_dataset is not None and cfg.val_pairs > 0:
        val_loader = create_pair_loader(
            val_dataset,
            pairs=cfg.val_pairs,
            positive_fraction=cfg.positive_fraction,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            seed=cfg.seed + 1,
            pin_memory=device.type == "cuda",
        )

    model = SiameseNetwork().to(device)
    criterion = ContrastiveLoss(margin=cfg.margin)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    history: List[Dict[str, float]] = []
    best_val = float("inf")
    best_path: Optional[Path] = None

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        record: Dict[str, float] = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
        }
        print(
            f"Epoch {epoch}: train loss {train_metrics['loss']:.4f}"
            f" ({train_metrics['steps']} steps, {train_metrics['time']:.1f}s)"
        )

        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, criterion, device)
            record.update(
                {
                    "val_loss": val_metrics["loss"],
                    "val_pos_distance": val_metrics["pos_distance"],
                    "val_neg_distance": val_metrics["neg_distance"],
                }
            )
            print(
                "  Validation: loss {loss:.4f}, pos_dist {pos:.3f}, neg_dist {neg:.3f}".format(
                    loss=val_metrics["loss"],
                    pos=val_metrics["pos_distance"],
                    neg=val_metrics["neg_distance"],
                )
            )
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                best_path = cfg.output_dir / "best_reid_model.pth"
                save_checkpoint(model, optimizer, cfg, epoch, record, best_path)
        else:
            # Without validation we keep the latest checkpoint
            best_path = cfg.output_dir / "latest_reid_model.pth"
            save_checkpoint(model, optimizer, cfg, epoch, record, best_path)

        history.append(record)

        latest_path = cfg.output_dir / "last_reid_model.pth"
        save_checkpoint(model, optimizer, cfg, epoch, record, latest_path)

    history_path = cfg.output_dir / "reid_training_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    if best_path is not None:
        print(f"Saved best checkpoint to {best_path.as_posix()}")


def parse_args(argv: Optional[Sequence[str]] = None) -> ReIDTrainingConfig:
    parser = argparse.ArgumentParser(description="Train a Siamese Re-ID network")
    parser.add_argument("images_root", type=Path, help="Path to MOT images root (e.g. data/train)")
    parser.add_argument(
        "annotations_root",
        type=Path,
        help="Path to processed annotations (e.g. data/processed_annotations)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reid"),
        help="Directory to store checkpoints and logs",
    )
    parser.add_argument(
        "--sequences",
        nargs="*",
        default=None,
        help="Optional list of sequences to include",
    )
    parser.add_argument("--context-scale", type=float, default=1.2)
    parser.add_argument("--min-visibility", type=float, default=0.0)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--positive-fraction", type=float, default=0.5)
    parser.add_argument("--pairs-per-epoch", type=int, default=10000)
    parser.add_argument("--val-pairs", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default=None,
        help="Force a specific device",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args(argv)
    return ReIDTrainingConfig(
        images_root=args.images_root,
        annotations_root=args.annotations_root,
        output_dir=args.output_dir,
        sequences=args.sequences,
        context_scale=args.context_scale,
        min_visibility=args.min_visibility,
        crop_size=args.crop_size,
        positive_fraction=args.positive_fraction,
        pairs_per_epoch=args.pairs_per_epoch,
        val_pairs=args.val_pairs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        margin=args.margin,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
    )


def main() -> None:
    cfg = parse_args()
    run_training(cfg)


if __name__ == "__main__":  # pragma: no cover
    try:
        import torch.multiprocessing as mp

        mp.freeze_support()
    except (ImportError, AttributeError):
        pass
    main()
