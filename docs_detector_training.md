# Detector Fine-Tuning Guide

This repository now includes a standalone training script for fine-tuning Faster R-CNN on MOT-style datasets.

## Quick Start

1. Ensure you have the processed annotations (run `python -m src.data.build_dataset_index data --output data/processed_annotations`).
2. Activate an environment with `torch>=2.1` and `torchvision>=0.16`.
3. Launch training:

```bash
python -m src.training.detector_trainer \
    data/train \
    data/processed_annotations \
    --output-dir outputs/detector \
    --epochs 12 \
    --batch-size 2 \
    --lr 0.005 \
    --val-fraction 0.1 \
    --device cpu
```

### Common Flags

- `--train-seqs`, `--val-seqs`: explicit sequence filters.
- `--val-fraction`: random split when validation sequences are not specified.
- `--device`: force `cpu`, `cuda`, or `mps` (Metal falls back to CPU for stability).
- `--no-freeze-backbone`: allow the backbone to update (default freezes it for few-shot tuning).
- `--image-min-side`, `--image-max-side`: control resizing bounds during augmentation.
- `--no-amp`: disable mixed precision training.

Checkpoints and training logs (`training_history.json`) are stored in the chosen `--output-dir`.

## Notes

- The script automatically limits MKL/OMP thread usage to improve stability inside constrained containers.
- Validation computes the same multi-task detector losses for monitoring; integrate COCO-style metrics later if needed.
- For longer runs, increase `--num-workers` to take advantage of multi-core dataloading.
