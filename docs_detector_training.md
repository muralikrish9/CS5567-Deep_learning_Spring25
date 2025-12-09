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

## What We Actually Ran

I fine-tuned `fasterrcnn_resnet50_fpn` on MOT16 with the defaults above, but on GPU (`--device cuda`) and let it run for the full 12 epochs. Validation loss settled around **0.88** and the training loss ended at **0.92**; both numbers come from `outputs/detector/training_history.json`. That’s roughly where the curve flattened, so more epochs would be diminishing returns unless we bring in extra regularisation or stronger augmentation.

For person re-identification I trained the Siamese encoder for 20 epochs. The best validation contrastive loss was **0.062** (epoch 11) with positive-pair distances ≈0.20 and negative ≈0.98. Those ratios are healthy enough that the tracker can lean on the embeddings without constantly second guessing itself.

During tracking I stuck to MOT16-02 as the tuning playground. With the latest settings the tracker touches all 600 frames, averages ~12 tracks per frame (min 6, max 19) and records 285 unique IDs. That’s the configuration that produces the `outputs/tracks/MOT16-02_overlay.mp4` demo.

## Reproduce the Full Pipeline

```bash
# 1. Parse raw MOT annotations into per-frame JSON
python -m src.data.build_dataset_index data/MOT16 --output data/processed_annotations

# 2. Fine-tune Faster R-CNN (12 epochs on GPU)
python -m src.training.detector_trainer \
    data/MOT16/train \
    data/processed_annotations \
    --output-dir outputs/detector \
    --epochs 12 \
    --batch-size 2 \
    --lr 0.005 \
    --val-fraction 0.1 \
    --device cuda

# 3. Train the Siamese Re-ID network (pairs drawn on the fly)
python -m src.training.reid_trainer \
    data/MOT16/train \
    data/processed_annotations \
    --output-dir outputs/reid \
    --pairs-per-epoch 10000 \
    --val-pairs 2000 \
    --epochs 20 \
    --batch-size 128 \
    --device cuda

# 4. Run tracking with tuned thresholds and box smoothing
python -m src.tracking.inference \
    data/MOT16/train \
    data/processed_annotations \
    outputs/detector/detector_epoch_012.pth \
    outputs/reid/best_reid_model.pth \
    --output-dir outputs/tracks \
    --sequences MOT16-02 \
    --device cuda \
    --detection-threshold 0.80 \
    --max-distance 0.23 \
    --iou-weight 0.75 \
    --max-track-age 32 \
    --smoothing-alpha 0.80 \
    --context-scale 1.30 \
    --reactivation-distance 0.30

# 5. Render a sanity-check overlay (requires opencv-python)
python scripts/render_tracks.py \
    data/MOT16/train/MOT16-02/img1 \
    outputs/tracks/MOT16-02_tracks.json \
    outputs/tracks/MOT16-02_overlay.mp4 \
    --fps 10
```

Use the last command as a qualitative gut check; if the boxes jitter, nudge `--detection-threshold` down a bit (e.g., 0.75) or increase `--smoothing-alpha`. Only re-enable unmatched emission or lengthen `--max-track-age` if you prefer persistence over ghost suppression.

## Report Notes (for Future Self)

- Methodology section should call out: detector freeze strategy, Siamese architecture, the association cost (cosine + IOU) and the smoothing we layered on as a last-mile fix.
- Results section ought to carry the loss numbers above, plus a short qualitative note about MOT16-02 coverage (~12 tracks per frame, 285 IDs) and the overlay video as evidence.
- Challenges worth mentioning: PowerShell line continuation headaches, PyTorch 2.6’s `torch.load` safety changes, the amount of time spent chasing down jitter.
- Self-eval: detector is competent, Re-ID embedding separation looks solid, tracker still lacks a proper quantitative metric (no MOTA/IDF1 yet) so call that out as future work.
