# MOT16 Tracking Pipeline

Minimal MOT16 reproduction recipe: preprocess the dataset, fine-tune a Faster R-CNN detector, train a Siamese Re-ID encoder, then run the tracker (with optional smoothing and overlays). Tested on macOS Sonoma (Apple Silicon + Intel) and Windows 11; Linux users can follow the macOS shell commands.

---

## 1. Dataset
- Download MOT16 from https://motchallenge.net/data/MOT16/
- Place the archive in the project root as `MOT16.zip` (or under `data/`)
- Extract to `data/MOT16/`

---

## 2. Environment

### macOS / Linux (bash/zsh)
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> `requirements.txt` installs CPU/Metal wheels for PyTorch. If you have CUDA, reinstall `torch` and `torchvision` with the matching index URL for your GPU.

---

## 3. Data Preparation

### macOS / Linux
```bash
unzip MOT16.zip -d data/MOT16
python -m src.data.build_dataset_index data/MOT16 --output data/processed_annotations
```

### Windows (PowerShell)
```powershell
Expand-Archive MOT16.zip -DestinationPath data\MOT16
python -m src.data.build_dataset_index data/MOT16 --output data/processed_annotations
```

This creates JSON annotations under `data/processed_annotations/.../frames/` plus a `summary.json`.

---

## 4. Training

### 4.1 Fine-tune Faster R-CNN
```bash
# macOS / Linux (swap --device mps for cuda/cpu as needed)
python -m src.training.detector_trainer \
    data/MOT16/train \
    data/processed_annotations \
    --output-dir outputs/detector \
    --epochs 12 \
    --batch-size 2 \
    --lr 0.005 \
    --val-fraction 0.1 \
    --device mps
```
```powershell
# Windows
python -m src.training.detector_trainer `
    data/MOT16/train `
    data/processed_annotations `
    --output-dir outputs/detector `
    --epochs 12 `
    --batch-size 2 `
    --lr 0.005 `
    --val-fraction 0.1 `
    --device cuda
```
Outputs: checkpoints in `outputs/detector/`, loss curves in `training_history.json` (best val loss ≈ 0.88 at epoch 12).

### 4.2 Train the Siamese Re-ID Encoder
```bash
python -m src.training.reid_trainer \
    data/MOT16/train \
    data/processed_annotations \
    --output-dir outputs/reid \
    --pairs-per-epoch 10000 \
    --val-pairs 2000 \
    --epochs 20 \
    --batch-size 128 \
    --device mps
```
```powershell
python -m src.training.reid_trainer `
    data/MOT16/train `
    data/processed_annotations `
    --output-dir outputs/reid `
    --pairs-per-epoch 10000 `
    --val-pairs 2000 `
    --epochs 20 `
    --batch-size 128 `
    --device cuda
```
Check `outputs/reid/reid_training_history.json`; best val loss ≈ 0.062 with positive/negative distances ≈ 0.20 / 0.98.

---

## 5. Tracking & Overlay
```bash
python -m src.tracking.inference \
    data/MOT16/train \
    data/processed_annotations \
    outputs/detector/detector_epoch_012.pth \
    outputs/reid/best_reid_model.pth \
    --output-dir outputs/tracks \
    --sequences MOT16-02 \
    --device mps \
    --detection-threshold 0.68 \
    --max-distance 0.30 \
    --iou-weight 0.6 \
    --max-track-age 18 \
    --smoothing-alpha 0.5
```
```powershell
python -m src.tracking.inference `
    data/MOT16/train `
    data/processed_annotations `
    outputs/detector/detector_epoch_012.pth `
    outputs/reid/best_reid_model.pth `
    --output-dir outputs/tracks `
    --sequences MOT16-02 `
    --device cuda `
    --detection-threshold 0.68 `
    --max-distance 0.30 `
    --iou-weight 0.6 `
    --max-track-age 18 `
    --smoothing-alpha 0.5
```

Generate an MP4 overlay (optional but helpful):
```bash
python scripts/render_tracks.py \
    data/MOT16/train/MOT16-02/img1 \
    outputs/tracks/MOT16-02_tracks.json \
    outputs/tracks/MOT16-02_overlay.mp4 --fps 10
```
Same command works on Windows PowerShell. If boxes jitter, bump `--detection-threshold`, lower `--max-distance`, or tweak `--smoothing-alpha` and rerun the tracker + overlay.

---

## 6. Snapshot
- Detector best validation loss ≈ 0.88 (epoch 12)
- Re-ID best validation loss ≈ 0.062 (epoch 11), positive/negative distances ≈ 0.20 / 0.98
- MOT16-02 tracking: 600 frames processed, ~12 tracks per frame (range 6–19), 285 unique IDs

## 7. Repo Layout
- `src/data/` — dataset parsing, transforms, overlay helpers
- `src/training/` — detector + Re-ID trainers
- `src/tracking/` — association, smoothing, inference CLI
- `scripts/render_tracks.py` — MP4 overlay renderer
- `docs_detector_training.md` — extended notes / reproduction log
- `outputs/` — generated artifacts (git-ignored; regenerate using the steps above)

