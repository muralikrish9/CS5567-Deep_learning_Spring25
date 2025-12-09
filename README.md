# MOT16 Tracking Pipeline

End-to-end MOT16 pipeline: preprocess data, fine-tune Faster R-CNN, train a Siamese Re-ID encoder, and run tracking with smoothing and reactivation. Tested on macOS, Windows 11, and Linux.

- **Single-target demo (MOT16 test, ID 7):** [`outputs/tracks_single/MOT16-07_track7_overlay.mp4`](outputs/tracks_single/MOT16-07_track7_overlay.mp4) (~62 MB)
- **Final report (PDF):** [`MOT_project (1).pdf`](MOT_project%20(1).pdf)

> Note: Model weights (.pth) and raw MOT16 data are not committed. Recreate them via the steps below.

---

## Quickstart (tracking only)
```bash
# 1) Create environment
python -m venv .venv && source .venv/bin/activate        # PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# 2) Prepare data indices (after unzipping MOT16 to data/MOT16)
python -m src.data.build_dataset_index data/MOT16 --output data/processed_annotations

# 3) Run tracking on a test sequence (multi-target)
python -m src.tracking.inference \
    data/MOT16/test \
    data/processed_annotations \
    outputs/detector/detector_epoch_003.pth \
    outputs/reid/best_reid_model.pth \
    --output-dir outputs/tracks \
    --sequences MOT16-07 \
    --detection-threshold 0.80 \
    --max-distance 0.23 \
    --iou-weight 0.75 \
    --max-track-age 32 \
    --smoothing-alpha 0.80 \
    --context-scale 1.30 \
    --reactivation-distance 0.30

# 4) Render overlay
python scripts/render_tracks.py \
    data/MOT16/test/MOT16-07/img1 \
    outputs/tracks/MOT16-07_tracks.json \
    outputs/tracks/MOT16-07_overlay.mp4 \
    --fps 10
```

---

## Environment
```bash
python -m venv .venv
source .venv/bin/activate       # PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If you have CUDA, reinstall `torch`/`torchvision` with the matching CUDA wheels. Metal (mps) works; CPU is fine for short runs.

## Data
1) Download MOT16 from https://motchallenge.net/data/MOT16/ and place `MOT16.zip` in the repo (or `data/`).  
2) Extract to `data/MOT16/`.  
   - macOS/Linux: `unzip MOT16.zip -d data/MOT16`  
   - Windows: `Expand-Archive MOT16.zip -DestinationPath data\MOT16`  
3) Build indices: `python -m src.data.build_dataset_index data/MOT16 --output data/processed_annotations`

## Training (optional if you already have weights)
- Detector (Faster R-CNN): see `docs_detector_training.md` for full notes. Example:
```bash
python -m src.training.detector_trainer \
    data/MOT16/train \
    data/processed_annotations \
    --output-dir outputs/detector \
    --epochs 12 --batch-size 2 --lr 0.005 --val-fraction 0.1 \
    --device cuda
```
- Re-ID (Siamese encoder):
```bash
python -m src.training.reid_trainer \
    data/MOT16/train \
    data/processed_annotations \
    --output-dir outputs/reid \
    --pairs-per-epoch 10000 --val-pairs 2000 --epochs 20 --batch-size 128 \
    --device cuda
```

## Tracking (multi-target)
Defaults tuned to reduce ghost boxes: `--detection-threshold 0.80`, `--max-distance 0.23`, `--max-track-age 32`, `--smoothing-alpha 0.80`, `--reactivation-distance 0.30`, `--emit-unmatched` disabled by default. Backends: Faster R-CNN (default) or YOLOv8 (`--detector-backend yolov8`).
```bash
python -m src.tracking.inference \
    data/MOT16/test \
    data/processed_annotations \
    outputs/detector/detector_epoch_003.pth \
    outputs/reid/best_reid_model.pth \
    --output-dir outputs/tracks \
    --sequences MOT16-07 \
    --detector-backend fasterrcnn \
    --detection-threshold 0.80 \
    --max-distance 0.23 \
    --iou-weight 0.75 \
    --max-track-age 32 \
    --smoothing-alpha 0.80 \
    --context-scale 1.30 \
    --reactivation-distance 0.30
```
Render an overlay:
```bash
python scripts/render_tracks.py \
    data/MOT16/test/MOT16-07/img1 \
    outputs/tracks/MOT16-07_tracks.json \
    outputs/tracks/MOT16-07_overlay.mp4 \
    --fps 10
```
Tuning tips: if ghosts appear, raise `--detection-threshold` (e.g., 0.82) or lower `--reactivation-distance` (e.g., 0.25). If IDs drop too quickly, nudge `--max-track-age` upward modestly.

### Using YOLOv8 detector
Install dependency (already in `requirements.txt`): `pip install ultralytics`. Then run:
```bash
python -m src.tracking.inference \
    data/MOT16/test \
    data/processed_annotations \
    path/to/yolov8n.pt \
    outputs/reid/best_reid_model.pth \
    --output-dir outputs/tracks \
    --sequences MOT16-07 \
    --detector-backend yolov8 \
    --person-class 0 \
    --detection-threshold 0.25 \
    --max-distance 0.23 \
    --iou-weight 0.75 \
    --max-track-age 32 \
    --smoothing-alpha 0.80 \
    --context-scale 1.30 \
    --reactivation-distance 0.30
```
Adjust `--person-class` if your YOLO weights use a different label mapping. Lower the detection threshold for smaller YOLO models; increase if ghosts appear.

## Single-target demo (how this repo’s MP4 was made)
1) Run inference (as above) on a MOT16 **test** sequence.  
2) Pick a stable track_id (e.g., 7 for MOT16-07 in our run):
```bash
python - <<'PY'
import json, collections
data=json.load(open("outputs/tracks/MOT16-07_tracks.json"))
print(collections.Counter(d["track_id"] for d in data).most_common(5))
PY
```
3) Filter to that ID and render:
```bash
python - <<'PY'
import json
inp="outputs/tracks/MOT16-07_tracks.json"
out="outputs/tracks/MOT16-07_track7.json"
target=7
json.dump([d for d in json.load(open(inp)) if d["track_id"]==target], open(out,"w"), indent=2)
PY

python scripts/render_tracks.py \
    data/MOT16/test/MOT16-07/img1 \
    outputs/tracks/MOT16-07_track7.json \
    outputs/tracks/MOT16-07_track7_overlay.mp4 \
    --fps 10
```
The resulting MP4 is committed at `outputs/tracks_single/MOT16-07_track7_overlay.mp4`.

## Deliverables included
- Source code (no weights).  
- Single-target test video: `outputs/tracks_single/MOT16-07_track7_overlay.mp4` (ID 7).  
- Final report PDF: `MOT_project (1).pdf`.


