import json
import shutil
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
DATA_DIR = TESTS_DIR / "data"

import numpy as np
from PIL import Image

from src.data import (
    MotAnnotation,
    annotations_by_frame,
    build_dataset_index,
    decode_rle_mask,
    find_gt_files,
    generate_dataset_overlays,
    generate_sequence_overlays,
    parse_mot_annotations,
    render_annotations_overlay,
)
from src.data.generate_overlays import frame_ids_to_sample


def encode_rle(mask: np.ndarray) -> str:
    flat = mask.reshape(-1, order="F")
    counts = []
    current_value = 0
    run_length = 0
    for value in flat:
        if value == current_value:
            run_length += 1
        else:
            counts.append(run_length)
            run_length = 1
            current_value = value
    counts.append(run_length)
    return " ".join(str(count) for count in counts)


def test_parse_mot_annotations():
    annotations = parse_mot_annotations(DATA_DIR / "sample_gt.txt", include_masks=True)
    assert len(annotations) == 2

    first = annotations[0]
    assert first.frame_id == 1
    assert first.object_id == 1
    assert first.bbox == (100.0, 150.0, 40.0, 80.0)
    assert first.confidence == 1.0
    assert first.class_id == 1
    assert first.visibility == 0.9
    assert first.mask_rle == "3 4 5 6 7"

    second = annotations[1]
    assert second.frame_id == 2
    assert second.class_id == -1
    assert second.mask_rle is None


def test_decode_rle_mask():
    rle = "1 4 4"
    mask = decode_rle_mask(rle, height=3, width=3)

    expected_flat = np.zeros(9, dtype=np.uint8)
    expected_flat[1:5] = 1
    expected = expected_flat.reshape((3, 3), order="F")

    assert mask.shape == (3, 3)
    assert np.array_equal(mask, expected)


def test_annotations_by_frame():
    annotations = [
        MotAnnotation(1, 1, (0, 0, 10, 10), 1.0),
        MotAnnotation(1, 2, (5, 5, 10, 10), 1.0),
        MotAnnotation(2, 1, (0, 0, 5, 5), 1.0),
    ]
    grouped = annotations_by_frame(annotations)
    assert list(grouped.keys()) == [1, 2]
    assert len(grouped[1]) == 2
    assert len(grouped[2]) == 1


def test_render_annotations_overlay(tmp_path):
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(image_path)

    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 1
    mask_rle = encode_rle(mask)

    annotation = MotAnnotation(
        frame_id=1,
        object_id=7,
        bbox=(2, 2, 4, 4),
        confidence=1.0,
        mask_rle=mask_rle,
    )

    output_path = tmp_path / "overlay.png"
    rendered = render_annotations_overlay(image_path, [annotation], output_path=output_path)

    assert output_path.exists()
    rendered_array = np.array(rendered)
    assert rendered_array.mean() > 0


def test_frame_ids_to_sample():
    assert frame_ids_to_sample(10, stride=3) == [1, 4, 7, 10]
    assert frame_ids_to_sample(5, stride=10) == [1, 5]


def _create_dummy_sequence(root: Path) -> Path:
    seq_dir = root / "train" / "SEQ"
    img_dir = seq_dir / "img1"
    gt_dir = seq_dir / "gt"
    img_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)

    for idx in range(1, 6):
        image = Image.new("RGB", (8, 8), color=(idx * 20, 0, 0))
        image.save(img_dir / f"{idx:06d}.jpg")

    gt_lines = [
        f"{frame},1,1,1,3,3,1" for frame in range(1, 6)
    ]
    (gt_dir / "gt.txt").write_text("\n".join(gt_lines), encoding="utf-8")
    return seq_dir


def test_generate_sequence_overlays(tmp_path):
    seq_dir = _create_dummy_sequence(tmp_path)
    output = tmp_path / "overlays"
    written = generate_sequence_overlays(seq_dir, output, stride=2)
    assert written == 3
    assert len(list(output.glob("*.jpg"))) == 3


def test_generate_dataset_overlays(tmp_path):
    seq_dir = _create_dummy_sequence(tmp_path)
    dataset_root = seq_dir.parent.parent
    output_root = tmp_path / "dataset_overlays"
    written = generate_dataset_overlays(dataset_root, output_root, stride=2)
    assert written == 3
    assert len(list(output_root.glob("train/SEQ/*.jpg"))) == 3


def test_build_dataset_index(tmp_path):
    dataset_root = tmp_path / "mot"
    seq_dir = dataset_root / "train" / "TEST_SEQ"
    gt_dir = seq_dir / "gt"
    gt_dir.mkdir(parents=True)
    shutil.copy(DATA_DIR / "sample_gt.txt", gt_dir / "gt.txt")

    output_root = tmp_path / "processed"
    summary = build_dataset_index(dataset_root, output_root, include_masks=True)

    assert "train/TEST_SEQ" in summary
    frames_dir = output_root / "train" / "TEST_SEQ" / "frames"
    first_frame = frames_dir / "000001.json"
    assert first_frame.exists()

    frame_payload = json.loads(first_frame.read_text(encoding="utf-8"))
    assert frame_payload[0]["object_id"] == 1
    assert frame_payload[0]["mask_rle"] == "3 4 5 6 7"

    discovered = find_gt_files(dataset_root)
    assert len(discovered) == 1
