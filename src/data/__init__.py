"""Data processing utilities for the MOT/MOTS project."""

from .build_dataset_index import build_dataset_index, find_gt_files
from .generate_overlays import (
    generate_dataset_overlays,
    generate_sequence_overlays,
)
from .mots_parser import (
    MotAnnotation,
    annotations_by_frame,
    decode_rle_mask,
    mot_annotations_to_dataframe,
    parse_mot_annotations,
    render_annotations_overlay,
)
try:  # Optional dependency (torch)
    from .reid_dataset import (
        MOTReIDPatchDataset,
        PatchRecord,
        ReIDPairDataset,
    )
except ModuleNotFoundError:  # pragma: no cover - allow lightweight installs
    MOTReIDPatchDataset = PatchRecord = ReIDPairDataset = None

__all__ = [
    "MotAnnotation",
    "parse_mot_annotations",
    "annotations_by_frame",
    "decode_rle_mask",
    "mot_annotations_to_dataframe",
    "render_annotations_overlay",
    "build_dataset_index",
    "find_gt_files",
    "generate_sequence_overlays",
    "generate_dataset_overlays",
]

if PatchRecord is not None:
    __all__.extend(["PatchRecord", "MOTReIDPatchDataset", "ReIDPairDataset"])
