"""Tracking utilities combining detector outputs with Re-ID embeddings."""

from .association import associate_detections
from .inference import TrackerConfig, run_tracker

__all__ = [
    "associate_detections",
    "TrackerConfig",
    "run_tracker",
]
