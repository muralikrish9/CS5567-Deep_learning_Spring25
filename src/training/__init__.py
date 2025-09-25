"""Training utilities for the MOT project."""

from .reid_trainer import ReIDTrainingConfig, run_training  # noqa: F401

__all__ = ["ReIDTrainingConfig", "run_training"]
