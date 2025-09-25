"""Association helpers for matching detections to existing tracks."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
try:  # SciPy provides an efficient Hungarian implementation
    from scipy.optimize import linear_sum_assignment  # type: ignore
except ImportError:  # pragma: no cover - fallback to greedy matching
    linear_sum_assignment = None


def cosine_distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return cosine distance matrix between two sets of embeddings."""

    if a.size == 0 or b.size == 0:
        return np.empty((len(a), len(b)))
    a_norm = np.linalg.norm(a, axis=1, keepdims=True) + 1e-8
    b_norm = np.linalg.norm(b, axis=1, keepdims=True) + 1e-8
    similarity = (a @ b.T) / (a_norm * b_norm.T)
    # Cosine distance = 1 - cosine similarity
    return 1.0 - similarity


def gate_cost_matrix(
    cost_matrix: np.ndarray,
    *,
    max_cost: float,
) -> np.ndarray:
    """Clamp costs above threshold to a high sentinel so they won't be chosen."""

    gated = cost_matrix.copy()
    gated[gated > max_cost] = max_cost + 1.0
    return gated


def linear_assignment(cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if cost_matrix.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    if linear_sum_assignment is not None:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return row_ind, col_ind

    # Greedy fallback (may produce sub-optimal matches, but avoids dependency)
    cost = cost_matrix.copy()
    row_ind: List[int] = []
    col_ind: List[int] = []
    while cost.size and np.isfinite(cost).any():
        idx = np.unravel_index(np.argmin(cost, axis=None), cost.shape)
        r, c = idx
        if not np.isfinite(cost[r, c]):
            break
        row_ind.append(int(r))
        col_ind.append(int(c))
        cost[r, :] = np.inf
        cost[:, c] = np.inf
    return np.array(row_ind, dtype=int), np.array(col_ind, dtype=int)


def associate_detections(
    track_embeddings: np.ndarray,
    detection_embeddings: np.ndarray,
    *,
    max_distance: float,
    iou_matrix: Optional[np.ndarray] = None,
    iou_weight: float = 0.5,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Match tracks to detections using embeddings (and optional IOU)."""

    if track_embeddings.size == 0 or detection_embeddings.size == 0:
        return [], list(range(len(track_embeddings))), list(range(len(detection_embeddings)))

    distance_matrix = cosine_distance_matrix(track_embeddings, detection_embeddings)
    cost_matrix = distance_matrix

    if iou_matrix is not None and iou_matrix.size:
        # IOU matrix is in [0, 1]; convert to a distance penalty
        iou_distance = 1.0 - np.clip(iou_matrix, 0.0, 1.0)
        cost_matrix = (1.0 - iou_weight) * distance_matrix + iou_weight * iou_distance

    gated = gate_cost_matrix(cost_matrix, max_cost=max_distance)
    row_ind, col_ind = linear_assignment(gated)

    matches: List[Tuple[int, int]] = []
    unmatched_tracks = set(range(len(track_embeddings)))
    unmatched_detections = set(range(len(detection_embeddings)))

    for track_idx, det_idx in zip(row_ind, col_ind):
        if gated[track_idx, det_idx] > max_distance:
            continue
        matches.append((track_idx, det_idx))
        unmatched_tracks.discard(track_idx)
        unmatched_detections.discard(det_idx)

    return matches, sorted(unmatched_tracks), sorted(unmatched_detections)


__all__ = ["associate_detections", "cosine_distance_matrix"]
