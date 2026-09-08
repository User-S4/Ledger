"""Grid-cell discretization for embedding-space coverage tracking."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def assign_cell(embedding: Iterable[float], cell_size: float = 0.2) -> tuple[int, ...]:
    """
    Bucket a query embedding into a discrete grid cell.

    Each dimension is divided into steps of ``cell_size`` and rounded to the
    nearest step index, producing a hashable cell tuple.
    """
    if cell_size <= 0:
        raise ValueError("cell_size must be positive")

    arr = np.asarray(list(embedding), dtype=float)
    indices = np.round(arr / cell_size).astype(int)
    return tuple(indices.tolist())


def coverage_ratio(covered_cells: set, total_cells_estimate: int) -> float:
    """Return the fraction of the estimated total cell space that is covered."""
    if total_cells_estimate <= 0:
        raise ValueError("total_cells_estimate must be positive")
    return len(covered_cells) / total_cells_estimate
