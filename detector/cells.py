"""Grid-cell discretization for embedding-space coverage tracking.

P1 owns this file. P3 changed two defaults on Day 2 -- see the note below.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

# Measured on 5000 real CIFAR-10 embeddings (python -m victim.embed diagnose).
#
# The old defaults -- all 32 dimensions at cell_size 0.2 -- put every single
# query in its own cell: 100% unique, so coverage never saturates and every
# client looks identical. The number of cells multiplies with each dimension,
# and by 32 there is far more room than there are queries to fill it.
#
# 6 dimensions at cell_size 1.0 gives ~16% unique on real embeddings: clients
# revisit ground, coverage saturates, and a client sweeping the whole space
# stands out. Re-measure against real traffic with:
#     python -m victim.embed diagnose --db data/ledger.db --run-id <run>
DEFAULT_DIMS = 6
DEFAULT_CELL_SIZE = 1.0


def assign_cell(embedding: Iterable[float],
                cell_size: float = DEFAULT_CELL_SIZE,
                dims: int | None = DEFAULT_DIMS) -> tuple[int, ...]:
    """
    Bucket a query embedding into a discrete grid cell.

    Each dimension is divided into steps of ``cell_size`` and rounded to the
    nearest step index, producing a hashable cell tuple.

    ``dims`` limits how many leading dimensions take part. Passing None uses
    all of them, which is the old behaviour and produces no usable signal on
    32-dimensional embeddings.
    """
    if cell_size <= 0:
        raise ValueError("cell_size must be positive")

    arr = np.asarray(list(embedding), dtype=float)
    if dims is not None:
        arr = arr[:dims]
    indices = np.round(arr / cell_size).astype(int)
    return tuple(indices.tolist())


def coverage_ratio(covered_cells: set, total_cells_estimate: int) -> float:
    """Return the fraction of the estimated total cell space that is covered."""
    if total_cells_estimate <= 0:
        raise ValueError("total_cells_estimate must be positive")
    return len(covered_cells) / total_cells_estimate