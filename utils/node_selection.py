from __future__ import annotations

import numpy as np

from typing import Optional


def select_interpolation_nodes(
    x: np.ndarray,
    y: np.ndarray,
    n_nodes: int,
    strategy: str = "sequential",
    target_x: Optional[float] = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select interpolation nodes using different
    node selection strategies.

    Supported strategies
    --------------------
    sequential:
        Progressive node growth from the
        beginning of the dataset.

    distributed:
        Uniform node distribution over the
        complete experimental domain.

    nearest:
        Local node selection around target_x.
    """

    # Validate node count
    if n_nodes > len(x):
        raise ValueError(
            "n_nodes exceeds dataset size."
        )

    # Progressive node growth
    if strategy == "sequential":

        indices = np.arange(
            n_nodes
        )

    # Uniform domain distribution
    elif strategy == "distributed":

        indices = np.linspace(
            0,
            len(x) - 1,
            n_nodes,
            dtype=int
        )

    # Local interpolation around target_x
    elif strategy == "nearest":

        if target_x is None:
            raise ValueError(
                "target_x is required for "
                "'nearest' strategy."
            )

        # Distance to target point
        distances = np.abs(
            x - target_x
        )

        # Closest node indices
        indices = np.argsort(
            distances
        )[:n_nodes]

        # Preserve increasing order
        indices = np.sort(
            indices
        )

    else:
        raise ValueError(
            f"Unknown strategy: {strategy}"
        )

    return (
        x[indices],
        y[indices]
    )