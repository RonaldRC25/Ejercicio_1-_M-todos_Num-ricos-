"""
===========================================================
random_local_window.py
===========================================================

Selección aleatoria de ventanas locales
coherentes para interpolación y validación.

Author
------
Biomedical Numerical System Project
===========================================================
"""

from __future__ import annotations

import numpy as np

from utils.node_selection import (
    select_interpolation_nodes
)


def random_local_window(
    x: np.ndarray,
    y: np.ndarray,
    window_size: int = 5,
    random_state: int | None = None
) -> tuple[
    np.ndarray,
    np.ndarray,
    float
]:
    """
    Selecciona una ventana local aleatoria
    usando estrategia nearest.
    """

    if window_size >= len(x):
        raise ValueError(
            "window_size excede "
            "el tamaño del dataset."
        )

    rng = np.random.default_rng(
        random_state
    )

    # Punto de referencia aleatorio
    target_x = float(
        rng.choice(x)
    )

    # Ventana local coherente
    x_window, y_window = (
        select_interpolation_nodes(
            x=x,
            y=y,
            n_nodes=window_size,
            strategy="nearest",
            target_x=target_x
        )
    )

    return (
        x_window,
        y_window,
        target_x
    )