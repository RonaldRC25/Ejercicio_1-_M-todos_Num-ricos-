"""
===========================================================
metrics.py
===========================================================

Métricas de error para validación numérica.

Author
------
Biomedical Numerical System Project
===========================================================
"""

from __future__ import annotations

import numpy as np


def absolute_error(
    y_true: float,
    y_pred: float
) -> float:
    """
    Error absoluto.
    """

    return float(
        np.abs(y_true - y_pred)
    )


def relative_error(
    y_true: float,
    y_pred: float
) -> float:
    """
    Error relativo porcentual.
    """

    if np.isclose(y_true, 0.0):
        raise ValueError(
            "No se puede calcular "
            "error relativo con y_true=0."
        )

    return float(
        np.abs(
            (y_true - y_pred) / y_true
        ) * 100.0
    )


def mae(
    errors: np.ndarray
) -> float:
    """
    Mean Absolute Error.
    """

    return float(
        np.mean(
            np.abs(errors)
        )
    )


def rmse(
    errors: np.ndarray
) -> float:
    """
    Root Mean Square Error.
    """

    return float(
        np.sqrt(
            np.mean(errors ** 2)
        )
    )