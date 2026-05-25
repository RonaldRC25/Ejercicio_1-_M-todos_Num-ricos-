"""
===========================================================
convergence_metrics.py
===========================================================

Métricas de convergencia para métodos iterativos.

Características
----------------
- Error relativo aproximado
- Error absoluto
- Criterios de parada
- Utilidades reutilizables

===========================================================
"""

from __future__ import annotations

import numpy as np


def relative_approximate_error(
    current_value: float,
    previous_value: float
) -> float:
    """
    Calcula el error relativo aproximado:

        Ea = |(x_i - x_(i-1))/x_i| * 100

    Parámetros
    ----------
    current_value : float
        Valor actual de la iteración.

    previous_value : float
        Valor previo de la iteración.

    Retorna
    -------
    float
        Error relativo aproximado [%].
    """

    current_value = float(
        current_value
    )

    previous_value = float(
        previous_value
    )

    if np.isclose(
        current_value,
        0.0
    ):

        return np.inf

    return abs(
        (
            current_value -
            previous_value
        ) / current_value
    ) * 100.0


def absolute_error(
    current_value: float,
    previous_value: float
) -> float:
    """
    Calcula error absoluto:

        |x_i - x_(i-1)|

    Parámetros
    ----------
    current_value : float
        Valor actual.

    previous_value : float
        Valor previo.

    Retorna
    -------
    float
        Error absoluto.
    """

    return abs(
        current_value -
        previous_value
    )


def has_converged(
    error: float,
    tolerance: float
) -> bool:
    """
    Verifica convergencia numérica.

    Parámetros
    ----------
    error : float
        Error actual.

    tolerance : float
        Tolerancia máxima permitida.

    Retorna
    -------
    bool
        True si converge.
    """

    return error < tolerance