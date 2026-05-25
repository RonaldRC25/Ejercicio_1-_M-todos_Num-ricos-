"""
===========================================================
loo_validation.py
===========================================================

Validación Leave-One-Out (LOO)
para interpolación polinómica.

Author
------
Biomedical Numerical System Project
===========================================================
"""

from __future__ import annotations

import numpy as np

from core.validation.metrics import (
    absolute_error,
    relative_error,
    mae,
    rmse
)

from utils.random_local_window import (
    random_local_window
)


class LOOValidator:
    """
    Validador Leave-One-Out genérico.
    """

    def __init__(
        self,
        interpolator_class,
        x: np.ndarray,
        y: np.ndarray,
        degree: int = 4,
        window_size: int = 5,
        random_state: int | None = None
    ) -> None:

        self.interpolator_class = (
            interpolator_class
        )

        self.degree = degree
        self.window_size = window_size

        # Selección local aleatoria
        (
            self.x_window,
            self.y_window,
            self.target_x
        ) = random_local_window(
            x=x,
            y=y,
            window_size=window_size,
            random_state=random_state
        )

    def run(self) -> dict:
        """
        Ejecuta validación LOO.
        """

        iterations = []

        abs_errors = []
        rel_errors = []

        n = len(self.x_window)

        for i in range(n):

            # Punto removido
            x_removed = (
                self.x_window[i]
            )

            y_removed = (
                self.y_window[i]
            )

            # Subconjunto restante
            mask = np.ones(
                n,
                dtype=bool
            )

            mask[i] = False

            x_train = (
                self.x_window[mask]
            )

            y_train = (
                self.y_window[mask]
            )

            # Reconstrucción polinómica
            interpolator = (
                self.interpolator_class(
                    x=x_train,
                    y=y_train,
                    degree=len(x_train)-1
                )
            )

            # Predicción del punto removido
            y_pred = interpolator.evaluate(
                x_removed
            )

            # Cálculo de errores
            abs_err = absolute_error(
                y_removed,
                y_pred
            )

            rel_err = relative_error(
                y_removed,
                y_pred
            )

            abs_errors.append(
                abs_err
            )

            rel_errors.append(
                rel_err
            )

            # Malla densa local
            x_dense, y_dense = (
                interpolator.generate_dense_mesh()
            )

            iterations.append({

                "removed_x": x_removed,
                "removed_y": y_removed,

                "predicted_y": y_pred,

                "absolute_error": abs_err,
                "relative_error": rel_err,

                "x_train": x_train,
                "y_train": y_train,

                "x_dense": x_dense,
                "y_dense": y_dense
            })

        abs_errors = np.array(
            abs_errors
        )

        rel_errors = np.array(
            rel_errors
        )

        return {

            "target_x": self.target_x,

            "window_x": self.x_window,
            "window_y": self.y_window,

            "iterations": iterations,

            "mae": mae(abs_errors),

            "rmse": rmse(abs_errors),

            "mean_relative_error": float(
                np.mean(rel_errors)
            ),

            "max_relative_error": float(
                np.max(rel_errors)
            )
        }