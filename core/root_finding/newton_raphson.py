"""
===========================================================
newton_raphson.py
===========================================================

Método de Newton-Raphson.

Características
----------------
- Convergencia rápida
- Uso de derivadas spline
- Error relativo aproximado
- Exportación CSV

===========================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pathlib import Path
from typing import Callable


class NewtonRaphsonMethod:
    """
    Método de Newton-Raphson.

    Resuelve:
        f(x)=0

    usando:

        x_(n+1)=x_n-f(x_n)/f'(x_n)
    """

    def __init__(
        self,
        function: Callable,
        derivative: Callable,
        x0: float,
        tolerance: float = 1e-6,
        max_iter: int = 100,
        derivative_tol: float = 1e-12,
        export_csv: bool = False,
        export_path: str | None = None
    ) -> None:

        self.function = function

        self.derivative = derivative

        self.x0 = float(x0)

        self.tolerance = tolerance

        self.max_iter = max_iter

        self.derivative_tol = derivative_tol

        self.export_csv = export_csv

        self.export_path = export_path

        self.history = []

    def solve(
        self
    ) -> dict:
        """
        Ejecuta Newton-Raphson.
        """

        x_n = self.x0

        for iteration in range(
            1,
            self.max_iter + 1
        ):

            f_x = self.function(x_n)

            df_x = self.derivative(x_n)

            # =============================================
            # Evitar división por valores cercanos a cero
            # =============================================

            if abs(df_x) < self.derivative_tol:

                raise ZeroDivisionError(
                    "Derivada cercana a cero."
                )

            # =============================================
            # Actualización Newton-Raphson
            # =============================================

            x_next = (
                x_n -
                (
                    f_x / df_x
                )
            )

            # =============================================
            # Error relativo aproximado
            # =============================================

            error = abs(
                (
                    x_next - x_n
                ) / x_next
            ) * 100

            # =============================================
            # Registro iterativo
            # =============================================

            self.history.append({
                "iter":
                iteration,

                "f_n+1":
                x_next,

                "f_n":
                x_n,

                "Z(x_n)":
                f_x,

                "Z'(x_n)":
                df_x,

                "error":
                error
            })

            # =============================================
            # Convergencia
            # =============================================

            if error < self.tolerance:

                x_n = x_next

                break

            x_n = x_next

        if self.export_csv:

            self.export_iterations()

        return {
            "root":
            x_n,

            "iterations":
            iteration,

            "error":
            error
        }

    def export_iterations(
        self
    ) -> None:
        """
        Exporta iteraciones.
        """

        if self.export_path is None:

            raise ValueError(
                "export_path no definido."
            )

        df = pd.DataFrame(
            self.history
        )

        Path(
            self.export_path
        ).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        df.to_csv(
            self.export_path,
            index=False
        )

        print(
            df.to_string(
                index=False
            )
        )