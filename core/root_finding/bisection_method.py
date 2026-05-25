"""
===========================================================
bisection_method.py
===========================================================

Método de bisección para búsqueda de raíces.

Características
----------------
- Intervalos configurables
- Error relativo aproximado
- Exportación CSV
- Registro iterativo
- Robustez numérica

===========================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pathlib import Path
from typing import Callable


class BisectionMethod:
    """
    Método de bisección clásico.

    Resuelve:
        f(x) = 0

    usando reducción iterativa de intervalos.
    """

    def __init__(
        self,
        function: Callable,
        f_i: float,
        f_s: float,
        tolerance: float = 1e-6,
        max_iter: int = 100,
        export_csv: bool = False,
        export_path: str | None = None
    ) -> None:

        self.function = function

        self.f_i = float(f_i)

        self.f_s = float(f_s)

        self.tolerance = tolerance

        self.max_iter = max_iter

        self.export_csv = export_csv

        self.export_path = export_path

        self.history = []

        # =================================================
        # Validación de cambio de signo
        # =================================================

        if (
            self.function(self.f_i) *
            self.function(self.f_s)
        ) > 0:

            raise ValueError(
                "El intervalo no contiene raíz."
            )

    def solve(
        self
    ) -> dict:
        """
        Ejecuta el método de bisección.
        """

        f_i = self.f_i

        f_s = self.f_s

        f_r_old = None

        for iteration in range(
            1,
            self.max_iter + 1
        ):

            # =============================================
            # Punto medio del intervalo
            # =============================================

            f_r = (
                f_i + f_s
            ) / 2.0

            g_i = self.function(f_i)

            g_r = self.function(f_r)

            # =============================================
            # Error relativo aproximado
            # =============================================

            if f_r_old is None:

                error = np.nan

            else:

                error = abs(
                    (
                        f_r - f_r_old
                    ) / f_r
                ) * 100

            # =============================================
            # Registro iterativo
            # =============================================

            self.history.append({
                "iter":
                iteration,

                "f_i":
                f_i,

                "f_r":
                f_r,

                "f_s":
                f_s,

                "error":
                error
            })

            # =============================================
            # Criterio de convergencia
            # =============================================

            if (
                error is not np.nan and
                error < self.tolerance
            ):
                break

            # =============================================
            # Reducción del intervalo
            # =================================================
            # Si existe cambio de signo entre:
            #     [f_i,f_r]
            # la raíz permanece en dicho intervalo.
            # =================================================

            if g_i * g_r < 0:

                f_s = f_r

            else:

                f_i = f_r

            f_r_old = f_r

        if self.export_csv:

            self.export_iterations()

        return {
            "root":
            f_r,

            "iterations":
            iteration,

            "error":
            error
        }

    def export_iterations(
        self
    ) -> None:
        """
        Exporta iteraciones a CSV.
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