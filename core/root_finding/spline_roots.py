"""
===========================================================
spline_roots.py
===========================================================

Búsqueda de raíces usando spline cúbico.

Características
----------------
- Construcción de:
      g(f)=|Z|(f)-Z_th
- Detección automática de cambios de signo
- Integración con:
      Bisección
      Newton-Raphson
- Consolidación de raíces

===========================================================
"""

from __future__ import annotations

import numpy as np

from core.interpolation.cubic_spline import (
    CubicSplineInterpolator
)

from core.root_finding.bisection_method import (
    BisectionMethod
)

from core.root_finding.newton_raphson import (
    NewtonRaphsonMethod
)

from utils.data_loader import (
    load_impedance_data
)


class SplineRootFinder:
    """
    Orquestador de búsqueda de raíces.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z_threshold: float = 150.0,
        dense_points: int = 5000
    ) -> None:

        self.x = np.asarray(
            x,
            dtype=np.float64
        )

        self.y = np.asarray(
            y,
            dtype=np.float64
        )

        self.z_threshold = (
            z_threshold
        )

        self.dense_points = (
            dense_points
        )

        # =================================================
        # Construcción spline cúbico
        # =================================================

        self.spline = (
            CubicSplineInterpolator(
                x=self.x,
                y=self.y,
                dense_points=dense_points
            )
        )

        # =================================================
        # Derivada analítica spline
        # =================================================

        self.spline_derivative = (
            self.spline.spline.derivative(
                1
            )
        )

    def g(
        self,
        frequency: float
    ) -> float:
        """
        Función objetivo:

            g(f)=|Z|(f)-Z_th
        """

        return (
            self.spline.evaluate(
                frequency
            ) -
            self.z_threshold
        )

    def find_sign_changes(
        self
    ) -> list[tuple[float, float]]:
        """
        Detecta intervalos con cambio de signo.
        """

        f_dense = np.linspace(
            self.x.min(),
            self.x.max(),
            self.dense_points
        )

        g_dense = self.g(
            f_dense
        )

        intervals = []

        for i in range(
            len(g_dense) - 1
        ):

            if (
                g_dense[i] *
                g_dense[i + 1]
            ) < 0:

                intervals.append(
                    (
                        f_dense[i],
                        f_dense[i + 1]
                    )
                )

        return intervals

    def solve_bisection(
        self,
        tolerance: float = 1e-6
    ) -> list[dict]:
        """
        Resuelve raíces mediante bisección.
        """

        roots = []

        intervals = (
            self.find_sign_changes()
        )

        for (
            f_i,
            f_s
        ) in intervals:

            solver = (
                BisectionMethod(
                    function=self.g,
                    f_i=f_i,
                    f_s=f_s,
                    tolerance=tolerance
                )
            )

            roots.append(
                solver.solve()
            )

        return roots

    def solve_newton(
        self,
        tolerance: float = 1e-6
    ) -> list[dict]:
        """
        Resuelve raíces mediante Newton.
        """

        roots = []

        intervals = (
            self.find_sign_changes()
        )

        for (
            f_i,
            f_s
        ) in intervals:

            # =============================================
            # Semilla inicial:
            # punto medio del intervalo.
            # =============================================

            x0 = (
                f_i + f_s
            ) / 2.0

            solver = (
                NewtonRaphsonMethod(
                    function=self.g,
                    derivative=
                    self.spline_derivative,
                    x0=x0,
                    tolerance=tolerance
                )
            )

            roots.append(
                solver.solve()
            )

        return roots


if __name__ == "__main__":

    Z, f = load_impedance_data()

    finder = (
        SplineRootFinder(
            x=f,
            y=Z,
            z_threshold=150.0
        )
    )

    print("\nBisección:\n")

    print(
        finder.solve_bisection()
    )

    print("\nNewton-Raphson:\n")

    print(
        finder.solve_newton()
    )