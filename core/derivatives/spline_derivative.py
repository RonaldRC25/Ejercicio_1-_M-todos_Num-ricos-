"""
===========================================================
spline_derivative.py
===========================================================

Spline-based numerical differentiation.

Features
--------
- First derivative evaluation
- Second derivative evaluation
- Minimum detection
- Stability analysis
- Dense derivative mesh generation

Author
------
Biomedical Numerical System Project
===========================================================
"""

from __future__ import annotations

import numpy as np

from scipy.optimize import (
    brentq
)

from core.interpolation.cubic_spline import (
    CubicSplineInterpolator
)

from utils.data_loader import (
    load_impedance_data
)


class SplineDerivativeAnalyzer:
    """
    Analyze spline derivatives.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        dense_points: int = 5000
    ) -> None:

        # =================================================
        # Store experimental data
        # =================================================

        self.x = np.asarray(
            x,
            dtype=np.float64
        )

        self.y = np.asarray(
            y,
            dtype=np.float64
        )

        self.dense_points = (
            dense_points
        )

        # =================================================
        # Construct spline interpolator
        # =================================================

        self.spline = (
            CubicSplineInterpolator(
                x=self.x,
                y=self.y,
                dense_points=dense_points
            )
        )

        # =================================================
        # Derivative spline functions
        # =================================================

        self.first_derivative = (
            self.spline.spline.derivative(
                1
            )
        )

        self.second_derivative = (
            self.spline.spline.derivative(
                2
            )
        )

    def generate_dense_derivative_mesh(
        self
    ) -> tuple[
        np.ndarray,
        np.ndarray
    ]:
        """
        Generate dense first derivative mesh.
        """

        x_dense = np.linspace(
            self.x.min(),
            self.x.max(),
            self.dense_points
        )

        dy_dense = (
            self.first_derivative(
                x_dense
            )
        )

        return (
            x_dense,
            dy_dense
        )

    def find_minimum_frequency(
        self
    ) -> tuple[float, float]:
        """
        Detect minimum location using
        derivative zero-crossing.
        """

        x_dense, dy_dense = (
            self.generate_dense_derivative_mesh()
        )

        # =================================================
        # Detect sign change
        # =================================================

        for i in range(
            len(dy_dense) - 1
        ):

            if (
                dy_dense[i] < 0 and
                dy_dense[i + 1] > 0
            ):

                # =========================================
                # Root refinement
                # =========================================

                f_min = brentq(
                    self.first_derivative,
                    x_dense[i],
                    x_dense[i + 1]
                )

                z_min = (
                    self.spline.evaluate(
                        f_min
                    )
                )

                return (
                    float(f_min),
                    float(z_min)
                )

        raise RuntimeError(
            "Minimum not detected."
        )

    def evaluate_second_derivative(
        self,
        x_eval: float
    ) -> float:
        """
        Evaluate second derivative.
        """

        return float(
            self.second_derivative(
                x_eval
            )
        )

    def stability_analysis(
        self
    ) -> dict:
        """
        Analyze minimum stability.
        """

        f_min, z_min = (
            self.find_minimum_frequency()
        )

        d2z = (
            self.evaluate_second_derivative(
                f_min
            )
        )

        if d2z > 0:

            stability = (
                "Stable minimum"
            )

        elif d2z < 0:

            stability = (
                "Maximum detected"
            )

        else:

            stability = (
                "Degenerate critical point"
            )

        return {
            "f_min":
            f_min,

            "z_min":
            z_min,

            "second_derivative":
            d2z,

            "stability":
            stability
        }


if __name__ == "__main__":

    Z, f = load_impedance_data()

    analyzer = (
        SplineDerivativeAnalyzer(
            x=f,
            y=Z
        )
    )

    f_min, z_min = (
        analyzer.find_minimum_frequency()
    )

    print(
        f"Minimum frequency: "
        f"{f_min:.6f} Hz"
    )

    print(
        f"|Z|min = "
        f"{z_min:.6f} Ohm"
    )

    result = (
        analyzer.stability_analysis()
    )

    print(result)