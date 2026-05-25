"""
===========================================================
cubic_spline.py
===========================================================

Natural cubic spline interpolation module.

Features
--------
- Natural cubic spline interpolation
- Dense mesh generation
- Scalar/vector evaluation
- Consistent API with interpolation system
- Numerical stability

Author
------
Biomedical Numerical System Project
===========================================================
"""

from __future__ import annotations

import numpy as np

from scipy.interpolate import (
    CubicSpline
)

from typing import Optional


class CubicSplineInterpolator:
    """
    Natural cubic spline interpolator.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        bc_type: str = "natural",
        dense_points: int = 5000
    ) -> None:

        # =================================================
        # Convert input arrays
        # =================================================

        self.x = np.asarray(
            x,
            dtype=np.float64
        )

        self.y = np.asarray(
            y,
            dtype=np.float64
        )

        # =================================================
        # Validate dimensions
        # =================================================

        if (
            self.x.ndim != 1 or
            self.y.ndim != 1
        ):
            raise ValueError(
                "x and y must be 1-D arrays."
            )

        # =================================================
        # Validate sizes
        # =================================================

        if len(self.x) != len(self.y):
            raise ValueError(
                "x and y must have same length."
            )

        # =================================================
        # Minimum points
        # =================================================

        if len(self.x) < 3:
            raise ValueError(
                "At least 3 points required."
            )

        # =================================================
        # Strictly increasing domain
        # =================================================

        if np.any(
            np.diff(self.x) <= 0
        ):
            raise ValueError(
                "x must be strictly increasing."
            )

        self.bc_type = bc_type

        self.dense_points = (
            dense_points
        )

        # =================================================
        # Construct cubic spline
        # =================================================

        self.spline = CubicSpline(
            self.x,
            self.y,
            bc_type=self.bc_type
        )

    def evaluate(
        self,
        x_eval: float | np.ndarray
    ) -> np.ndarray:
        """
        Evaluate spline interpolation.
        """

        x_eval = np.asarray(
            x_eval,
            dtype=np.float64
        )

        return self.spline(
            x_eval
        )

    def generate_dense_mesh(
        self
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate dense interpolation mesh.
        """

        x_dense = np.linspace(
            self.x.min(),
            self.x.max(),
            self.dense_points
        )

        y_dense = self.evaluate(
            x_dense
        )

        return (
            x_dense,
            y_dense
        )

    def summary(
        self
    ) -> dict:
        """
        Return interpolation metadata.
        """

        return {
            "method":
            "Natural Cubic Spline",

            "bc_type":
            self.bc_type,

            "n_points":
            len(self.x),

            "dense_points":
            self.dense_points,

            "x_min":
            float(self.x.min()),

            "x_max":
            float(self.x.max())
        }


if __name__ == "__main__":

    from utils.data_loader import (
        load_impedance_data
    )

    Z, f = load_impedance_data()

    spline = (
        CubicSplineInterpolator(
            x=f,
            y=Z
        )
    )

    z_1000 = spline.evaluate(
        1000.0
    )

    print(
        f"|Z|(1000 Hz) = "
        f"{z_1000:.6f} Ohm"
    )

    f_dense, Z_dense = (
        spline.generate_dense_mesh()
    )

    print(
        f"Generated mesh: "
        f"{len(f_dense)} points"
    )

    print(
        spline.summary()
    )