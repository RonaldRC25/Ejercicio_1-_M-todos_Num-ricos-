"""
===========================================================
inter_matrix_method.py
===========================================================

Global polynomial interpolation using the
matrix-based Vandermonde formulation.

Features
--------
- Global polynomial interpolation
- Domain scaling for stability
- Centralized node selection
- Vandermonde matrix formulation
- Dense mesh generation
- Runge phenomenon analysis
- API consistency with Lagrange.py

Author
------
Biomedical Numerical System Project
===========================================================
"""

from __future__ import annotations

import numpy as np

from typing import Optional

from utils.node_selection import (
    select_interpolation_nodes
)


class MatrixPolynomialInterpolator:
    """
    Global polynomial interpolator using
    Vandermonde matrix formulation.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        degree: Optional[int] = None,
        dense_points: int = 5000,
        node_strategy: str = "sequential",
        target_x: Optional[float] = None
    ) -> None:

        # Convert inputs to numerical arrays
        self.x = np.asarray(
            x,
            dtype=np.float64
        )

        self.y = np.asarray(
            y,
            dtype=np.float64
        )

        # Validate input dimensions
        if self.x.ndim != 1 or self.y.ndim != 1:
            raise ValueError(
                "x and y must be one-dimensional arrays."
            )

        # Validate matching sizes
        if len(self.x) != len(self.y):
            raise ValueError(
                "x and y must have the same length."
            )

        # Minimum interpolation requirement
        if len(self.x) < 2:
            raise ValueError(
                "At least two points are required."
            )

        # Ensure strictly increasing nodes
        if np.any(np.diff(self.x) <= 0):
            raise ValueError(
                "x values must be strictly increasing."
            )

        self.n_total = len(self.x)

        # Use maximum possible degree by default
        self.degree = (
            self.n_total - 1
            if degree is None
            else int(degree)
        )

        # Validate polynomial degree
        if self.degree < 1:
            raise ValueError(
                "degree must be >= 1."
            )

        if self.degree >= self.n_total:
            raise ValueError(
                "degree must satisfy:"
                " degree < number of samples."
            )

        # Number of interpolation nodes
        self.n_nodes = self.degree + 1
        # Node selection configuration
        self.node_strategy = node_strategy
        self.target_x = target_x

        # Centralized node selection
        self.x_nodes, self.y_nodes = (
            select_interpolation_nodes(
                x=self.x,
                y=self.y,
                n_nodes=self.n_nodes,
                strategy=self.node_strategy,
                target_x=self.target_x
            )
        )

        # Domain scaling for numerical stability
        self.x_min = self.x_nodes.min()
        self.x_max = self.x_nodes.max()

        self.x_scaled = (
            self.x_nodes - self.x_min
        ) / (
            self.x_max - self.x_min
        )

        # Dense mesh storage
        self.dense_points = dense_points

        # Vandermonde matrix construction
        self.V = np.vander(
            self.x_scaled,
            N=self.n_nodes,
            increasing=True
        )

        # Solve interpolation system
        self.coefficients = np.linalg.solve(
            self.V,
            self.y_nodes
        )

        # Matrix conditioning metric
        self.condition_number = (
            np.linalg.cond(self.V)
        )

    def evaluate(
        self,
        x_eval: np.ndarray | float
    ) -> np.ndarray:
        """
        Evaluate interpolation polynomial.
        """

        scalar_input = np.isscalar(
            x_eval
        )

        # Support scalar/vector evaluation
        x_eval = np.atleast_1d(
            x_eval
        ).astype(np.float64)

        # Scale evaluation points
        x_scaled = (
            x_eval - self.x_min
        ) / (
            self.x_max - self.x_min
        )

        # Polynomial evaluation
        y_eval = (
            np.polynomial.polynomial.polyval(
                x_scaled,
                self.coefficients
            )
        )

        return (
            y_eval[0]
            if scalar_input
            else y_eval
        )

    def generate_dense_mesh(
        self,
        x_min: Optional[float] = None,
        x_max: Optional[float] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate dense interpolation mesh.
        """

        # Use active interpolation domain
        if x_min is None:
            x_min = self.x_nodes.min()

        if x_max is None:
            x_max = self.x_nodes.max()

        # Dense evaluation domain
        x_dense = np.linspace(
            x_min,
            x_max,
            self.dense_points
        )

        # Evaluate interpolation
        y_dense = self.evaluate(
            x_dense
        )

        return x_dense, y_dense

    def interpolation_error(
        self
    ) -> np.ndarray:
        """
        Compute interpolation residuals.
        """

        y_hat = self.evaluate(
            self.x_nodes
        )

        return (
            self.y_nodes - y_hat
        )

    def summary(
        self
    ) -> dict:
        """
        Return interpolation metadata.
        """

        return {
            "method": "Matrix Vandermonde",
            "degree": self.degree,
            "n_nodes": self.n_nodes,
            "target_x": self.target_x,
            "node_strategy": self.node_strategy,
            "dense_points": (
                self.dense_points
            ),
            "condition_number": (
                self.condition_number
            ),
            "x_min": float(
                self.x_nodes.min()
            ),
            "x_max": float(
                self.x_nodes.max()
            )
        }


def compute_matrix_interpolation(
    frequency: np.ndarray,
    impedance: np.ndarray,
    degree: Optional[int] = None,
    dense_points: int = 5000,
    node_strategy: str = "sequential",
    target_x: Optional[float] = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    High-level interpolation interface for matrix-based polynomial.
    """

    interpolator = MatrixPolynomialInterpolator(
        x=frequency,
        y=impedance,
        degree=degree,
        dense_points=dense_points,
        node_strategy=node_strategy,
        target_x=target_x   # <<--- importante incluir
    )

    return interpolator.generate_dense_mesh()

if __name__ == "__main__":

    from utils.data_loader import (
        load_impedance_data
    )

    # Load experimental dataset
    Z, f = load_impedance_data()

    # Create interpolator
    interpolator = (
        MatrixPolynomialInterpolator(
            x=f,
            y=Z,
            degree=4,
            node_strategy="nearest",
            target_x=1000.0
        )
    )

    # Evaluate impedance at 1000 Hz
    z_1000 = interpolator.evaluate(
        1000.0
    )

    print(
        f"|Z|(1000 Hz) = "
        f"{z_1000:.6f} Ohm"
    )

    # Generate dense mesh
    f_dense, Z_dense = (
        interpolator.generate_dense_mesh()
    )

    print(
        f"Generated mesh:"
        f" {len(f_dense)} points"
    )

    # Print interpolation metadata
    print(
        interpolator.summary()
    )