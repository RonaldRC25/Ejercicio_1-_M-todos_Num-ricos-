"""
===========================================================
runge_analysis.py
===========================================================

Análisis comparativo del fenómeno de Runge
para interpolación polinómica.

Author
------
Biomedical Numerical System Project
===========================================================
"""

from __future__ import annotations

import numpy as np

from typing import Type


class RungeAnalyzer:
    """
    Analizador de estabilidad polinómica.
    """

    def __init__(
        self,
        interpolator_class: Type,
        x: np.ndarray,
        y: np.ndarray,
        degrees: list[int],
        dense_points: int = 5000,
        node_strategy: str = "sequential"
    ) -> None:

        self.interpolator_class = (
            interpolator_class
        )

        self.x = np.asarray(
            x,
            dtype=np.float64
        )

        self.y = np.asarray(
            y,
            dtype=np.float64
        )

        self.degrees = degrees

        self.dense_points = dense_points

        self.node_strategy = (
            node_strategy
        )

    def run(self) -> dict:
        """
        Ejecuta análisis Runge.
        """

        results = {}

        for degree in self.degrees:

            # Construcción del interpolador
            interpolator = (
                self.interpolator_class(
                    x=self.x,
                    y=self.y,
                    degree=degree,
                    dense_points=self.dense_points,
                    node_strategy=self.node_strategy
                )
            )

            # Malla densa
            x_dense, y_dense = (
                interpolator.generate_dense_mesh()
            )

            # Amplitud oscilatoria
            oscillation = float(
                np.max(y_dense) -
                np.min(y_dense)
            )

            # Metadatos generales
            summary = (
                interpolator.summary()
            )

            results[degree] = {

                "x_dense": x_dense,
                "y_dense": y_dense,

                "x_nodes": (
                    interpolator.x_nodes
                ),

                "y_nodes": (
                    interpolator.y_nodes
                ),

                "summary": summary,

                "oscillation": oscillation
            }

        return results