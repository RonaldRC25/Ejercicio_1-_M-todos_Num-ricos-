"""
===========================================================
sensitivity_analysis.py
===========================================================

Análisis de sensibilidad de raíces.

Características
----------------
- Derivación numérica
- Diferencias centrales O(h²)
- Sensibilidad:
      df/d|Z|
- Estabilidad numérica

===========================================================
"""

from __future__ import annotations

import numpy as np

from core.interpolation.cubic_spline import (
    CubicSplineInterpolator
)

from utils.data_loader import (
    load_impedance_data
)


class SensitivityAnalysis:
    """
    Análisis de sensibilidad usando
    derivadas numéricas.

    La derivada se aproxima mediante:

        d|Z|/df ≈
        [S(f+h)-S(f-h)]/(2h)

    usando diferencias centrales
    de orden:

        O(h²)
    """

    def __init__(
        self,
        spline: CubicSplineInterpolator,
        root_frequency: float,
        h: float = 1e-3
    ) -> None:

        self.spline = spline

        self.root_frequency = float(
            root_frequency
        )

        self.h = float(h)

    def numerical_derivative(
        self
    ) -> float:
        """
        Calcula:

            d|Z|/df

        mediante diferencias centrales.
        """

        f = self.root_frequency

        h = self.h

        # =================================================
        # Diferencias centrales:
        #
        # d|Z|/df ≈
        # [S(f+h)-S(f-h)]/(2h)
        #
        # Error de truncamiento:
        #       O(h²)
        # =================================================

        derivative = (
            self.spline.evaluate(
                f + h
            )
            -
            self.spline.evaluate(
                f - h
            )
        ) / (2.0 * h)

        return derivative

    def inverse_sensitivity(
        self
    ) -> float:
        """
        Calcula sensibilidad inversa:

            df/d|Z|

        usando:

            df/d|Z| = 1 / (d|Z|/df)
        """

        dZ_df = (
            self.numerical_derivative()
        )

        if np.isclose(
            dZ_df,
            0.0
        ):

            raise ZeroDivisionError(
                "Derivada cercana a cero."
            )

        return 1.0 / dZ_df

    def summary(
        self
    ) -> dict:
        """
        Resumen del análisis.
        """

        dZ_df = (
            self.numerical_derivative()
        )

        df_dZ = (
            self.inverse_sensitivity()
        )

        return {
            "root_frequency":
            self.root_frequency,

            "h":
            self.h,

            "dZ_df":
            dZ_df,

            "df_dZ":
            df_dZ
        }


if __name__ == "__main__":

    # =====================================================
    # Carga experimental
    # =====================================================

    Z, f = load_impedance_data()

    # =====================================================
    # Construcción spline
    # =====================================================

    spline = (
        CubicSplineInterpolator(
            x=f,
            y=Z
        )
    )

    # =====================================================
    # Frecuencia cercana a raíz ~2000 Hz
    # =====================================================

    root_frequency = 2216.741272972865

    analysis = (
        SensitivityAnalysis(
            spline=spline,
            root_frequency=root_frequency,
            h=1e-3
        )
    )

    print(
        analysis.summary()
    )