"""
===========================================================
runge_comparison.py
===========================================================

Comparación visual del fenómeno de Runge
entre interpolación de Lagrange y método
matricial de Vandermonde.

Características:
----------------
- Escalado robusto adaptativo
- Percentiles sobre malla densa
- Padding dinámico
- Comparación multigrado
- Exportación científica

Author
------
Biomedical Numerical System Project
===========================================================
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from core.analysis.runge_analysis import (
    RungeAnalyzer
)

from core.interpolation.Lagrange import (
    LagrangeInterpolator
)

from core.interpolation.inter_matrix_method import (
    MatrixPolynomialInterpolator
)

from utils.data_loader import (
    load_impedance_data
)

from utils.config import (
    EXPORTS_DIR
)

from plots.styles.plot_config import (
    apply_plot_style
)

# =========================================================
# Directorio de exportación
# =========================================================

EXPORT_DIR = Path(
    EXPORTS_DIR / "runge_analysis"
)

EXPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# =========================================================
# Configuración global
# =========================================================

DEGREES = [5, 10, 15, 29]


def _compute_local_ylim(
    y_dense: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    margin_ratio: float = 0.10
) -> tuple[float, float]:
    """
    Calcula límites verticales robustos
    usando percentiles adaptativos sobre
    la malla densa interpolada.

    Parameters
    ----------
    y_dense : np.ndarray
        Malla densa interpolada.

    lower_percentile : float
        Percentil inferior robusto.

    upper_percentile : float
        Percentil superior robusto.

    margin_ratio : float
        Padding dinámico relativo.

    Returns
    -------
    tuple[float, float]
        Límites verticales robustos.
    """

    # Percentiles robustos
    y_lower = float(
        np.percentile(
            y_dense,
            lower_percentile
        )
    )

    y_upper = float(
        np.percentile(
            y_dense,
            upper_percentile
        )
    )

    # Evitar ventana degenerada
    delta = y_upper - y_lower

    if np.isclose(delta, 0.0):
        delta = 1.0

    # Padding dinámico
    padding = (
        margin_ratio * delta
    )

    return (
        y_lower - padding,
        y_upper + padding
    )


def _plot_method_analysis(
    method_name: str,
    results: dict,
    frequency: np.ndarray,
    impedance: np.ndarray,
    save_name: str,
    save: bool = True,
    show: bool = True
) -> None:
    """
    Genera subplots comparativos
    del fenómeno de Runge.
    """

    # =====================================================
    # Figura principal
    # =====================================================

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10),
        sharex=True
    )

    axes = axes.flatten()

    # =====================================================
    # Iteración sobre grados
    # =====================================================

    for ax, degree in zip(
        axes,
        DEGREES
    ):

        data = results[degree]
        # =================================================
        # Datos experimentales
        # =================================================

        ax.scatter(
            frequency,
            impedance,
            s=30,
            alpha=0.70,
            zorder=5
        )

        # =================================================
        # Curva interpolante
        # =================================================

        ax.plot(
            data["x_dense"],
            data["y_dense"],
            linewidth=2.0,
            alpha=0.85,
            zorder=2
        )

        # =================================================
        # Nodos de interpolación
        # =================================================

        ax.scatter(
            data["x_nodes"],
            data["y_nodes"],
            s=70,
            marker="x",
            zorder=6
        )

        # =================================================
        # Escalado robusto adaptativo
        # =================================================

        y_min, y_max = (
            _compute_local_ylim(
                data["y_dense"]
            )
        )

        ax.set_ylim(
            y_min,
            y_max
        )

        # =================================================
        # Información numérica
        # =================================================

        title = (
            f"deg={degree}"
        )

        # Condicionamiento
        if (
            "condition_number"
            in data["summary"]
        ):

            cond = (
                data["summary"][
                    "condition_number"
                ]
            )

            title += (
                f"\ncond(V)="
                f"{cond:.2e}"
            )

        # Oscilación global
        title += (
            f"\nOsc="
            f"{data['oscillation']:.2e}"
        )

        ax.set_title(title)

        # =================================================
        # Grid
        # =================================================

        ax.grid(
            alpha=0.30
        )

    # =====================================================
    # Etiquetas globales
    # =====================================================

    fig.supxlabel(
        "Frequency [Hz]",
        y=0.05
    )

    fig.supylabel(
        r"$|Z|$ [$\Omega$]"
    )

    # =====================================================
    # Título general
    # =====================================================

    fig.suptitle(
        (
            f"{method_name}\n"
            "Runge Analysis"
        )
    )

    # =====================================================
    # Leyenda global
    # =====================================================

    fig.legend(
        [
            "Experimental data",
            r"$P_n(f)$",
            "Interpolation nodes"
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01)
    )

    # =====================================================
    # Ajuste de espaciado
    # =====================================================

    fig.subplots_adjust(
        bottom=0.18
    )

    # =====================================================
    # Exportación
    # =====================================================

    if save:

        fig.savefig(
            EXPORT_DIR /
            f"{save_name}.png",
            bbox_inches="tight"
        )

        fig.savefig(
            EXPORT_DIR /
            f"{save_name}.pdf",
            bbox_inches="tight"
        )

        fig.savefig(
            EXPORT_DIR /
            f"{save_name}.svg",
            bbox_inches="tight"
        )

    # =====================================================
    # Mostrar figura
    # =====================================================

    if show:
        plt.show()

    plt.close(fig)


def run_runge_analysis(
    save: bool = True,
    show: bool = True
) -> None:
    """
    Ejecuta análisis completo
    del fenómeno de Runge.
    """

    # =====================================================
    # Estilo global
    # =====================================================

    apply_plot_style()

    # =====================================================
    # Datos experimentales
    # =====================================================

    Z, f = load_impedance_data()

    # =====================================================
    # Lagrange
    # =====================================================

    lagrange_analyzer = (
        RungeAnalyzer(
            interpolator_class=(
                LagrangeInterpolator
            ),
            x=f,
            y=Z,
            degrees=DEGREES,
            node_strategy="sequential"
        )
    )

    lagrange_results = (
        lagrange_analyzer.run()
    )

    _plot_method_analysis(
        method_name="Lagrange",
        results=lagrange_results,
        frequency=f,
        impedance=Z,
        save_name="lagrange_runge",
        save=save,
        show=show
    )

    # =====================================================
    # Vandermonde
    # =====================================================

    matrix_analyzer = (
        RungeAnalyzer(
            interpolator_class=(
                MatrixPolynomialInterpolator
            ),
            x=f,
            y=Z,
            degrees=DEGREES,
            node_strategy="sequential"
        )
    )

    matrix_results = (
        matrix_analyzer.run()
    )

    _plot_method_analysis(
        method_name="Vandermonde",
        results=matrix_results,
        frequency=f,
        impedance=Z,
        save_name="vandermonde_runge",
        save=save,
        show=show
    )


if __name__ == "__main__":

    run_runge_analysis(
        save=True,
        show=True
    )
    