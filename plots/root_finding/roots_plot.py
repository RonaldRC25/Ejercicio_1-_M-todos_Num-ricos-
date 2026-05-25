"""
===========================================================
roots_plot.py
===========================================================

Visualización de raíces usando spline cúbico.

Características
----------------
- Curva spline interpolante
- Umbral crítico Z_th
- Raíces detectadas
- Comparación:
      Bisección
      Newton-Raphson

===========================================================
"""

from __future__ import annotations

import numpy as np

import matplotlib.pyplot as plt

from pathlib import Path

from utils.data_loader import (
    load_impedance_data
)

from utils.config import (
    EXPORTS_DIR
)

from plots.styles.plot_config import (
    apply_plot_style
)

from core.root_finding.spline_roots import (
    SplineRootFinder
)

# =========================================================
# Directorio exportación
# =========================================================

EXPORT_DIR = Path(
    EXPORTS_DIR / "root_finding"
)

EXPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def plot_roots(
    z_threshold: float = 150.0,
    dense_points: int = 5000
) -> None:
    """
    Grafica raíces encontradas mediante:

        |Z|(f)-Z_th=0
    """

    apply_plot_style()

    # =====================================================
    # Datos experimentales
    # =====================================================

    Z, f = load_impedance_data()

    # =====================================================
    # Root finder
    # =====================================================

    finder = (
        SplineRootFinder(
            x=f,
            y=Z,
            z_threshold=z_threshold,
            dense_points=dense_points
        )
    )

    # =====================================================
    # Malla fina spline
    # =====================================================

    f_dense = np.linspace(
        f.min(),
        f.max(),
        dense_points
    )

    Z_dense = finder.spline.evaluate(
        f_dense
    )

    # =====================================================
    # Resolución raíces
    # =====================================================

    roots_bisection = (
        finder.solve_bisection()
    )

    roots_newton = (
        finder.solve_newton()
    )

    # =====================================================
    # Figura principal
    # =====================================================

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    # =====================================================
    # Datos experimentales
    # =====================================================

    ax.scatter(
        f,
        Z,
        s=40,
        alpha=0.75,
        label="Datos experimentales"
    )

    # =====================================================
    # Spline cúbico
    # =====================================================

    ax.plot(
        f_dense,
        Z_dense,
        linewidth=2.2,
        label="Spline cúbico"
    )

    # =====================================================
    # Umbral crítico
    # =====================================================

    ax.axhline(
        y=z_threshold,
        linestyle="--",
        linewidth=2,
        label=rf"$Z_{{th}}={z_threshold:.0f}\ \Omega$"
    )

    # =====================================================
    # Raíces bisección
    # =====================================================

    for root_data in roots_bisection:

        root = root_data["root"]

        ax.scatter(
            root,
            z_threshold,
            marker="x",
            s=120,
            linewidths=2.5,
            label="Raíz bisección"
        )

    # =====================================================
    # Raíces Newton
    # =====================================================

    for root_data in roots_newton:

        root = root_data["root"]

        ax.scatter(
            root,
            z_threshold,
            marker="o",
            s=90,
            facecolors="none",
            linewidths=2.0,
            label="Raíz Newton"
        )

    # =====================================================
    # Configuración visual
    # =====================================================

    ax.set_title(
        "Spline Root Finding",
        fontsize=20,
        pad=15
    )

    ax.set_xlabel(
        "Frecuencia [Hz]",
        fontsize=16
    )

    ax.set_ylabel(
        r"$|Z|$ [$\Omega$]",
        fontsize=16
    )

    ax.grid(
        alpha=0.35
    )

    # =====================================================
    # Evitar duplicación leyendas
    # =====================================================

    handles, labels = (
        ax.get_legend_handles_labels()
    )

    unique = dict(
        zip(labels, handles)
    )

    ax.legend(
        unique.values(),
        unique.keys(),
        fontsize=12,
        loc="best"
    )

    plt.tight_layout()

    # =====================================================
    # Exportación
    # =====================================================

    export_base = (
        EXPORT_DIR / "roots_plot"
    )

    fig.savefig(
        f"{export_base}.png",
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        f"{export_base}.pdf",
        bbox_inches="tight"
    )

    fig.savefig(
        f"{export_base}.svg",
        bbox_inches="tight"
    )

    plt.show()


if __name__ == "__main__":

    plot_roots()