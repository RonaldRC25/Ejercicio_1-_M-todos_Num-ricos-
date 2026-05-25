"""
===========================================================
safe_band_plot.py
===========================================================

Visualización de banda segura de operación.

Características
----------------
- Regiones:
      seguras
      inseguras
- Umbral Z_th
- Spline cúbico
- Sombreado dinámico

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


def plot_safe_band(
    z_threshold: float = 150.0,
    dense_points: int = 5000
) -> None:
    """
    Grafica banda segura:

        |Z|(f) <= Z_th
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
    # Máscaras operación
    # =====================================================

    safe_mask = (
        Z_dense <= z_threshold
    )

    unsafe_mask = (
        Z_dense > z_threshold
    )

    # =====================================================
    # Figura
    # =====================================================

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    # =====================================================
    # Curva spline
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
    # Región segura
    # =====================================================

    ax.fill_between(
        f_dense,
        Z_dense,
        z_threshold,
        where=safe_mask,
        alpha=0.35,
        interpolate=True,
        label="Banda segura"
    )

    # =====================================================
    # Región insegura
    # =====================================================

    ax.fill_between(
        f_dense,
        Z_dense,
        z_threshold,
        where=unsafe_mask,
        alpha=0.25,
        interpolate=True,
        label="Banda insegura"
    )

    # =====================================================
    # Datos experimentales
    # =====================================================

    ax.scatter(
        f,
        Z,
        s=35,
        alpha=0.75,
        label="Datos experimentales"
    )

    # =====================================================
    # Configuración visual
    # =====================================================

    ax.set_title(
        "Safe Operating Band",
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
    # Leyenda sin duplicados
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
        EXPORT_DIR / "safe_band_plot"
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

    plot_safe_band()