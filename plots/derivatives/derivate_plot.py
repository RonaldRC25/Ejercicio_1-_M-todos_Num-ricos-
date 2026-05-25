"""
===========================================================
derivative_plot.py
===========================================================

Spline derivative visualization.

Features
--------
- First derivative plotting
- Minimum detection visualization
- Stability analysis annotation
- Scientific export

Author
------
Biomedical Numerical System Project
===========================================================
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from pathlib import Path

from core.derivatives.spline_derivative import (
    SplineDerivativeAnalyzer
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
# Export directory
# =========================================================

EXPORT_DIR = Path(
    EXPORTS_DIR / "derivative_graph"
)

EXPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def plot_spline_derivative(
    dense_points: int = 5000,
    save: bool = True,
    show: bool = True
) -> dict:
    """
    Plot spline first derivative.
    """

    apply_plot_style()

    # =====================================================
    # Load experimental data
    # =====================================================

    Z, f = load_impedance_data()

    # =====================================================
    # Derivative analyzer
    # =====================================================

    analyzer = (
        SplineDerivativeAnalyzer(
            x=f,
            y=Z,
            dense_points=dense_points
        )
    )

    # =====================================================
    # Dense derivative mesh
    # =====================================================

    f_dense, dZ_dense = (
        analyzer.generate_dense_derivative_mesh()
    )

    # =====================================================
    # Minimum analysis
    # =====================================================

    result = (
        analyzer.stability_analysis()
    )

    f_min = result["f_min"]

    z_min = result["z_min"]

    d2z = result[
        "second_derivative"
    ]

    stability = result[
        "stability"
    ]

    # =====================================================
    # Figure
    # =====================================================

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    # =====================================================
    # First derivative curve
    # =====================================================

    ax.plot(
        f_dense,
        dZ_dense,
        linewidth=2.5,
        label=r"$d|Z|/df$"
    )

    # =====================================================
    # Zero reference
    # =====================================================

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.2
    )

    # =====================================================
    # Minimum marker
    # =====================================================

    ax.scatter(
        f_min,
        0.0,
        s=120,
        marker="x",
        label=(
            f"Minimum at "
            f"{f_min:.2f} Hz"
        ),
        zorder=6
    )

    # =====================================================
    # Labels
    # =====================================================

    ax.set_xlabel(
        "Frequency [Hz]"
    )

    ax.set_ylabel(
        r"$d|Z|/df$"
    )

    # =====================================================
    # Title
    # =====================================================

    ax.set_title(
        (
            "Spline First Derivative\n"
            f"$d^2|Z|/df^2={d2z:.3e}$ "
            f"({stability})"
        )
    )

    # =====================================================
    # Grid
    # =====================================================

    ax.grid(
        alpha=0.30
    )

    # =====================================================
    # Legend
    # =====================================================

    ax.legend()

    # =====================================================
    # Export figures
    # =====================================================

    if save:

        fig.savefig(
            EXPORT_DIR /
            "spline_derivative.png",
            bbox_inches="tight"
        )

        fig.savefig(
            EXPORT_DIR /
            "spline_derivative.pdf",
            bbox_inches="tight"
        )

        fig.savefig(
            EXPORT_DIR /
            "spline_derivative.svg",
            bbox_inches="tight"
        )

    if show:
        plt.show()

    plt.close(fig)

    return result


if __name__ == "__main__":

    result = (
        plot_spline_derivative(
            dense_points=5000,
            save=True,
            show=True
        )
    )

    print(result)