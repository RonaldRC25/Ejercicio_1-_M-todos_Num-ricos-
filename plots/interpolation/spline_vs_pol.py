"""
===========================================================
spline_plot.py
===========================================================

Comparison between:
- Natural cubic spline
- Polynomial interpolation

Features
--------
- Dense mesh comparison
- Interpolation at 1000 Hz
- Scientific visualization
- Export support

Author
------
Biomedical Numerical System Project
===========================================================
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from pathlib import Path

from core.interpolation.cubic_spline import (
    CubicSplineInterpolator
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
# Export directory
# =========================================================

EXPORT_DIR = Path(
    EXPORTS_DIR / "spline_graph"
)

EXPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)
print(f"Export directory: {EXPORT_DIR}")

def plot_spline_interpolation(
    degree: int = 10,
    polynomial_method: str = "lagrange",
    spline_bc: str = "natural",
    target_x: float = 1000.0,
    dense_points: int = 5000,
    compare: bool = True,
    save: bool = True,
    show: bool = True
) -> dict:
    """
    Compare spline and polynomial interpolation.
    """

    apply_plot_style()

    # =====================================================
    # Load experimental data
    # =====================================================

    Z, f = load_impedance_data()

    # =====================================================
    # Spline interpolator
    # =====================================================

    spline = (
        CubicSplineInterpolator(
            x=f,
            y=Z,
            bc_type=spline_bc,
            dense_points=dense_points
        )
    )

    f_spline, Z_spline = (
        spline.generate_dense_mesh()
    )

    z_spline = spline.evaluate(
        target_x
    )

    # =====================================================
    # Polynomial interpolator
    # =====================================================

    if polynomial_method == "lagrange":

        poly = (
            LagrangeInterpolator(
                x=f,
                y=Z,
                degree=degree,
                dense_points=dense_points,
                node_strategy="nearest",
                target_x=target_x
            )
        )

        poly_label = (
            f"Lagrange deg={degree}"
        )

    elif polynomial_method == "vandermonde":

        poly = (
            MatrixPolynomialInterpolator(
                x=f,
                y=Z,
                degree=degree,
                dense_points=dense_points,
                node_strategy="nearest",
                target_x=target_x
            )
        )

        poly_label = (
            f"Vandermonde deg={degree}"
        )

    else:
        raise ValueError(
            "Unknown polynomial method."
        )

    f_poly, Z_poly = (
        poly.generate_dense_mesh()
    )

    z_poly = poly.evaluate(
        target_x
    )

    # =====================================================
    # Figure
    # =====================================================

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    # =====================================================
    # Experimental data
    # =====================================================

    ax.scatter(
        f,
        Z,
        s=40,
        alpha=0.75,
        label="Experimental data",
        zorder=5
    )

    # =====================================================
    # Spline curve
    # =====================================================

    ax.plot(
        f_spline,
        Z_spline,
        linewidth=2.5,
        label="Natural cubic spline",
        zorder=3
    )

    # =====================================================
    # Polynomial curve
    # =====================================================

    if compare:

        ax.plot(
            f_poly,
            Z_poly,
            linewidth=2.0,
            linestyle="--",
            label=poly_label,
            zorder=2
        )

    # =====================================================
    # Target frequency marker
    # =====================================================

    ax.scatter(
        target_x,
        z_spline,
        s=120,
        marker="x",
        label=(
            f"Spline({target_x:.0f} Hz)"
            f" = {z_spline:.3f}"
        ),
        zorder=6
    )

    if compare:

        ax.scatter(
            target_x,
            z_poly,
            s=120,
            marker="o",
            facecolors="none",
            label=(
                f"Poly({target_x:.0f} Hz)"
                f" = {z_poly:.3f}"
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
        r"$|Z|$ [$\Omega$]"
    )

    # =====================================================
    # Title
    # =====================================================

    ax.set_title(
        (
            "Natural Cubic Spline "
            "vs Polynomial Interpolation"
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
    # Export
    # =====================================================

    if save:

        fig.savefig(
            EXPORT_DIR /
            f"spline_comparison_{degree}.png",
            bbox_inches="tight"
        )

        fig.savefig(
            EXPORT_DIR /
            f"spline_comparison_{degree}.pdf",
            bbox_inches="tight"
        )

        fig.savefig(
            EXPORT_DIR /
            f"spline_comparison_{degree}.svg",
            bbox_inches="tight"
        )
    print(f"Figures saved to: {EXPORT_DIR}/spline_comparison_{degree}.*")

    if show:
        plt.show()

    plt.close(fig)

    return {
        "f_spline":
        f_spline,

        "Z_spline":
        Z_spline,

        "f_poly":
        f_poly,

        "Z_poly":
        Z_poly,

        "spline_1000":
        float(z_spline),

        "poly_1000":
        float(z_poly)
    }


if __name__ == "__main__":

    plot_spline_interpolation(
        degree=10,
        polynomial_method="lagrange",
        target_x=1000.0,
        dense_points=5000,
        compare=True,
        save=True,
        show=True
    )