"""
===========================================================
matrix_graph.py
===========================================================

Professional visualization for matrix-based
polynomial interpolation.

Compatible with:
    - inter_matrix_method.py
    - plot_config.py

Features
--------
- Global polynomial visualization
- Runge phenomenon analysis
- Dense interpolation plotting
- Condition number visualization
- Scientific export formats

Author
------
Biomedical Numerical System Project
===========================================================
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from pathlib import Path
from typing import Optional

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

# Export directory
EXPORT_DIR = Path(
    EXPORTS_DIR / "interpolation" / "matrix_graph"
)
EXPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print(f"Export directory: {EXPORT_DIR}")

def plot_matrix_interpolation(
    degree: Optional[int] = None,
    dense_points: int = 5000,
    figsize: tuple[int, int] = (10, 6),
    node_strategy: str = "sequential",
    save: bool = True,
    show: bool = True
) -> None:
    """
    Plot matrix-based polynomial interpolation.
    """

    # Apply scientific style
    apply_plot_style()

    # Load experimental data
    Z, f = load_impedance_data()

    # Create interpolator
    interpolator = (
        MatrixPolynomialInterpolator(
            x=f,
            y=Z,
            degree=degree,
            dense_points=dense_points,
            node_strategy=node_strategy
        )
    )

    # Generate dense interpolation mesh
    f_dense, Z_dense = (
        interpolator.generate_dense_mesh()
    )

    # Create figure
    fig, ax = plt.subplots(
        figsize=figsize
    )

    # Experimental data
    ax.scatter(
        f,
        Z,
        s=45,
        color="black",
        zorder=5,
        label="Experimental data"
    )

    # Interpolation curve
    ax.plot(
        f_dense,
        Z_dense,
        linewidth=2.2,
        label=(
            f"Matrix "
            f"(deg={interpolator.degree}, "
            f"nodes={interpolator.n_nodes})"
        )
    )

    # Active interpolation nodes
    ax.scatter(
        interpolator.x_nodes,
        interpolator.y_nodes,
        s=65,
        marker="x",
        zorder=6,
        label="Interpolation nodes"
    )

    # Labels
    ax.set_title(
        "Matrix Polynomial Interpolation"
    )

    ax.set_xlabel(
        "Frequency [Hz]"
    )

    ax.set_ylabel(
        r"$|Z|$ [$\Omega$]"
    )

    # Condition number annotation
    ax.text(
        0.02,
        0.95,
        (
            r"$cond(V)$ = "
            f"{interpolator.condition_number:.2e}"
        ),
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round",
            alpha=0.15
        )
    )

    ax.legend()

    fig.tight_layout()

    # Export figures
    if save:

        degree_name = (
            "global"
            if degree is None
            else f"deg_{degree}"
        )

        fig.savefig(
            EXPORT_DIR /
            f"matrix_{degree_name}.png",
            bbox_inches="tight"
        )

        fig.savefig(
            EXPORT_DIR /
            f"matrix_{degree_name}.pdf",
            bbox_inches="tight"
        )

        fig.savefig(
            EXPORT_DIR /
            f"matrix_{degree_name}.svg",
            bbox_inches="tight"
        )

    # Display figure
    if show:
        plt.show()

    plt.close(fig)


def quick_matrix_analysis() -> None:
    """
    Quick visualization for debugging/testing.
    """

    plot_matrix_interpolation(
        degree=10,
        dense_points=5000,
        node_strategy="sequential",
        save=False,
        show=True
    )


if __name__ == "__main__":

    # Example visualization
    plot_matrix_interpolation(
        degree=29,
        dense_points=5000,
        node_strategy="sequential",
        save=True,
        show=True
    )