"""
Professional visualization for Lagrange interpolation.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from pathlib import Path
from typing import Optional

from core.interpolation.Lagrange import (
    LagrangeInterpolator
)

from utils.data_loader import (
    load_impedance_data
)

from plots.styles.plot_config import (
    apply_plot_style
)
from utils.config import (
    EXPORTS_DIR
)

# Export directory
EXPORT_DIR = Path(
    EXPORTS_DIR / "interpolation" / "lagrange_graph"
)

print(f"Export directory: {EXPORT_DIR}")

def plot_lagrange_interpolation(
    degree: Optional[int] = None,
    dense_points: int = 5000,
    figsize: tuple[int, int] = (10, 6),
    save: bool = True,
    show: bool = True
) -> None:
    """
    Plot global Lagrange interpolation.
    """

    # Apply scientific style
    apply_plot_style()

    # Load experimental data
    Z, f = load_impedance_data()
    print(Z[:15])
    print(f[:15])

    # Create interpolator
    interpolator = (
        LagrangeInterpolator(
            x=f,
            y=Z,
            degree=degree,
            dense_points=dense_points
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
            f"Lagrange "
            f"(deg={interpolator.degree}, "
            f"nodes={interpolator.n_nodes})"
        )
    )

    # Interpolation nodes used
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
        "Global Lagrange Interpolation"
    )

    ax.set_xlabel(
        "Frequency [Hz]"
    )

    ax.set_ylabel(
        r"$|Z|$ [$\Omega$]"
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
            f"lagrange_{degree_name}.png",
            bbox_inches="tight"
        )

        fig.savefig(
            EXPORT_DIR /
            f"lagrange_{degree_name}.pdf",
            bbox_inches="tight"
        )

        fig.savefig(
            EXPORT_DIR /
            f"lagrange_{degree_name}.svg",
            bbox_inches="tight"
        )

    # Display figure
    if show:
        plt.show()

    plt.close(fig)


def quick_lagrange_analysis() -> None:
    """
    Quick visualization for debugging/testing.
    """

    plot_lagrange_interpolation(
        degree=10,
        dense_points=5000,
        save=False,
        show=True
    )


if __name__ == "__main__":

    # Example visualization
    plot_lagrange_interpolation(
        degree=29
    )