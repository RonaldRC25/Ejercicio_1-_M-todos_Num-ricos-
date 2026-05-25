"""
Global plotting configuration.
"""

from __future__ import annotations

import matplotlib.pyplot as plt


def apply_plot_style() -> None:
    """
    Apply consistent scientific plotting style.
    """

    plt.rcParams.update({

        # Figure
        "figure.figsize": (10, 6),
        "figure.dpi": 120,
        "savefig.dpi": 300,

        # Fonts
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,

        # Grid
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",

        # Lines
        "lines.linewidth": 2.2,

        # Axes
        "axes.spines.top": False,
        "axes.spines.right": False,

        # Layout
        "figure.autolayout": True
    })