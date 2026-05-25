"""
===========================================================
loo_validation_plot.py
===========================================================

Visualización profesional de validación
Leave-One-Out (LOO).

Author
------
Biomedical Numerical System Project
===========================================================
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from matplotlib.lines import (
    Line2D
)

from pathlib import Path

from core.validation.loo_validation import (
    LOOValidator
)

from core.interpolation.Lagrange import (
    LagrangeInterpolator
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

# Directorio de exportación
EXPORT_DIR = Path(
    EXPORTS_DIR / "loo_validation"
)

EXPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def plot_loo_validation(
    random_state: int = 42,
    save: bool = True,
    show: bool = True
) -> None:
    """
    Genera visualización profesional
    de validación Leave-One-Out.
    """

    # Aplicar estilo global
    apply_plot_style()

    # Datos experimentales
    Z, f = load_impedance_data()

    # Validador LOO
    validator = LOOValidator(
        interpolator_class=LagrangeInterpolator,
        x=f,
        y=Z,
        degree=4,
        window_size=5,
        random_state=random_state
    )

    results = validator.run()

    # Figura principal
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 9),
        sharex=True,
        sharey=True
    )

    axes = axes.flatten()

    # =====================================================
    # Ventana local completa
    # =====================================================

    ax0 = axes[0]

    ax0.scatter(
        results["window_x"],
        results["window_y"],
        s=70,
        alpha=0.85
    )

    ax0.set_title(
        "Ventana local seleccionada"
    )

    # =====================================================
    # Iteraciones Leave-One-Out
    # =====================================================

    for idx, iteration in enumerate(
        results["iterations"]
    ):

        ax = axes[idx + 1]

        # Curva interpolante
        ax.plot(
            iteration["x_dense"],
            iteration["y_dense"]
        )

        # Puntos de entrenamiento
        ax.scatter(
            iteration["x_train"],
            iteration["y_train"],
            s=55
        )

        # Punto removido
        ax.scatter(
            iteration["removed_x"],
            iteration["removed_y"],
            s=90,
            color="red",
            marker="x"
        )

        # Punto predicho
        ax.scatter(
            iteration["removed_x"],
            iteration["predicted_y"],
            s=85,
            color="green",
            marker="*"
        )

        # Título compacto
        ax.set_title(
            (
                f"LOO #{idx+1} | "
                f"εr="
                f"{iteration['relative_error']:.2f}%"
            )
        )

    # =====================================================
    # Etiquetas globales
    # =====================================================

    fig.supxlabel(
        "Frecuencia [Hz]",
        y=0.05
    )

    fig.supylabel(
        r"$|Z|$ [$\Omega$]"
    )

    # =====================================================
    # Leyenda global
    # =====================================================

    legend_elements = [

        Line2D(
            [0],
            [0],
            linewidth=2.2,
            label=r"$P_4(f)$"
        ),

        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=8,
            label="Entrenamiento"
        ),

        Line2D(
            [0],
            [0],
            marker="x",
            linestyle="None",
            color="red",
            markersize=9,
            label="Extraído"
        ),

        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="None",
            color="green",
            markersize=10,
            label="Predicción"
        )
    ]

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        frameon=False
    )

    # Espacio para leyenda inferior
    fig.subplots_adjust(
        bottom=0.22
    )

    # =====================================================
    # Métricas globales
    # =====================================================

    fig.suptitle(
        (
            "Leave-One-Out Validation\n"
            f"MAE={results['mae']:.4f} | "
            f"RMSE={results['rmse']:.4f} | "
            f"Mean εr="
            f"{results['mean_relative_error']:.2f}%"
        )
    )

    # =====================================================
    # Exportación
    # =====================================================

    if save:

        fig.savefig(
            EXPORT_DIR /
            "loo_validation.png",
            bbox_inches="tight"
        )

        fig.savefig(
            EXPORT_DIR /
            "loo_validation.pdf",
            bbox_inches="tight"
        )

        fig.savefig(
            EXPORT_DIR /
            "loo_validation.svg",
            bbox_inches="tight"
        )

    if show:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":

    plot_loo_validation(
        random_state=42,
        save=False,
        show=True
    )