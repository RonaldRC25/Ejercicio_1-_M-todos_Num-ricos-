# ==========================================
# utils/data_loader.py
# ==========================================

import pandas as pd
import numpy as np

from utils.config import DATA_FILE


def load_impedance_data():
    """
    Carga datos biomédicos desde CSV.

    El archivo debe contener columnas:
    - Z : impedancia
    - f : frecuencia

    Modelo:
        Z(f)

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        impedance, frequency
    """

    try:

        # Leer archivo CSV
        df = pd.read_csv(DATA_FILE)

        # Validar columnas requeridas
        required_columns = ["Z", "f"]

        for column in required_columns:
            if column not in df.columns:
                raise ValueError(
                    f"Falta la columna requerida: '{column}'"
                )

        # Convertir columnas a arrays numpy
        impedance = df["Z"].to_numpy(dtype=float)

        frequency = df["f"].to_numpy(dtype=float)

        return impedance, frequency

    except FileNotFoundError:
        raise FileNotFoundError(
            f"No se encontró el archivo:\n{DATA_FILE}"
        )

    except Exception as error:
        raise RuntimeError(
            f"Error cargando datos: {error}"
        )