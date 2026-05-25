# ==========================================
# utils/config.py
# ==========================================

from pathlib import Path

# Directorio raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

# ==========================================
# RUTAS DE DATOS
# ==========================================

DATA_DIR = BASE_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

EXPORTS_DIR = DATA_DIR / "exports"

# Archivo principal CSV
DATA_FILE = RAW_DATA_DIR / "impedance_data.csv"

# ==========================================
# CONFIGURACIÓN GENERAL
# ==========================================

APP_NAME = "Biomedical Numerical System"

APP_VERSION = "1.0.0"

# ==========================================
# CONFIGURACIÓN DE GRÁFICAS
# ==========================================

PLOT_STYLE = "default"

FIGURE_SIZE = (10, 6)

DPI = 120