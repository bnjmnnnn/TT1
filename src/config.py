"""Configuracion unica del proyecto (Fase 0 de desarrollo_modelo.md).

Toda constante compartida entre notebooks y scripts vive aqui.
"""
from pathlib import Path

# --- Rutas (siempre relativas a la raiz del repo) ---
ROOT = Path(__file__).resolve().parents[1]
RAW_SERMIG = ROOT / "8. baseregiones_limpio.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUTS_DIR = ROOT / "data" / "outputs"
DATASET_REGION = PROCESSED_DIR / "dataset_region.csv"

# --- Reproducibilidad ---
SEED = 42

# --- Protocolo temporal (Fase 4.1) ---
# Filtro del profesor: solo anios >= 2020. El anio 2020 se usa unicamente
# como fuente del rezago (lag), por lo que el panel modelable es 2021-2023.
PANEL_START = 2020
TEST_YEAR = 2023  # hold-out temporal: ultimo anio disponible

# Region 17 = "REGION IGNORADA": no es una unidad territorial predecible
# (su share fluctua entre 3.5% y 13% del stock). Se excluye y la tasa se
# define sobre el stock regionalizado (suma de las 16 regiones).
UNKNOWN_REGION_CODE = 17

# --- Criterios (reunion con el profesor) ---
VIF_THRESHOLD = 10
R2_SUCCESS_THRESHOLD = 0.60

# --- Features ---
# Todas las features estan rezagadas a t-1 para evitar fuga de informacion
# contemporanea: al predecir el anio t solo se usa lo conocido en t-1.
# Set inicial conservador (N_train ~32 -> max 5-8 features, regla Fase 3).
# Las demas columnas *_lag1 quedan en dataset_region.csv como candidatas
# para la seleccion con evidencia (Fase 3 / tarea T5).
FEATURES = [
    "rate_lag1",
    "pct_women_lag1",
    "mean_age_lag1",
    "pct_irregular_lag1",
    "pct_venezuela_lag1",
]

TARGET = "rate"
