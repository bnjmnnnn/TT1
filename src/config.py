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
# Se alimenta el modelo con toda la historia disponible del SERMIG
# (2018-2023): XGBoost duplica su R2 de test y las predicciones casi no
# cambian (ver docs/comparacion_filtro_temporal.md). El primer anio se usa
# unicamente como fuente del rezago (lag) -> panel modelable 2019-2023.
PANEL_START = 2018
TEST_YEAR = 2023  # hold-out temporal: ultimo anio disponible

# Region 17 = "REGION IGNORADA": no es una unidad territorial predecible
# (su share fluctua entre 3.5% y 13% del stock). Se excluye y la tasa se
# define sobre el stock regionalizado (suma de las 16 regiones).
UNKNOWN_REGION_CODE = 17

# --- Horizonte de prediccion ---
# 2024-2026 se predicen de forma recursiva (encadenando rezagos predichos).
# 2027-2028 quedan pendientes como trabajo futuro (el error acumulado de
# la recursion crece con cada paso y debe evaluarse antes de extenderlo).
FORECAST_END = 2026

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
