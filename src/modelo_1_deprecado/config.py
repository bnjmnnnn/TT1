"""Configuracion del modelo 1 (baseline): rutas, features y target.

Constantes de uso exclusivo de este modelo. SEED y TEST_YEAR se repiten
literalmente aqui (mismos valores que src/common/config.py, que usan
modelo_2_regional_v2 y modelo_3_conteo) para evitar un import cruzado
entre carpetas solo por dos constantes.
"""
from pathlib import Path

SEED = 42
TEST_YEAR = 2023

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parents[1]
OUTPUTS_DIR = AQUI / "outputs"
DATASET_REGION = OUTPUTS_DIR / "dataset_region.csv"

# Fuente cruda -- NOTA: este archivo ya no existe en el repo (se perdio en
# una reorganizacion anterior a esta rama). build_datasets.py no se puede
# regenerar desde cero hoy; DATASET_REGION (su output ya calculado) sigue
# disponible y es lo que usa train.py y lo que lee modelo_2_regional_v2.
RAW_SERMIG = ROOT / "8. baseregiones_limpio.csv"

# --- Protocolo temporal (Fase 4.1) ---
# El anio 2020 se usa unicamente como fuente del rezago (lag), por lo que
# el panel modelable es 2021-2023.
PANEL_START = 2020

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
