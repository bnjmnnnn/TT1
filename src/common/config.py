"""Constantes compartidas por MAS de un modelo (SEED, TEST_YEAR).

Todo lo demas (rutas de datos, features, target) es especifico de cada
modelo y vive en su propia carpeta (ver modelo_1_deprecado/config.py para
las constantes que eran de uso exclusivo del modelo 1).
"""

# --- Reproducibilidad ---
SEED = 42

# --- Protocolo temporal ---
# Filtro del profesor: solo anios >= 2020. El anio 2020 se usa unicamente
# como fuente del rezago (lag) en el panel del modelo 1.
TEST_YEAR = 2023  # hold-out temporal: ultimo anio disponible, mismo en los 3 modelos
