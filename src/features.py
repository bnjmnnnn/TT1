"""Construccion de la variable objetivo y features regionales (Fases 1 y 2).

Variable objetivo (Opcion B, recomendada en desarrollo_modelo.md):
    rate_{r,t} = ESTIMACION_{r,t} / sum_r ESTIMACION_{r,t}
Interpretacion: como se reparte territorialmente el stock migrante.
Suma 1 entre las 16 regiones para cada anio (check en build_datasets.py).
"""
import pandas as pd

# Codificacion de limpieza_sermig.py: SEXO H->1, M->0; EDAD IGNORADA -> -1.
FEMALE_SEX_CODE = 0
UNKNOWN_AGE_CODE = -1

# Nacionalidades principales del stock (top-5 en 2023), via codigo ISO.
TOP_COUNTRIES = {"VE": "venezuela", "PE": "peru", "CO": "colombia",
                 "HT": "haiti", "BO": "bolivia"}


def compute_rate(df, numerator_col, denominator_col):
    """Tasa de concentracion por unidad territorial-anio."""
    if (df[denominator_col] <= 0).any():
        raise ValueError("Denominador con ceros o negativos: revisar joins previos")
    rate = df[numerator_col] / df[denominator_col]
    if rate.isna().any():
        raise ValueError("Tasa con nulos: revisar integracion de fuentes")
    return rate


def aggregate_region_year(df):
    """Agrega los microdatos SERMIG a una fila por region-anio.

    Espera el CSV limpio (una fila por sexo x edad x pais x region x anio)
    y devuelve el panel con la estimacion total y la composicion del stock.
    """
    groups = df.groupby(["CODREGEO", "REGION", "ANIO"], as_index=False)
    panel = groups.agg(
        estimation=("ESTIMACION", "sum"),
        rraa_irregular=("RRAA_IRREGULAR", "sum"),
    )

    # % de mujeres del stock estimado
    women = (df[df["SEXO"] == FEMALE_SEX_CODE]
             .groupby(["CODREGEO", "ANIO"])["ESTIMACION"].sum()
             .rename("women_estimation"))
    panel = panel.merge(women, on=["CODREGEO", "ANIO"], how="left")
    panel["pct_women"] = panel["women_estimation"] / panel["estimation"]

    # Edad media ponderada por estimacion (excluye edad ignorada)
    with_age = df[df["EDAD_NUMERICA"] != UNKNOWN_AGE_CODE].copy()
    with_age["weighted_age"] = with_age["EDAD_NUMERICA"] * with_age["ESTIMACION"]
    age = with_age.groupby(["CODREGEO", "ANIO"]).agg(
        age_sum=("weighted_age", "sum"),
        est_sum=("ESTIMACION", "sum"),
    )
    panel = panel.merge(
        (age["age_sum"] / age["est_sum"]).rename("mean_age"),
        on=["CODREGEO", "ANIO"], how="left",
    )

    # % del stock en situacion irregular (RRAA irregular / estimacion total)
    panel["pct_irregular"] = panel["rraa_irregular"] / panel["estimation"]

    # % por nacionalidad principal
    for iso, name in TOP_COUNTRIES.items():
        country = (df[df["PAIS_ISO"] == iso]
                   .groupby(["CODREGEO", "ANIO"])["ESTIMACION"].sum()
                   .rename(f"est_{name}"))
        panel = panel.merge(country, on=["CODREGEO", "ANIO"], how="left")
        panel[f"pct_{name}"] = panel[f"est_{name}"].fillna(0) / panel["estimation"]
        panel = panel.drop(columns=[f"est_{name}"])

    return panel.drop(columns=["women_estimation", "rraa_irregular"])


def add_lags(panel, columns):
    """Crea <col>_lag1 con el valor del anio anterior de la misma region.

    El rezago es legitimo como predictor: se conoce al momento de predecir.
    Costo: se pierde el primer anio del panel (queda solo como fuente del lag).
    """
    panel = panel.sort_values(["CODREGEO", "ANIO"]).copy()
    for col in columns:
        panel[f"{col}_lag1"] = panel.groupby("CODREGEO")[col].shift(1)
    return panel
