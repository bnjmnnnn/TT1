"""Construye outputs/dataset_region.csv (Fase 2 de desarrollo_modelo.md).

Una fila por region-anio con la variable objetivo (rate) y las features
candidatas rezagadas. Incluye los checks de calidad del checklist Fase 2:
se ejecutan siempre y el script falla si alguno no se cumple.

Uso:  python src/modelo_1_deprecado/build_datasets.py
"""
import sys

import pandas as pd

import config
from features import aggregate_region_year, add_lags, compute_rate, TOP_COUNTRIES

# Columnas de composicion que se rezagan a t-1 (junto con la tasa)
COMPOSITION_COLS = ["pct_women", "mean_age", "pct_irregular"] + [
    f"pct_{name}" for name in TOP_COUNTRIES.values()
]


def build_region_dataset():
    raw = pd.read_csv(config.RAW_SERMIG)

    # Filtro temporal del profesor (>= 2020) y exclusion de REGION IGNORADA
    raw = raw[raw["ANIO"] >= config.PANEL_START]
    raw = raw[raw["CODREGEO"] != config.UNKNOWN_REGION_CODE]

    panel = aggregate_region_year(raw)

    # Variable objetivo: share del stock regionalizado por anio (Opcion B)
    national_total = panel.groupby("ANIO")["estimation"].transform("sum")
    panel["national_total"] = national_total
    panel["rate"] = compute_rate(panel, "estimation", "national_total")

    panel = add_lags(panel, ["rate"] + COMPOSITION_COLS)

    # Se pierde el primer anio del panel (2020 solo sirve como fuente del lag)
    n_before = len(panel)
    dataset = panel.dropna(subset=["rate_lag1"]).reset_index(drop=True)
    n_lost = n_before - len(dataset)

    # Columnas contemporaneas de composicion NO van al dataset final:
    # solo se permiten features conocidas en t-1 (regla anti-leakage).
    keep = (["CODREGEO", "REGION", "ANIO", "estimation", "rate"]
            + [f"{c}_lag1" for c in ["rate"] + COMPOSITION_COLS])
    dataset = dataset[keep]

    run_quality_checks(panel, dataset, n_lost)

    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(config.DATASET_REGION, index=False)
    print(f"\nOK -> {config.DATASET_REGION}")
    print(f"Dimensiones: {dataset.shape[0]} filas x {dataset.shape[1]} columnas")
    print(f"Anios modelables: {sorted(dataset['ANIO'].unique())} "
          f"({n_lost} filas de {config.PANEL_START} usadas solo como lag)")
    return dataset


def run_quality_checks(panel, dataset, n_lost):
    """Checklist Fase 2: cada check corta la ejecucion si falla."""
    # Opcion B: la suma de tasas por anio debe ser ~1 (tolerancia 0.001)
    rate_sums = panel.groupby("ANIO")["rate"].sum()
    assert ((rate_sums - 1).abs() < 1e-3).all(), \
        f"Suma de tasas por anio != 1: {rate_sums.to_dict()}"

    # Sin duplicados region-anio
    assert dataset.duplicated(["CODREGEO", "ANIO"]).sum() == 0, \
        "Duplicados region-anio en el dataset"

    # Cero nulos en target y features
    nulls = dataset.isna().sum()
    assert nulls.sum() == 0, f"Nulos inesperados:\n{nulls[nulls > 0]}"

    # N filas = N regiones x N anios modelables (menos filas perdidas por lag)
    n_regions = dataset["CODREGEO"].nunique()
    n_years = dataset["ANIO"].nunique()
    assert len(dataset) == n_regions * n_years, \
        "Panel incompleto: falta alguna combinacion region-anio"
    assert n_lost == n_regions, \
        "Las filas perdidas por rezago no corresponden a un anio completo"

    # Todas las columnas numericas con dtype numerico (no object)
    numeric_cols = [c for c in dataset.columns if c != "REGION"]
    bad = [c for c in numeric_cols
           if not pd.api.types.is_numeric_dtype(dataset[c])]
    assert not bad, f"Columnas no numericas: {bad}"

    # Rezago verificado a mano para una region (Valparaiso, cod. 6):
    # la fila 2022 debe traer en rate_lag1 el valor real de 2021.
    valpo_2021 = panel.query("CODREGEO == 6 and ANIO == 2021")["rate"].iloc[0]
    valpo_lag_2022 = dataset.query("CODREGEO == 6 and ANIO == 2022")["rate_lag1"].iloc[0]
    assert valpo_2021 == valpo_lag_2022, "Rezago mal construido (Valparaiso 2022)"

    # Sanity check externo: la RM debe concentrar la mayor tasa del panel
    last_year = dataset["ANIO"].max()
    top_region = (dataset[dataset["ANIO"] == last_year]
                  .sort_values("rate", ascending=False)["REGION"].iloc[0])
    assert "METROPOLITANA" in top_region.upper(), \
        f"Region con mayor tasa inesperada: {top_region}"

    print("Checks de calidad Fase 2: OK (suma tasas ~1, sin duplicados, "
          "sin nulos, panel completo, dtypes, lag verificado, RM lidera)")


if __name__ == "__main__":
    sys.exit(0 if build_region_dataset() is not None else 1)
