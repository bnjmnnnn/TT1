"""Prediccion recursiva del horizonte futuro (2024-2026).

Esquema:
  1. Los modelos finales se reentrenan con TODO el panel disponible
     (2021-2023): el protocolo train/test de train.py ya cumplio su rol
     de evaluacion honesta y sus metricas quedan en metricas.csv.
  2. Se predice de forma encadenada: 2024 usa los datos reales de 2023;
     2025 usa el share predicho de 2024; 2026 usa el de 2025.
  3. Supuesto declarado: la composicion demografica (pct_women, mean_age,
     pct_irregular, pct_venezuela) se congela en su ultimo valor observado
     (2023). Solo el share (rate_lag1) evoluciona con la recursion.
  4. Los shares predichos se renormalizan para sumar 1 por anio
     (consistente con la definicion Opcion B del target).

Limitacion a declarar en el informe: el error se acumula en cada paso de
la recursion (2026 es menos confiable que 2024) y no existen metricas
para 2024-2026 hasta que el SERMIG publique los datos reales.

Uso:  python src/predict.py   (requiere dataset_region.csv y la grilla
                               ganadora en xgb_region_meta.json)
"""
import json

import pandas as pd

import config
from features import aggregate_region_year
from train import make_baseline, make_xgboost

FIRST_FORECAST_YEAR = config.TEST_YEAR + 1

# Features de composicion que se congelan en el ultimo anio observado
FROZEN_FEATURES = [f for f in config.FEATURES if f != "rate_lag1"]


def fit_final_models(df):
    """Reentrena baseline y XGBoost (hiperparametros ganadores de la
    grilla, leidos del meta) con todas las filas del panel."""
    X, y = df[config.FEATURES], df[config.TARGET]

    baseline = make_baseline()
    baseline.fit(X, y)

    meta_path = config.OUTPUTS_DIR / "xgb_region_meta.json"
    best_params = json.loads(meta_path.read_text(encoding="utf-8"))["hyperparams"]
    xgb = make_xgboost()
    xgb.set_params(**best_params)
    xgb.fit(X, y)
    return {"linear_regression": baseline, "xgboost": xgb}


def observed_composition(year):
    """Composicion demografica observada en `year`, recalculada desde el
    CSV crudo (el dataset de modelamiento solo trae rezagos, por la regla
    anti-leakage, pero para predecir year+1 el lag ES lo observado en year)."""
    raw = pd.read_csv(config.RAW_SERMIG)
    raw = raw[(raw["ANIO"] == year)
              & (raw["CODREGEO"] != config.UNKNOWN_REGION_CODE)]
    panel = aggregate_region_year(raw).set_index("CODREGEO")
    return panel.rename(columns={c: f"{c}_lag1" for c in panel.columns})


def recursive_forecast(df, models):
    """Encadena predicciones anio a anio desde TEST_YEAR+1 a FORECAST_END."""
    last = df[df["ANIO"] == config.TEST_YEAR].set_index("CODREGEO")
    regions = last[["REGION"]].copy()

    # Supuesto declarado: composicion congelada en lo observado el ultimo
    # anio (TEST_YEAR); es el lag correcto para TEST_YEAR+1 y la mejor
    # aproximacion disponible para los anios siguientes.
    frozen = observed_composition(config.TEST_YEAR)[FROZEN_FEATURES]

    rate_lag = last["rate"]  # share real del ultimo anio observado
    rows = []
    for year in range(FIRST_FORECAST_YEAR, config.FORECAST_END + 1):
        X = frozen.copy()
        X["rate_lag1"] = rate_lag
        X = X[config.FEATURES]

        preds = {}
        for name, model in models.items():
            raw = pd.Series(model.predict(X), index=X.index).clip(lower=0)
            preds[name] = raw / raw.sum()  # renormaliza: los shares suman 1

        for cod in X.index:
            rows.append({
                "CODREGEO": cod,
                "REGION": regions.loc[cod, "REGION"],
                "ANIO": year,
                "pred_linear": preds["linear_regression"][cod],
                "pred_xgboost": preds["xgboost"][cod],
            })

        # La recursion continua con el modelo optimo (menor RMSE de test)
        rate_lag = preds["linear_regression"]

    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(config.DATASET_REGION)
    models = fit_final_models(df)
    print(f"Modelos finales reentrenados con N={len(df)} filas "
          f"({sorted(df['ANIO'].unique())})")

    forecast = recursive_forecast(df, models)
    out_path = (config.OUTPUTS_DIR /
                f"predicciones_{FIRST_FORECAST_YEAR}_{config.FORECAST_END}.csv")
    forecast.to_csv(out_path, index=False)

    # Verificacion: shares por anio y modelo suman 1
    sums = forecast.groupby("ANIO")[["pred_linear", "pred_xgboost"]].sum()
    # Tolerancia 1e-6: XGBoost predice en float32
    assert ((sums - 1).abs() < 1e-6).all().all(), "Shares no suman 1"

    print(f"OK -> {out_path}\n")
    top = (forecast[forecast["CODREGEO"].isin([7, 3, 6, 2])]
           .pivot(index="REGION", columns="ANIO", values="pred_linear"))
    print("Share predicho (modelo optimo: regresion lineal), regiones principales:")
    print((top * 100).round(2).to_string())
    print("\nNota: sin metricas para estos anios hasta que exista el dato real;"
          "\nlas metricas de evaluacion (test 2023) estan en metricas.csv")


if __name__ == "__main__":
    main()
