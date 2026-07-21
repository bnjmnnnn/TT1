"""Modelo 3: predice la CANTIDAD de inmigrantes (estimacion absoluta) del
anio siguiente por region, no el share/tasa (modelos 1 y 2).

Cambia el target de `rate` (participacion sobre el stock nacional) a
`estimation` (numero absoluto de personas). Reusa las mismas 7 features de
composicion/administrativas rezagadas del panel v2, cambiando `rate_lag1`
por `estimation_lag1` como predictor autorregresivo (equivalente conceptual,
pero en la escala de cantidad, no de proporcion).
"""
import sys
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parents[1]
sys.path.insert(0, str(ROOT / "src" / "common"))
from config import SEED, TEST_YEAR  # noqa: E402

PANEL_V2 = ROOT / "src" / "modelo_2_regional_v2" / "outputs" / "dataset_region_v2.csv"
COMBINADO = ROOT / "dataset_combinado.csv"
OUT_DIR = AQUI / "outputs"

TARGET = "estimation"
FEATURES = [
    "estimation_lag1", "pct_women_lag1", "mean_age_lag1",
    "pct_irregular_lag1", "pct_venezuela_lag1",
    "sol_share_lag1", "sol_pct_otorga_lag1",
    "macro_desempleo_origen_lag1",
]

CV = KFold(n_splits=5, shuffle=True, random_state=SEED)

MODELOS = {
    "regresion_lineal_conteo": (
        Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        {},
    ),
    "random_forest_conteo": (
        RandomForestRegressor(random_state=SEED),
        {
            "n_estimators": [200, 500],
            "max_depth": [2, 3, 4],
            "min_samples_leaf": [2, 4],
            "max_features": [0.5, 1.0],
        },
    ),
    "gradient_boosting_conteo": (
        GradientBoostingRegressor(random_state=SEED),
        {
            "n_estimators": [200, 500],
            "learning_rate": [0.01, 0.03, 0.05],
            "max_depth": [2, 3],
            "min_samples_leaf": [2, 4],
            "subsample": [0.8, 1.0],
        },
    ),
}


def estimation_lag1() -> pd.DataFrame:
    """ESTIMACION agregada por region-anio, rezagada a t-1 (2018-2023 alcanza
    para cubrir el rezago del panel 2021-2023 sin perder filas)."""
    dc = pd.read_csv(COMBINADO)
    dc = dc[dc["CODREGEO"] != 17]
    agg = dc.groupby(["CODREGEO", "AÑO"])["ESTIMACION"].sum().reset_index()
    agg["AÑO"] = agg["AÑO"] + 1
    return agg.rename(columns={"ESTIMACION": "estimation_lag1", "AÑO": "ANIO"})


def evaluar(nombre, modelo, X_tr, y_tr, X_te, y_te):
    filas = []
    for split, X, y in (("train", X_tr, y_tr), ("test", X_te, y_te)):
        pred = modelo.predict(X)
        filas.append({
            "modelo": nombre, "split": split,
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "mae": float(mean_absolute_error(y, pred)),
            "r2": float(r2_score(y, pred)),
        })
    return filas


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(PANEL_V2)
    n0 = len(panel)
    panel = panel.merge(estimation_lag1(), on=["CODREGEO", "ANIO"], how="left")
    assert len(panel) == n0, "El merge altero el numero de filas"
    assert panel["estimation_lag1"].notna().all(), "estimation_lag1 con nulos"

    train = panel[panel["ANIO"] < TEST_YEAR]
    test = panel[panel["ANIO"] == TEST_YEAR]
    X_tr, y_tr = train[FEATURES], train[TARGET]
    X_te, y_te = test[FEATURES], test[TARGET]
    print(f"Train {sorted(train['ANIO'].unique())} N={len(train)} | "
          f"Test {TEST_YEAR} N={len(test)} | target=estimation (cantidad) | "
          f"{len(FEATURES)} features")

    metricas, mejores = [], {}
    preds = test[["CODREGEO", "REGION", "ANIO", TARGET]].copy()
    for nombre, (est, grid) in MODELOS.items():
        if grid:
            gs = GridSearchCV(est, grid, cv=CV, n_jobs=-1,
                              scoring="neg_root_mean_squared_error")
            gs.fit(X_tr, y_tr)
            modelo, params = gs.best_estimator_, gs.best_params_
        else:
            modelo, params = est.fit(X_tr, y_tr), {}
        mejores[nombre] = params
        metricas += evaluar(nombre, modelo, X_tr, y_tr, X_te, y_te)
        preds[f"pred_{nombre}"] = modelo.predict(X_te)
        joblib.dump(modelo, OUT_DIR / f"{nombre}.pkl")
        print(f"{nombre}: {params}")

    df_m = pd.DataFrame(metricas)
    df_m.to_csv(OUT_DIR / "metricas_conteo.csv", index=False)
    preds.to_csv(OUT_DIR / "predicciones_test_conteo.csv", index=False)
    (OUT_DIR / "hiperparametros_conteo.json").write_text(
        json.dumps({"target": TARGET, "features": FEATURES, "cv": "KFold(5)",
                    "seed": SEED, "mejores_params": mejores}, indent=2))

    print("\n== Resultados (test 2023, target = cantidad absoluta) ==")
    print(df_m[df_m["split"] == "test"].sort_values("r2", ascending=False)
              .to_string(index=False))
    print(f"\n-> {OUT_DIR / 'metricas_conteo.csv'}")
    print(f"-> {OUT_DIR / 'predicciones_test_conteo.csv'}")


if __name__ == "__main__":
    main()
