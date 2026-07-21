"""Modelo 2: Random Forest sobre el panel v2 de 3 fuentes, con el MISMO
protocolo temporal del modelo 1 para comparar.

Lineal v2 y Gradient Boosting v2 se descartaron de esta carpeta por decision
propia (ya se cuenta con un Gradient Boosting de referencia fuera de v2, en
modelo_combinado/) -- nota: esto se aparta del anteproyecto aprobado, que
compromete comparar los 3 algoritmos (Lineal, RF, GB) para el objetivo
especifico 2; declarar el motivo si se pregunta.

No toca ningun artefacto del modelo 1 (../modelo_1_baseline/outputs/metricas.csv
se lee tal cual y se copia a la tabla comparativa).

Salidas en outputs/:
  metricas_v2.csv, comparacion_modelos.csv, predicciones_test_v2.csv,
  random_forest_v2.pkl
"""
import sys
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parents[1]
sys.path.insert(0, str(ROOT / "src" / "common"))
from config import SEED, TEST_YEAR  # noqa: E402

PANEL_V2 = AQUI / "outputs" / "dataset_region_v2.csv"
METRICAS_M1 = ROOT / "src" / "modelo_1_baseline" / "outputs" / "metricas.csv"
OUT_DIR = AQUI / "outputs"

TARGET = "rate"
# Set del modelo 1 (comparabilidad) + las 3 fuentes nuevas, todo rezagado a t-1.
# Las censo_* (foto 2024, posterior al test 2023) quedan fuera por leakage.
FEATURES_V2 = [
    "rate_lag1", "pct_women_lag1", "mean_age_lag1",
    "pct_irregular_lag1", "pct_venezuela_lag1",
    "sol_share_lag1", "sol_pct_otorga_lag1",
    "macro_desempleo_origen_lag1",
]

CV = KFold(n_splits=5, shuffle=True, random_state=SEED)  # k=5 segun documento

# Solo Random Forest v2: Lineal v2 y Gradient Boosting v2 se descartaron de
# esta carpeta por decision propia (ya se cuenta con un Gradient Boosting
# de referencia fuera de v2, en modelo_combinado/).
MODELOS = {
    "random_forest_v2": (
        RandomForestRegressor(random_state=SEED),
        {  # grilla conservadora: N_train = 32
            "n_estimators": [200, 500],
            "max_depth": [2, 3, 4],
            "min_samples_leaf": [2, 4],
            "max_features": [0.5, 1.0],
        },
    ),
}


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
    train = panel[panel["ANIO"] < TEST_YEAR]
    test = panel[panel["ANIO"] == TEST_YEAR]
    X_tr, y_tr = train[FEATURES_V2], train[TARGET]
    X_te, y_te = test[FEATURES_V2], test[TARGET]
    print(f"Train {sorted(train['ANIO'].unique())} N={len(train)} | "
          f"Test {TEST_YEAR} N={len(test)} | {len(FEATURES_V2)} features")

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
    df_m.to_csv(OUT_DIR / "metricas_v2.csv", index=False)
    preds.to_csv(OUT_DIR / "predicciones_test_v2.csv", index=False)
    (OUT_DIR / "hiperparametros_v2.json").write_text(
        json.dumps({"features": FEATURES_V2, "cv": "KFold(5)",
                    "seed": SEED, "mejores_params": mejores}, indent=2))

    # Tabla comparativa: modelo 1 (intacto) vs modelo 2
    m1 = (pd.read_csv(METRICAS_M1)
            .rename(columns={"model": "modelo", "RMSE": "rmse",
                             "MAE": "mae", "R2": "r2"})
            [["modelo", "split", "rmse", "mae", "r2"]])
    m1["version"] = "modelo_1 (5 features SERMIG)"
    df_m["version"] = "modelo_2 (3 fuentes, 8 features)"
    comp = pd.concat([m1, df_m], ignore_index=True)
    comp.to_csv(OUT_DIR / "comparacion_modelos.csv", index=False)

    print("\n== Comparacion (test 2023) ==")
    print(comp[comp["split"] == "test"]
          .sort_values("r2", ascending=False)
          .to_string(index=False))


if __name__ == "__main__":
    main()
