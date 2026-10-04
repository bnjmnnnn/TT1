"""Comparacion completa de algoritmos para la tasa de concentracion regional
(rate), sobre el panel v2 de 3 fuentes (8 features): los 3 algoritmos
comprometidos en el anteproyecto (Regresion Lineal, Random Forest, Gradient
Boosting) + XGBoost como referencia adicional, mas una fila de persistencia
(prediccion = rate_lag1, sin entrenar ningun modelo) como piso de
comparacion.

Por que existe aparte de train_v2.py: ese script entrena y exporta el
modelo final del proyecto (Random Forest, el unico que se usa en
produccion). Este script es solo para generar la Tabla 5.1 del informe --
compara los 4 algoritmos en igualdad de condiciones (mismas 8 features,
mismo protocolo) y no exporta ningun modelo nuevo para usar despues.

Motivacion del baseline de persistencia: con un target tan persistente en
el tiempo como rate, un R2 alto no es evidencia por si solo de que el
modelo aprendio el fenomeno migratorio -- puede ser solo el reflejo de que
el fenomeno casi no cambia de un anio a otro. Comparar contra "copiar el
valor del anio anterior, sin modelo" es el primer filtro antes de creerle
el R2 a cualquier algoritmo.

Salidas en outputs/:
  comparacion_completa_v2.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parents[1]
sys.path.insert(0, str(ROOT / "src" / "common"))
from config import SEED, TEST_YEAR  # noqa: E402

PANEL_V2 = AQUI / "outputs" / "dataset_region_v2.csv"
OUT_DIR = AQUI / "outputs"

TARGET = "rate"
FEATURES_V2 = [
    "rate_lag1", "pct_women_lag1", "mean_age_lag1",
    "pct_irregular_lag1", "pct_venezuela_lag1",
    "sol_share_lag1", "sol_pct_otorga_lag1",
    "macro_desempleo_origen_lag1",
]

CV = KFold(n_splits=5, shuffle=True, random_state=SEED)  # mismo k que train_v2.py

# Grillas conservadoras dado N_train=32 (misma logica que el resto del proyecto)
MODELOS = {
    "regresion_lineal_v2": (
        make_pipeline(StandardScaler(), LinearRegression()),
        None,  # sin hiperparametros que ajustar
    ),
    "random_forest_v2": (
        RandomForestRegressor(random_state=SEED),
        {
            "n_estimators": [200, 500],
            "max_depth": [2, 3, 4],
            "min_samples_leaf": [2, 4],
            "max_features": [0.5, 1.0],
        },
    ),
    "gradient_boosting_v2": (
        GradientBoostingRegressor(random_state=SEED),
        {
            "n_estimators": [200, 300],
            "max_depth": [2, 3],
            "learning_rate": [0.03, 0.05, 0.1],
            "min_samples_leaf": [2, 4],
        },
    ),
    "xgboost_v2": (
        XGBRegressor(random_state=SEED, n_jobs=1),
        {
            "n_estimators": [200, 300],
            "max_depth": [2, 3],
            "learning_rate": [0.03, 0.05, 0.1],
            "min_child_weight": [1, 3],
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


def evaluar_persistencia(X_tr, y_tr, X_te, y_te):
    """Prediccion = rate_lag1 directo, sin entrenar ningun modelo."""
    filas = []
    for split, X, y in (("train", X_tr, y_tr), ("test", X_te, y_te)):
        pred = X["rate_lag1"].to_numpy()
        filas.append({
            "modelo": "persistencia", "split": split,
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

    metricas = evaluar_persistencia(X_tr, y_tr, X_te, y_te)
    for nombre, (est, grid) in MODELOS.items():
        if grid:
            gs = GridSearchCV(est, grid, cv=CV, n_jobs=-1,
                              scoring="neg_root_mean_squared_error")
            gs.fit(X_tr, y_tr)
            modelo, params = gs.best_estimator_, gs.best_params_
        else:
            modelo, params = est.fit(X_tr, y_tr), {}
        metricas += evaluar(nombre, modelo, X_tr, y_tr, X_te, y_te)
        print(f"{nombre}: {params}")

    df_m = pd.DataFrame(metricas)
    df_m.to_csv(OUT_DIR / "comparacion_completa_v2.csv", index=False)

    print("\n== Comparacion completa, 8 features (test 2023) ==")
    print(df_m[df_m["split"] == "test"]
          .sort_values("r2", ascending=False)
          .to_string(index=False))


if __name__ == "__main__":
    main()