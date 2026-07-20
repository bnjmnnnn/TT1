"""Test de permutacion para el modelo nucleo v2 (train_v2.py) — la misma
prueba que ya se le aplico al modelo 1 original (5 features, ver
docs/decisiones_modelo_regional.md, seccion 4, hallazgo 2), nunca aplicada
al modelo v2 de 8 features pese a tener R2 igual de sospechoso
(Lineal 0.9999, GB 0.9995, N_train=32).

Logica: se reentrena el modelo cientos de veces con el target de
ENTRENAMIENTO barajado al azar (misma X, y sin relacion real con rate).
Si el modelo real tiene informacion genuina (no leakage), el R2 en test
con targets barajados debe rondar 0 o ser negativo — el modelo no puede
"adivinar" una relacion que no existe. Si el R2 permutado tambien sale alto,
hay fuga de informacion en el pipeline (por ejemplo, alguna feature que
codifica el target de forma indirecta).

Uso:  python src/validacion_v2.py
"""
from pathlib import Path

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "processed" / "dataset_region_v2.csv"
SEED = 42
N_PERMUTACIONES = 500

FEATURES = ["rate_lag1", "pct_women_lag1", "mean_age_lag1",
            "pct_irregular_lag1", "pct_venezuela_lag1",
            "sol_share_lag1", "sol_pct_otorga_lag1",
            "macro_desempleo_origen_lag1"]
TARGET = "rate"
TEST_YEAR = 2023


def cargar():
    panel = pd.read_csv(PANEL)
    train = panel[panel["ANIO"] < TEST_YEAR]
    test = panel[panel["ANIO"] == TEST_YEAR]
    return train[FEATURES], train[TARGET], test[FEATURES], test[TARGET]


def test_permutacion(nombre, construir_modelo, X_tr, y_tr, X_te, y_te, n_perm):
    modelo_real = construir_modelo()
    modelo_real.fit(X_tr, y_tr)
    r2_real = r2_score(y_te, modelo_real.predict(X_te))

    rng = np.random.RandomState(SEED)
    r2_permutados = []
    for _ in range(n_perm):
        y_barajado = y_tr.sample(frac=1, random_state=rng.randint(1_000_000)).values
        modelo = construir_modelo()
        modelo.fit(X_tr, y_barajado)
        r2_permutados.append(r2_score(y_te, modelo.predict(X_te)))
    r2_permutados = np.array(r2_permutados)

    p_valor = (r2_permutados >= r2_real).mean()
    print(f"\n{nombre}")
    print(f"  R2 real (test 2023):        {r2_real:.4f}")
    print(f"  R2 permutado: media={r2_permutados.mean():.4f}  "
          f"std={r2_permutados.std():.4f}  "
          f"max={r2_permutados.max():.4f}")
    print(f"  p-valor (P[R2_permutado >= R2_real]): {p_valor:.4f}")
    veredicto = ("SIN evidencia de leakage (el modelo real destaca claramente "
                "sobre el ruido)" if p_valor < 0.01 else
                "REVISAR: el R2 real no se distingue del ruido permutado")
    print(f"  Veredicto: {veredicto}")
    return {"modelo": nombre, "r2_real": r2_real,
           "r2_permutado_media": r2_permutados.mean(),
           "r2_permutado_std": r2_permutados.std(), "p_valor": p_valor}


def main():
    X_tr, y_tr, X_te, y_te = cargar()
    print(f"Train N={len(X_tr)} (2021-2022) | Test N={len(X_te)} (2023) | "
          f"{N_PERMUTACIONES} permutaciones por modelo")

    resultados = []
    resultados.append(test_permutacion(
        "regresion_lineal_v2",
        lambda: Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        X_tr, y_tr, X_te, y_te, N_PERMUTACIONES))

    # Mismos hiperparametros que gano el GridSearchCV en train_v2.py
    # (ver hiperparametros_v2.json): GradientBoostingRegressor de sklearn,
    # NO XGBoost — son implementaciones distintas, no intercambiables.
    # Se incluye como modelo de referencia/control (no es el modelo
    # seleccionado por el equipo).
    resultados.append(test_permutacion(
        "gradient_boosting_v2 (referencia, no seleccionado)",
        lambda: GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.03, max_depth=3,
            min_samples_leaf=2, subsample=1.0, random_state=SEED),
        X_tr, y_tr, X_te, y_te, N_PERMUTACIONES))

    # Random Forest v2: el modelo REALMENTE seleccionado por el equipo
    # (train_v2.py). Hiperparametros ganadores del GridSearchCV real.
    rf_params = json.loads((ROOT / "data" / "outputs" / "v2" /
                            "hiperparametros_v2.json").read_text(
        encoding="utf-8"))["mejores_params"]["random_forest_v2"]
    resultados.append(test_permutacion(
        "random_forest_v2 (modelo seleccionado)",
        lambda: RandomForestRegressor(random_state=SEED, **rf_params),
        X_tr, y_tr, X_te, y_te, N_PERMUTACIONES))

    out = pd.DataFrame(resultados)
    out.to_csv(ROOT / "data" / "outputs" / "v2" / "test_permutacion_v2.csv", index=False)
    print(f"\n-> {ROOT / 'data' / 'outputs' / 'v2' / 'test_permutacion_v2.csv'}")


if __name__ == "__main__":
    main()
