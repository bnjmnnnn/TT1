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
COMBINADO = ROOT / "data" / "raw" / "dataset_combinado.csv"
OUT_DIR = AQUI / "outputs"

TARGET = "estimation"
FEATURES = [
    "estimation_lag1", "pct_women_lag1", "mean_age_lag1",
    "pct_irregular_lag1", "pct_venezuela_lag1",
    "sol_share_lag1", "sol_pct_otorga_lag1",
    "macro_desempleo_origen_lag1",
]

# Validacion cruzada: parte las 32 filas de entrenamiento en 5 grupos y en cada
# ronda ajusta con 4 y valida con el que quedo fuera. Sirve para elegir los
# hiperparametros sin tocar el anio de prueba (2023).
# OJO: shuffle=True mezcla filas de 2021 y 2022 dentro de los folds; en una serie
# de tiempo lo ideal es separar por anio (ver
# docs/glosario_hiperparametros_y_metricas.md, seccion 5).
CV = KFold(n_splits=5, shuffle=True, random_state=SEED)

# ---------------------------------------------------------------------------
# Modo del target (ver docs/plan_mejoras.md seccion 2):
#   "nivel"       = se predice la cantidad de personas directamente. Es el
#                   comportamiento historico y reproduce las metricas que hoy
#                   cita el informe (metricas_conteo.csv).
#   "crecimiento" = se predice el crecimiento log(t / t-1) y se reconstruye la
#                   cantidad multiplicando por el nivel del anio anterior.
#                   Motivo: un arbol de decision no puede predecir un valor
#                   mayor que el maximo que vio en entrenamiento, y esta serie
#                   crece todos los anios -> estaba obligado a subestimar.
# Los artefactos de cada modo se guardan con nombres distintos para no pisar
# los vigentes.
MODO_TARGET = "crecimiento"
SUFIJO = "" if MODO_TARGET == "nivel" else "_crecimiento"
assert MODO_TARGET in ("nivel", "crecimiento"), "MODO_TARGET invalido"

MODELOS = {
    "regresion_lineal_conteo": (
        # StandardScaler: lleva cada feature a media 0 y desviacion 1 (hace
        # falta porque estimation_lag1 va de 3.220 a 976.302 y las proporciones
        # van de 0 a 1: sin escalar, la regresion se sesga hacia la mas grande).
        # LinearRegression: NO tiene hiperparametros que elegir, solo aprende
        # un coeficiente por feature.
        Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        {},
    ),
    "random_forest_conteo": (
        # Bosque: muchos arboles independientes que votan el promedio. Cada uno
        # ve una muestra distinta de los datos. Menos varianza que un arbol solo.
        RandomForestRegressor(random_state=SEED),
        {
            # cuantos arboles votan. Mas = mas estable y mas lento; en Random
            # Forest subirlo nunca empeora (a diferencia del boosting).
            "n_estimators": [200, 500],
            # preguntas encadenadas por arbol. 1 = un solo corte; 3 = tres
            # cortes seguidos. Mas profundidad = mas capacidad y mas riesgo de
            # memorizar (con 32 filas, profundidad 4 llega a 16 hojas de 2 filas).
            "max_depth": [2, 3, 4],
            # minimo de filas por hoja. Evita hojas con 1 sola fila, que son
            # memorizacion pura. Mas alto = arbol mas conservador.
            "min_samples_leaf": [2, 4],
            # fraccion de features que mira en cada corte. Con 1.0 todos los
            # arboles miran las 8 y se parecen entre si; con 0.5 (4 de 8) se
            # diversifican y el promedio de sus votos es mejor.
            "max_features": [0.5, 1.0],
        },
    ),
    "gradient_boosting_conteo": (
        # Boosting: arboles encadenados, cada uno se ajusta al error que dejo el
        # anterior (no votan: suman). Suele acertar mas que el bosque y
        # sobreajustar antes.
        GradientBoostingRegressor(random_state=SEED),
        {
            # cuantos arboles en la cadena. Aqui SI puede sobreajustar si son
            # demasiados (cada uno persigue el error que quedo).
            "n_estimators": [200, 500],
            # cuanto pesa la correccion de cada arbol (0.05 = 5%). Es un freno:
            # mas bajo = mas prudente y necesita mas arboles para lo mismo; mas
            # alto = avanza rapido pero puede pasarse de largo.
            "learning_rate": [0.01, 0.03, 0.05],
            # profundidad de cada arbol corrector: 2-3 basta, porque cada uno
            # corrige un pedacito del error, no el problema completo.
            "max_depth": [2, 3],
            # minimo de filas por hoja (igual que en el bosque).
            "min_samples_leaf": [2, 4],
            # fraccion de filas que usa cada arbol. Con 0.8 cada arbol ve un 80%
            # al azar: agrega variedad y reduce el sobreajuste.
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


def a_crecimiento(y_nivel, lag):
    """Nivel (personas) -> crecimiento logaritmico respecto del anio anterior.

    lag = cantidad de personas del anio t-1 (columna estimation_lag1).
    """
    return np.log(np.asarray(y_nivel, dtype=float) / np.asarray(lag, dtype=float))


def desde_crecimiento(g_hat, lag):
    """Crecimiento logaritmico -> nivel (personas) del anio que se predice."""
    return np.asarray(lag, dtype=float) * np.exp(np.asarray(g_hat, dtype=float))


def evaluar(nombre, modelo, X_tr, y_tr, lag_tr, X_te, y_te, lag_te):
    """Metricas en la escala original (personas).

    Si el modelo se entreno sobre el crecimiento, se reconstruye el nivel antes
    de medir, para que los numeros sean comparables con los del modo "nivel".
    """
    filas = []
    for split, X, y, lag in (("train", X_tr, y_tr, lag_tr),
                             ("test", X_te, y_te, lag_te)):
        pred = modelo.predict(X)
        if MODO_TARGET == "crecimiento":
            pred = desde_crecimiento(pred, lag)
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
    # nivel del anio anterior: divisor al construir el crecimiento y
    # multiplicador al reconstruir la cantidad
    lag_tr, lag_te = train["estimation_lag1"], test["estimation_lag1"]
    # lo que se le entrega al algoritmo para ajustar (nivel o crecimiento)
    y_fit = a_crecimiento(y_tr, lag_tr) if MODO_TARGET == "crecimiento" else y_tr
    print(f"Train {sorted(train['ANIO'].unique())} N={len(train)} | "
          f"Test {TEST_YEAR} N={len(test)} | target=estimation (cantidad) | "
          f"modo={MODO_TARGET} | {len(FEATURES)} features")

    metricas, mejores = [], {}
    preds = test[["CODREGEO", "REGION", "ANIO", TARGET]].copy()
    preds["estimation_lag1"] = lag_te.to_numpy()
    for nombre, (est, grid) in MODELOS.items():
        if grid:
            gs = GridSearchCV(est, grid, cv=CV, n_jobs=-1,
                              scoring="neg_root_mean_squared_error")
            # el CV se hace sobre y_fit (en modo crecimiento, sobre el
            # crecimiento: equivale a optimizar el error relativo)
            gs.fit(X_tr, y_fit)
            modelo, params = gs.best_estimator_, gs.best_params_
        else:
            modelo, params = est.fit(X_tr, y_fit), {}
        mejores[nombre] = params
        metricas += evaluar(nombre, modelo, X_tr, y_tr, lag_tr, X_te, y_te, lag_te)
        pred_te = modelo.predict(X_te)
        if MODO_TARGET == "crecimiento":
            # se vuelve a la escala de personas
            pred_te = desde_crecimiento(pred_te, lag_te)
        preds[f"pred_{nombre}"] = pred_te
        joblib.dump(modelo, OUT_DIR / f"{nombre}{SUFIJO}.pkl")
        print(f"{nombre}: {params}")

    df_m = pd.DataFrame(metricas)
    df_m.to_csv(OUT_DIR / f"metricas_conteo{SUFIJO}.csv", index=False)
    preds.to_csv(OUT_DIR / f"predicciones_test_conteo{SUFIJO}.csv", index=False)
    (OUT_DIR / f"hiperparametros_conteo{SUFIJO}.json").write_text(
        json.dumps({"target": TARGET, "modo_target": MODO_TARGET,
                    "features": FEATURES, "cv": "KFold(5)",
                    "seed": SEED, "mejores_params": mejores}, indent=2))

    print(f"\n== Resultados (test {TEST_YEAR}, target = cantidad absoluta, "
          f"modo = {MODO_TARGET}) ==")
    print(df_m[df_m["split"] == "test"].sort_values("r2", ascending=False)
              .to_string(index=False))
    print(f"\n-> {OUT_DIR / f'metricas_conteo{SUFIJO}.csv'}")
    print(f"-> {OUT_DIR / f'predicciones_test_conteo{SUFIJO}.csv'}")


if __name__ == "__main__":
    main()
