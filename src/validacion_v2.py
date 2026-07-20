"""Test de permutacion para el modelo nucleo v2 (train_v2.py) — la misma
prueba que ya se le aplico al modelo 1 original (5 features, ver
docs/decisiones_modelo_regional.md, seccion 4, hallazgo 2).

Valida random_forest_v2 (el UNICO modelo que train_v2.py entrena y exporta
desde que Lineal v2 y Gradient Boosting v2 se descartaron de esta carpeta,
ver docs/decisiones_v2.md). Version anterior de este script validaba
Lineal v2 y Gradient Boosting v2 -- modelos que train_v2.py ya no entrena --
y nunca corria la prueba sobre el modelo que realmente se usa; era codigo
que quedo desalineado tras el cambio de alcance.

Logica: se reentrena el modelo cientos de veces con el target de
ENTRENAMIENTO barajado al azar (misma X, y sin relacion real con rate).
Si el modelo real tiene informacion genuina (no leakage), el R2 en test
con targets barajados debe rondar 0 o ser negativo — el modelo no puede
"adivinar" una relacion que no existe. Si el R2 permutado tambien sale alto,
hay fuga de informacion en el pipeline (por ejemplo, alguna feature que
codifica el target de forma indirecta).

Uso:  python src/validacion_v2.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from config import SEED, TEST_YEAR

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "processed" / "dataset_region_v2.csv"
HIPERPARAMS = ROOT / "data" / "outputs" / "v2" / "hiperparametros_v2.json"
N_PERMUTACIONES = 500


def cargar():
    hp = json.loads(HIPERPARAMS.read_text())
    features = hp["features"]
    params = hp["mejores_params"]["random_forest_v2"]
    panel = pd.read_csv(PANEL)
    train = panel[panel["ANIO"] < TEST_YEAR]
    test = panel[panel["ANIO"] == TEST_YEAR]
    return (train[features], train["rate"], test[features], test["rate"], params)


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
    X_tr, y_tr, X_te, y_te, params = cargar()
    print(f"Train N={len(X_tr)} (2021-2022) | Test N={len(X_te)} (2023) | "
          f"{N_PERMUTACIONES} permutaciones")

    # Mismos hiperparametros que gano el GridSearchCV en train_v2.py
    # (ver hiperparametros_v2.json).
    resultados = [test_permutacion(
        "random_forest_v2",
        lambda: RandomForestRegressor(random_state=SEED, **params),
        X_tr, y_tr, X_te, y_te, N_PERMUTACIONES)]

    out = pd.DataFrame(resultados)
    out.to_csv(ROOT / "data" / "outputs" / "v2" / "test_permutacion_v2.csv", index=False)
    print(f"\n-> {ROOT / 'data' / 'outputs' / 'v2' / 'test_permutacion_v2.csv'}")


if __name__ == "__main__":
    main()
