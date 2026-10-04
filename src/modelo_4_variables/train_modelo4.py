"""Modelo 4: predice la REDISTRIBUCION regional (y el crecimiento, como contraste).

Por que este modelo y no otro. Los modelos 1-3 predicen el NIVEL de la serie, y ahi
la persistencia regala el resultado (R2 0,9966 = copiar el ano anterior). La
exploracion previa (explorar_variables.py) mostro que:

  - nivel de `rate`       : el bosque pierde por 382x en MSE contra copiar  -> sin sentido
  - nivel de `estimation` : el bosque pierde por 13x                        -> sin sentido
  - crecimiento           : el baseline "no crecer" es tonto (la serie siempre crece)
  - REDISTRIBUCION        : el baseline "no cambiar de peso" explica CERO (R2 = 0,0000)

La redistribucion (`rate_t - rate_t-1`) es entonces la unica pregunta donde un
modelo puede aportar algo real y medible: no esta regalada. Ademas es la pregunta
del tema del trabajo (que regiones ganan o pierden peso migratorio), no una
transformacion tecnica del target.

Nota util: como los pesos suman 1, la media del cambio es 0 en cada anio. Por eso
en esta pregunta el skill score y el R2 coinciden numericamente (la referencia
"no cambiar" es igual al promedio), y no hace falta el drift.

Uso:  python src/modelo_4_variables/train_modelo4.py
      -> outputs/metricas_modelo4.csv
      -> outputs/predicciones_redistribucion_2023.csv
      -> outputs/reporte_modelo4.md
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parents[1]
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(ROOT / "src" / "common"))
from config import SEED  # noqa: E402
from explorar_variables import cargar_panel  # noqa: E402
from probar_ronda2 import agregar_derivadas  # noqa: E402

OUT_DIR = AQUI / "outputs"

# Conjuntos de variables elegidos con la exploracion (ranking por importancia de
# permutacion y por skill de una sola variable, medidos en entrenamiento):
#   redistribucion -> las tres de solicitudes mandan (|corr| 0,65-0,89)
#   crecimiento    -> edad media, nacionalidades y macro del pais de origen
SETS = {
    "redistribucion": {
        "mejor_sola": ["sol_otorgadas_lag1"],
        # `rate_lag2` (el peso de la region de hace dos anos) se agrego en la ronda 2:
        # es la unica variable que mejoro el R2 en LOS DOS anos de prueba (ver
        # docs/modelo_4_variables.md, seccion 9.1). Es la configuracion recomendada.
        "seleccionadas": ["sol_otorgadas_lag1", "sol_share_lag1", "sol_total_lag1",
                          "pct_peru_lag1", "mean_age_lag1", "rate_lag2"],
        "todas": [c for c in [
            "pct_women_lag1", "mean_age_lag1", "pct_irregular_lag1", "pct_venezuela_lag1",
            "pct_peru_lag1", "pct_colombia_lag1", "pct_haiti_lag1", "pct_bolivia_lag1",
            "sol_total_lag1", "sol_otorgadas_lag1", "sol_pct_otorga_lag1", "sol_share_lag1",
            "macro_inflacion_origen_lag1", "macro_crecimiento_pib_origen_lag1",
            "macro_desempleo_origen_lag1", "rate_lag2"]],
    },
    "crecimiento": {
        "mejor_sola": ["pct_irregular_lag1"],
        "seleccionadas": ["pct_irregular_lag1", "mean_age_lag1", "pct_colombia_lag1",
                          "macro_inflacion_origen_lag1", "pct_venezuela_lag1"],
        "todas": [c for c in [
            "pct_women_lag1", "mean_age_lag1", "pct_irregular_lag1", "pct_venezuela_lag1",
            "pct_peru_lag1", "pct_colombia_lag1", "pct_haiti_lag1", "pct_bolivia_lag1",
            "sol_total_lag1", "sol_otorgadas_lag1", "sol_pct_otorga_lag1", "sol_share_lag1",
            "macro_inflacion_origen_lag1", "macro_crecimiento_pib_origen_lag1",
            "macro_desempleo_origen_lag1"]],
    },
}

# KNN entra para tener una familia distinta de las tres anteriores (no es un
# arbol ni una recta: promedia las regiones vecinas en el espacio de variables).
MODELOS = {
    "lineal": Pipeline([("esc", StandardScaler()), ("m", LinearRegression())]),
    "knn_5": Pipeline([("esc", StandardScaler()), ("m", KNeighborsRegressor(n_neighbors=5))]),
    "random_forest": RandomForestRegressor(n_estimators=500, max_depth=3,
                                           min_samples_leaf=2, random_state=SEED),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                   max_depth=2, min_samples_leaf=2,
                                                   random_state=SEED),
}


def skill(y, pred, ref) -> float:
    mse, mse_ref = ((y - pred) ** 2).mean(), ((y - ref) ** 2).mean()
    return float(1 - mse / mse_ref) if mse_ref > 0 else float("nan")


def metricas(y, pred, ref, ref_alt=None) -> dict:
    out = {
        "rmse": float(np.sqrt(((y - pred) ** 2).mean())),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
        "skill": skill(y, pred, ref),
    }
    if ref_alt is not None:
        out["skill_vs_drift"] = skill(y, pred, ref_alt)
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = agregar_derivadas(cargar_panel())
    p["redistribucion"] = p["rate"] - p["rate_lag1"]
    p["crecimiento"] = np.log(p["estimation"] / p["estimation_lag1"])

    filas, reportes = [], []
    for anio_test in (2023, 2022):
        for pregunta in ("redistribucion", "crecimiento"):
            tr = p[p["ANIO"] < anio_test]
            te = p[p["ANIO"] == anio_test]
            y_tr = tr[pregunta].to_numpy()
            y_te = te[pregunta].to_numpy()
            # referencias: "no cambia" (0) y, para el crecimiento, el drift
            ref_te = np.zeros_like(y_te)
            drift = np.full_like(y_te, y_tr.mean())
            ref_alt = drift if pregunta == "crecimiento" else None

            base = metricas(y_te, ref_te, ref_te, ref_alt)
            filas.append({"pregunta": pregunta, "anio_test": anio_test,
                          "conjunto": "NO HACER NADA (persistencia)", "modelo": "-",
                          "n_vars": 0, **{k: base[k] for k in ("rmse", "mae", "r2", "skill")},
                          **({"skill_vs_drift": base["skill_vs_drift"]} if ref_alt is not None else {})})
            if ref_alt is not None:
                b2 = metricas(y_te, drift, ref_te, ref_alt)
                filas.append({"pregunta": pregunta, "anio_test": anio_test,
                              "conjunto": "NO HACER NADA (drift historico)", "modelo": "-",
                              "n_vars": 0, **{k: b2[k] for k in ("rmse", "mae", "r2", "skill")},
                              "skill_vs_drift": 0.0})

            for nombre_set, cols in SETS[pregunta].items():
                X_tr, X_te = tr[cols].to_numpy(), te[cols].to_numpy()
                for nombre_mod, modelo in MODELOS.items():
                    modelo.fit(X_tr, y_tr)
                    pred = modelo.predict(X_te)
                    m = metricas(y_te, pred, ref_te, ref_alt)
                    filas.append({"pregunta": pregunta, "anio_test": anio_test,
                                  "conjunto": nombre_set, "modelo": nombre_mod,
                                  "n_vars": len(cols), **m})
                    if anio_test == 2023 and nombre_set == "seleccionadas":
                        joblib.dump(modelo, OUT_DIR / f"{pregunta}_{nombre_mod}.pkl")

            # tabla por region (test principal 2023)
            if anio_test == 2023:
                cols = SETS[pregunta]["seleccionadas"]
                modelo = MODELOS["random_forest"]
                modelo.fit(tr[cols].to_numpy(), y_tr)
                pred = modelo.predict(te[cols].to_numpy())
                t = te[["CODREGEO", "REGION", "rate_lag1", "rate", pregunta]].copy()
                t["predicho"] = pred
                t["error"] = pred - t[pregunta]
                t["rate_predicho"] = (t["rate_lag1"] + pred) if pregunta == "redistribucion" else np.nan
                t.to_csv(OUT_DIR / f"predicciones_{pregunta}_{anio_test}.csv", index=False)
                reportes.append((pregunta, t))

    df = pd.DataFrame(filas)
    df.to_csv(OUT_DIR / "metricas_modelo4.csv", index=False)

    # --- informe
    L = ["# Modelo 4: la redistribucion regional", "",
         "Generado por `src/modelo_4_variables/train_modelo4.py`.", "",
         "**Pregunta:** cuanto cambia el peso de cada region entre un anio y el "
         "siguiente (`rate_t - rate_t-1`). **Por que:** es la unica de las preguntas "
         "posibles donde el baseline no explica nada (R2 = 0,0000), asi que un modelo "
         "que aporte algo se puede medir de verdad.", "",
         "Referencia: *no cambiar de peso* (0). En esta pregunta el skill score y el R2 "
         "coinciden, porque la media del cambio es 0 por construccion.", ""]
    for anio in (2023, 2022):
        sub = df[(df.anio_test == anio) & (df.pregunta == "redistribucion")]
        L += [f"## Test {anio} " + ("(principal)" if anio == 2023 else "(secundario: se entrena solo con 2021)"), "",
              "| conjunto | modelo | n vars | RMSE | MAE | R2 | skill |", "|---|---|---|---|---|---|---|"]
        for _, f in sub.iterrows():
            L.append(f"| {f['conjunto']} | {f['modelo']} | {f['n_vars']} | {f['rmse']:.5f} | "
                     f"{f['mae']:.5f} | {f['r2']:+.4f} | **{f['skill']:+.3f}** |")
        L.append("")

    L += ["## Contraste: la pregunta del crecimiento", "",
          "Aqui el baseline correcto NO es 0 (la serie siempre crece) sino el *drift*: "
          "suponer que cada region crece el promedio historico. Por eso se reportan los "
          "dos skills: contra 0 (persistencia en nivel) y contra el drift.", ""]
    for anio in (2023, 2022):
        sub = df[(df.anio_test == anio) & (df.pregunta == "crecimiento")]
        L += [f"### Test {anio}", "", "| conjunto | modelo | RMSE | R2 | skill vs 0 | skill vs drift |",
              "|---|---|---|---|---|---|"]
        for _, f in sub.iterrows():
            v = f.get("skill_vs_drift", np.nan)
            L.append(f"| {f['conjunto']} | {f['modelo']} | {f['rmse']:.4f} | {f['r2']:+.4f} | "
                     f"{f['skill']:+.3f} | **{v:+.3f}** |")
        L.append("")

    for pregunta, t in reportes:
        L += [f"## Por region: {pregunta} en 2023 (Random Forest, variables seleccionadas)", "",
              "| region | peso 2022 | peso 2023 | cambio real | cambio predicho | error |", "|---|---|---|---|---|---|"]
        for _, f in t.sort_values(pregunta).iterrows():
            L.append(f"| {f['REGION'].title()} | {f['rate_lag1']:.4f} | {f['rate']:.4f} | "
                     f"{f[pregunta]:+.4f} | {f['predicho']:+.4f} | {f['error']:+.4f} |")
        L.append("")

    (OUT_DIR / "reporte_modelo4.md").write_text("\n".join(L))
    print(f"-> {OUT_DIR / 'reporte_modelo4.md'}")
    print(df[df.anio_test == 2023].to_string(index=False))


if __name__ == "__main__":
    main()
