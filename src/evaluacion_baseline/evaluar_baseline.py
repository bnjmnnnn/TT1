"""Evaluacion con BASELINE DE PERSISTENCIA y revision del modelo de conteo.

Problema que resuelve
---------------------
Los modelos del proyecto se comparan entre si (R2, RMSE, MAE), pero nunca
contra la regla mas simple posible: "el proximo anio la region vale lo mismo
que este anio". Si esa regla gana o empata, el R2 alto no demuestra aporte del
modelo, porque el target es muy persistente por construccion.

Que calcula
-----------
1. Baseline de persistencia para los dos targets (rate y estimation).
2. Skill score (1 - MSE_modelo / MSE_persistencia) de cada candidato.
3. Error especifico de la RM, Antofagasta y Valparaiso, donde el diagnostico
   propio del proyecto (docs/propuesta_solucion_rm.md) ya detecto el problema.
4. Candidatos nuevos: target logaritmico (comprime la escala 4.555 -> 1.089.049)
   y ExtraTrees (sin bootstrap, deberia aislar mejor a la RM).
5. Renormalizacion post-hoc (§3.1 del documento de solucion RM), en dos
   variantes: con el total nacional real (cota superior optimista) y con el
   total nacional proyectado solo con informacion pasada (la variante honesta).
6. Estabilidad: se repite todo con test 2022 (entrenando con 2021) para ver si
   el ranking depende del anio elegido como hold-out.

NO modifica ninguna carpeta de modelos existente: solo lee sus outputs
(paneles y .pkl) y escribe en src/evaluacion_baseline/outputs/.

Uso:  python src/evaluacion_baseline/evaluar_baseline.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
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

FEATURES = [
    "estimation_lag1", "pct_women_lag1", "mean_age_lag1",
    "pct_irregular_lag1", "pct_venezuela_lag1",
    "sol_share_lag1", "sol_pct_otorga_lag1",
    "macro_desempleo_origen_lag1",
]
FEATURES_RATE = ["rate_lag1"] + FEATURES[1:]

# regiones criticas del diagnostico propio del proyecto (propuesta_solucion_rm.md §1)
REGIONES_CRITICAS = {7: "METROPOLITANA", 3: "ANTOFAGASTA", 6: "VALPARAISO"}

CV = KFold(n_splits=5, shuffle=True, random_state=SEED)

# hiperparametros ganadores ya documentados por el proyecto (misma configuracion,
# para que la comparacion sea justa contra lo que hay hoy en el repo)
RF_PARAMS = dict(n_estimators=200, max_depth=4, min_samples_leaf=2,
                 max_features=1.0, random_state=SEED)
GB_PARAMS = dict(n_estimators=200, learning_rate=0.05, max_depth=3,
                 min_samples_leaf=2, random_state=SEED)


def candidatos() -> dict:
    """Modelos a comparar, en escala cruda y en escala logaritmica."""
    return {
        "lineal": (Pipeline([("esc", StandardScaler()), ("m", LinearRegression())]), {}),
        "random_forest": (RandomForestRegressor(**RF_PARAMS), {}),
        "gradient_boosting": (GradientBoostingRegressor(**GB_PARAMS), {}),
        "extra_trees": (ExtraTreesRegressor(random_state=SEED),
                        {"n_estimators": [200], "max_depth": [3, 4, None],
                         "min_samples_leaf": [1, 2], "max_features": [0.5, 1.0]}),
        "lineal_log": (Pipeline([("esc", StandardScaler()), ("m", LinearRegression())]), {}),
        "random_forest_log": (RandomForestRegressor(**RF_PARAMS), {}),
        "gradient_boosting_log": (GradientBoostingRegressor(**GB_PARAMS), {}),
        "extra_trees_log": (ExtraTreesRegressor(random_state=SEED),
                            {"n_estimators": [200], "max_depth": [3, 4, None],
                             "min_samples_leaf": [1, 2], "max_features": [0.5, 1.0]}),
    }


# --------------------------------------------------------------------------
def estimation_lag1() -> pd.DataFrame:
    """Estimacion SERMIG agregada por region-anio, rezagada a t-1 (misma
    definicion que train_conteo.py; se recalcula para no depender de su salida)."""
    dc = pd.read_csv(COMBINADO)
    dc = dc[dc["CODREGEO"] != 17]
    agg = dc.groupby(["CODREGEO", "AÑO"])["ESTIMACION"].sum().reset_index()
    agg["AÑO"] = agg["AÑO"] + 1
    return agg.rename(columns={"ESTIMACION": "estimation_lag1", "AÑO": "ANIO"})


def total_nacional() -> pd.Series:
    dc = pd.read_csv(COMBINADO)
    return dc[dc["CODREGEO"] != 17].groupby("AÑO")["ESTIMACION"].sum()


def medir(y: np.ndarray, pred: np.ndarray, base: np.ndarray) -> dict:
    mse, mse_base = mean_squared_error(y, pred), mean_squared_error(y, base)
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y, pred)),
        "mape": float(np.mean(np.abs((y - pred) / y)) * 100),
        "r2": float(r2_score(y, pred)),
        "r2_persistencia": float(r2_score(y, base)),
        "skill_vs_persistencia": float(1 - mse / mse_base),
    }


def error_region(test: pd.DataFrame, col_pred: str, objetivo: str) -> str:
    out = {}
    for cod, nombre in REGIONES_CRITICAS.items():
        fila = test[test["CODREGEO"] == cod]
        if fila.empty:
            continue
        real, pred = float(fila[objetivo].iloc[0]), float(fila[col_pred].iloc[0])
        out[nombre] = round(100 * (pred / real - 1), 1)
    return json.dumps(out, ensure_ascii=False)


def evaluar_target(panel: pd.DataFrame, target: str, features: list, test_year: int,
                   out_csv: Path) -> tuple:
    """Entrena todos los candidatos y los mide contra la persistencia."""
    tr = panel[panel["ANIO"] < test_year]
    te = panel[panel["ANIO"] == test_year].reset_index(drop=True)
    base_col = "rate_lag1" if target == "rate" else "estimation_lag1"
    X_tr, y_tr = tr[features], tr[target].to_numpy()
    y_te = te[target].to_numpy()
    base = te[base_col].to_numpy()
    te["persistencia"] = base

    filas = [{"modelo": "persistencia (anio anterior tal cual)", **medir(y_te, base, base),
              "error_regiones_pct": error_region(te, "persistencia", target)}]

    for nombre, (est, grid) in candidatos().items():
        y_ajustado = np.log(y_tr) if nombre.endswith("_log") else y_tr
        if grid:
            gs = GridSearchCV(est, grid, cv=CV, n_jobs=-1, scoring="neg_root_mean_squared_error")
            gs.fit(X_tr, y_ajustado)
            modelo = gs.best_estimator_
        else:
            modelo = est.fit(X_tr, y_ajustado)
        pred = modelo.predict(te[features])
        if nombre.endswith("_log"):
            pred = np.exp(pred)          # se vuelve a la escala original
        col = f"pred_{nombre}"
        te[col] = pred
        filas.append({"modelo": nombre, **medir(y_te, pred, base),
                      "error_regiones_pct": error_region(te, col, target)})

    tabla = pd.DataFrame(filas)
    tabla.insert(0, "anio_test", test_year)
    tabla.insert(1, "n_train", len(tr))
    tabla.to_csv(out_csv, index=False)
    te.to_csv(out_csv.with_name(f"predicciones_{target}_{test_year}.csv"), index=False)
    return tabla, te


def filas_md(df: pd.DataFrame, decimales: int = 0) -> str:
    def numero(v):
        return f"{v:,.{decimales}f}"
    return "\n".join(
        "| {} | {} | {} | {:.1f}% | {:.3f} | {:+.3f} | {} |".format(
            r.modelo, numero(r.rmse), numero(r.mae), r.mape, r.r2,
            r.skill_vs_persistencia, r.error_regiones_pct)
        for r in df.itertuples())


def limite_teorico_arboles(panel: pd.DataFrame, test_year: int) -> dict:
    """Un modelo de arbol no puede predecir mas que el maximo target que vio en
    entrenamiento (sus hojas promedian valores del entrenamiento). Si la serie
    crece, el arbol queda obligado a subestimar la region dominante.
    Esto no es un diagnostico especulativo: es una cota superior demostrable."""
    tr = panel[panel["ANIO"] < test_year]
    te = panel[panel["ANIO"] == test_year]
    fila_tr = tr[tr["CODREGEO"] == 7]["estimation"]
    fila_te = te[te["CODREGEO"] == 7]["estimation"]
    return {
        "max_train": float(tr["estimation"].max()),
        "real_test": float(te["estimation"].max()),
        "subestimacion_minima_pct": float(100 * (te["estimation"].max() / tr["estimation"].max() - 1)),
        "rm_train": [round(v) for v in fila_tr.tolist()],
        "rm_test": [round(v) for v in fila_te.tolist()],
        "rm_rate_train": [round(v, 4) for v in tr[tr["CODREGEO"] == 7]["rate"].tolist()],
        "rm_rate_test": [round(v, 4) for v in te[te["CODREGEO"] == 7]["rate"].tolist()],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(PANEL_V2)
    n0 = len(panel)
    panel = panel.merge(estimation_lag1(), on=["CODREGEO", "ANIO"], how="left")
    assert len(panel) == n0 and panel["estimation_lag1"].notna().all()

    tot = total_nacional()
    anios_previos = [a for a in sorted(tot.index) if a < TEST_YEAR][-3:]
    crecimiento = float(np.mean([tot[a] / tot[a - 1] for a in anios_previos if a - 1 in tot.index]))
    total_proyectado = float(tot[TEST_YEAR - 1]) * crecimiento
    total_real = float(tot[TEST_YEAR])
    print(f"Total nacional {TEST_YEAR-1}: {tot[TEST_YEAR-1]:,.0f} | "
          f"proyectado {TEST_YEAR} (crecimiento pasado {crecimiento-1:.1%}): {total_proyectado:,.0f} | "
          f"real {TEST_YEAR}: {total_real:,.0f}")
    print(f"Panel: {sorted(int(a) for a in panel['ANIO'].unique())}, "
          f"{panel['CODREGEO'].nunique()} regiones")

    print("\n== target rate: test 2023 ==")
    t_rate, te_rate = evaluar_target(panel, "rate", FEATURES_RATE, TEST_YEAR,
                                     OUT_DIR / "comparacion_rate_test2023.csv")
    print("\n== target estimation: test 2023 ==")
    t_est, te_est = evaluar_target(panel, "estimation", FEATURES, TEST_YEAR,
                                   OUT_DIR / "comparacion_estimation_test2023.csv")
    print("\n== estabilidad: test 2022 (entrenando con 2021) ==")
    t_rate22, _ = evaluar_target(panel, "rate", FEATURES_RATE, 2022,
                                 OUT_DIR / "comparacion_rate_test2022.csv")
    t_est22, _ = evaluar_target(panel, "estimation", FEATURES, 2022,
                                OUT_DIR / "comparacion_estimation_test2022.csv")

    # ---- renormalizacion post-hoc ----
    print("\n== renormalizacion post-hoc (§3.1) ==")
    renorm = []
    for target, te, totales in (
        ("estimation", te_est, [("total_real (cota optimista)", total_real),
                                ("total_proyectado (honesta)", total_proyectado)]),
        ("rate", te_rate, [("suma = 1", 1.0)]),
    ):
        for col in [c for c in te.columns if c.startswith("pred_")] + ["persistencia"]:
            pred = te[col].to_numpy()
            for etiqueta, total in totales:
                ajustada = pred * (total / pred.sum())
                renorm.append({"target": target,
                               "modelo": f"{col} + renorm ({etiqueta})",
                               **medir(te[target].to_numpy(), ajustada, te["persistencia"].to_numpy())})
    t_renorm = pd.DataFrame(renorm)
    t_renorm.to_csv(OUT_DIR / "comparacion_renormalizada.csv", index=False)

    # ---- cota superior demostrable de los modelos de arbol ----
    cota = limite_teorico_arboles(panel, TEST_YEAR)
    print("\n== cota superior de los arboles (target estimation) ==")
    print(f"  max target visto en entrenamiento: {cota['max_train']:,.0f}")
    print(f"  target real del anio de test:      {cota['real_test']:,.0f}")
    print(f"  -> todo modelo de arbol subestima la RM al menos {cota['subestimacion_minima_pct']:.1f}%")

    # ---- resumen ----
    mejor_est = t_est.loc[t_est["rmse"].idxmin()]
    mejor_rate = t_rate.loc[t_rate["rmse"].idxmin()]
    rf_est = t_est.loc[t_est["modelo"] == "random_forest"].iloc[0]
    rf_rate = t_rate.loc[t_rate["modelo"] == "random_forest"].iloc[0]
    pers_est = t_est.iloc[0]
    pers_rate = t_rate.iloc[0]
    md = f"""# Evaluacion con baseline de persistencia

**Test principal:** {TEST_YEAR} (16 regiones). **Entrenamiento:** 2021-2022 (32 filas).
**Test secundario:** 2022 (entrenando con 2021), para ver si el ranking depende del anio.
**Script:** `src/evaluacion_baseline/evaluar_baseline.py`.

El baseline es la regla mas simple posible: *la region vale el proximo anio lo
mismo que este anio* (`rate_lag1` / `estimation_lag1`). El skill score es
`1 - MSE_modelo / MSE_persistencia`: **positivo = el modelo aporta sobre no
hacer nada**; negativo = aporta menos que copiar el anio anterior.

## 1. target `rate` (participacion) · test {TEST_YEAR}

| modelo | RMSE | MAE | MAPE | R2 | skill | error RM / Antofagasta / Valparaiso (%) |
|---|---|---|---|---|---|---|
{filas_md(t_rate, 5)}

## 2. target `estimation` (personas) · test {TEST_YEAR}

| modelo | RMSE | MAE | MAPE | R2 | skill | error RM / Antofagasta / Valparaiso (%) |
|---|---|---|---|---|---|---|
{filas_md(t_est)}

## 3. Estabilidad del ranking · test 2022

| modelo | RMSE | MAE | MAPE | R2 | skill | error RM / Antofagasta / Valparaiso (%) |
|---|---|---|---|---|---|---|
{filas_md(t_est22)}

## 4. Renormalizacion post-hoc (§3.1 del documento de la RM)

| variante | RMSE | MAE | MAPE | R2 | skill |
|---|---|---|---|---|---|
{chr(10).join("| {} | {:,.0f} | {:,.0f} | {:.1f}% | {:.3f} | {:+.3f} |".format(
    r.modelo, r.rmse, r.mae, r.mape, r.r2, r.skill_vs_persistencia) for r in t_renorm.itertuples())}

**Tres lecturas que matizan la propuesta §3.1 del documento de la RM:**

1. Con el total nacional **real**, la renormalizacion repara casi todo el error de
   los arboles en `estimation` (gradient boosting: skill -0.113 -> +0.975; extra
   trees: -0.139 -> +0.988). O sea, en ese target el error de los arboles es **de
   nivel, no de reparto**: se equivocan en cuanto crece el pais, no en como se
   reparte entre regiones.
2. En `rate` la renormalizacion **no** rescata al Random Forest (su skill sigue en
   -366): ahi el error si es de reparto (la RM se mezcla con regiones de escala
   intermedia) y reescalar no toca la causa.
3. Con un total **proyectado desde el pasado** (que subestima {100*(total_real/total_proyectado-1):.1f}%),
   la renormalizacion **empeora a todos los modelos** respecto de no
   renormalizar: el lineal baja de +0.571 a +0.376. No es una correccion gratis:
   exige un total nacional confiable, externo al modelo regional.

## 5. Por que los arboles fallan en `estimation`: cota demostrable, no hipotesis

Un modelo de arbol predice promediando valores **del entrenamiento** (sus hojas
guardan medias de targets vistos). Entonces su prediccion nunca puede superar el
maximo target del entrenamiento. Con el panel actual (2021-2022):

- maximo target visto en entrenamiento: **{cota['max_train']:,.0f}** (RM {cota['rm_train'][-1]:,.0f} en 2022)
- target real de la RM en el anio de test: **{cota['real_test']:,.0f}**
- por lo tanto **todo modelo de arbol subestima la RM al menos {cota['subestimacion_minima_pct']:.1f}%**, sin importar
  sus hiperparametros.

El Random Forest vigente predice la RM en 653.338 (-40,0%): esta **muy por debajo
incluso de esa cota**. Una regresion lineal no tiene ese limite porque extrapola
la tendencia. Esto reemplaza la hipotesis de la seccion 2 del documento de la RM
(bootstrap + `min_samples_leaf`) por un argumento mas simple y verificable: el
mecanismo es que **el arbol no puede extrapolar una serie que crece**.

En `rate` la cota no aplica (la tasa de la RM en entrenamiento, {cota['rm_rate_train']},
es casi igual a la de test, {cota['rm_rate_test']}), asi que ahi la causa es la
mezcla con regiones de escala intermedia que el documento ya describia.

## 6. Lectura

- **Los modelos de arboles no le ganan a la persistencia en ningun target.** En
  `estimation`, el Random Forest que hoy esta en el repo queda con skill
  {rf_est.skill_vs_persistencia:+.3f} y MAPE {rf_est.mape:.1f}%, contra
  {pers_est.mape:.1f}% de copiar el anio anterior. En `rate`, el Random Forest v2
  queda con skill {rf_rate.skill_vs_persistencia:+.3f} (MAPE {rf_rate.mape:.1f}%)
  contra {pers_rate.mape:.1f}% de la persistencia.
- **Mejor candidato en `estimation`: `{mejor_est.modelo}`** (RMSE {mejor_est.rmse:,.0f},
  MAPE {mejor_est.mape:.1f}%, skill {mejor_est.skill_vs_persistencia:+.3f}).
- En `rate` el mejor RMSE lo da `{mejor_rate.modelo}` (RMSE {mejor_rate.rmse:.5f}, skill
  {mejor_rate.skill_vs_persistencia:+.3f}), pero **no es estable**: en el test 2022 el mismo
  modelo cae a skill {float(t_rate22.loc[t_rate22['modelo'] == 'extra_trees_log', 'skill_vs_persistencia'].iloc[0]):+.1f}.
  El unico candidato con skill **positivo en los dos targets y en los dos anios de
  test** es la **regresion lineal multiple** (rate: +0.611 y +0.409; estimation: +0.571
  y +0.933), sin depender de la transformacion logaritmica — que en algunos casos
  explota (lineal sobre log: skill -2.773 en rate, -1.534 en estimation, con
  predicciones de hasta 1.145.746 personas de RMSE).
- En `rate`, la renormalizacion a suma 1 es **gratis y legitima** (las
  participaciones tienen que sumar 1 por definicion): el lineal pasa de skill
  +0.611 a +0.788 (RMSE {float(t_renorm[(t_renorm.target == 'rate') & (t_renorm.modelo.str.startswith('pred_lineal '))].rmse.iloc[0]):.5f}).
  En `estimation` no es gratis, porque exige conocer el total nacional.
- Sobre el argumento de sobreajuste que llevo a descartar el modelo lineal
  (RMSE train x21,7 en test): el error absoluto crece porque el nivel de la
  serie crece (el total nacional subio {100*(total_real/tot[TEST_YEAR-1]-1):.0f}%
  en el anio de test), no porque el modelo se degrade. En error **relativo**
  (MAPE) el lineal es el mejor y estable en los dos anios de test.
- La escala logaritmica mejora el error de las regiones pequenas, pero **no
  arregla la RM por si sola**.
- La renormalizacion post-hoc **solo sirve si el total nacional es confiable**:
  con el total real el error baja fuerte, pero con un total proyectado desde el
  pasado — que subestima en {100*(total_real/total_proyectado-1):.1f}% — el error **sube**
  respecto del mismo modelo sin renormalizar. Como el total real no se conoce al
  pronosticar, sirve como diagnostico (cuanto del error es reparto y cuanto es
  nivel), no como solucion.
- **El panel es 2021-2023**, asi que agregar 2020 al entrenamiento — la
  solucion que el documento de la RM considera mas robusta — exige reconstruir
  `dataset_region_v2.csv`: el dataset del modelo 1 no se puede regenerar (su
  fuente cruda se perdio) y el panel hereda esos anios. Las 8 features si son
  derivables de `dataset_combinado.csv` + el xlsx de solicitudes para 2019-2020,
  asi que el trabajo es acotado y ataca el mecanismo exacto de la RM (pasar de
  2 a 4 filas de entrenamiento para la region dominante).
- **Decision sugerida a documentar:** reemplazar el criterio "estabilidad
  train/test" por "skill score positivo contra persistencia + error aceptable en
  la RM + sin sesgo sistematico en las regiones grandes".
"""
    (ROOT / "docs" / "evaluacion_baseline.md").write_text(md, encoding="utf-8")

    for nombre, t in (("rate 2023", t_rate), ("estimation 2023", t_est),
                      ("estimation 2022", t_est22)):
        print(f"\n-- {nombre} --")
        print(t[["modelo", "rmse", "mae", "mape", "r2", "skill_vs_persistencia"]]
              .to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    print(f"\n-> {ROOT / 'docs' / 'evaluacion_baseline.md'}")
    for f in sorted(OUT_DIR.glob("*.csv")):
        print(f"-> {f.name}")


if __name__ == "__main__":
    main()
