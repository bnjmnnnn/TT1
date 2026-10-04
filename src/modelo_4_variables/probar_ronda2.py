"""Ronda 2 del modelo 4: se puede subir el R2, o ya llego a su limite?

Pregunta del usuario: "en este punto subir un 5% el R2 es una noticia muy
alentadora". Este script prueba si es alcanzable y, si no, donde esta el techo.

Que se agrega respecto de la ronda 1:
  1. Variables DERIVADAS con mecanismo (no estaban en el panel):
     - `presion_lag1`      = sol_share_lag1 - rate_lag1
        Una region que recibe una fraccion de las aprobaciones mayor que su peso
        actual esta "presionando" al alza; deberia ganar peso.
     - `sol_por_mil_lag1`  = sol_otorgadas_lag1 / estimation_lag1 * 1000
        Intensidad de solicitudes por cada mil habitantes (normaliza por tamano).
     - `redistribucion_lag1` = rate_lag1 - rate_lag2
        El cambio de peso del ano anterior. Si la desconcentracion es persistente,
        esta variable deberia predecir la del ano siguiente.
     - `rate_lag2`         = el peso de hace dos anos (reversion mas lenta).
     Se construyen desde dataset_combinado.csv (que cubre 2018-2023), asi que no
   hay que rehacer el panel.
  2. Transformacion multiplicativa del target: log(rate_t / rate_lag1).
  3. Regularizacion (Ridge/Lasso/ElasticNet) para el caso de muchas variables.
  4. Combinacion de los tres modelos (promedio simple).
  5. ANALISIS DE TECHO: el R2 que se obtendria ajustando los mismos datos EN el
     ano de prueba (oraculo, no es un resultado). Si el techo esta cerca del
     desempeno fuera de muestra, no hay modelo que lo mejore: habria que mejorar
     los datos o las variables.

Regla de honestidad: para elegir una configuracion se mira la validacion interna
(KFold sobre los anos de entrenamiento) y la ESTABILIDAD entre los dos anos de
test. Elegir la que mejor le va en 2023 seria fuga de datos.

Uso:  python src/modelo_4_variables/probar_ronda2.py
      -> outputs/ronda2_metricas.csv
      -> outputs/ronda2_observaciones.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parents[1]
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(ROOT / "src" / "common"))
from config import SEED  # noqa: E402
from explorar_variables import COMBINADO, cargar_panel  # noqa: E402

OUT_DIR = AQUI / "outputs"
CV = KFold(n_splits=5, shuffle=True, random_state=SEED)

BASE5 = ["sol_otorgadas_lag1", "sol_share_lag1", "sol_total_lag1",
         "pct_peru_lag1", "mean_age_lag1"]
DERIVADAS = ["presion_lag1", "sol_por_mil_lag1", "redistribucion_lag1", "rate_lag2"]
TODAS15 = ["pct_women_lag1", "mean_age_lag1", "pct_irregular_lag1", "pct_venezuela_lag1",
           "pct_peru_lag1", "pct_colombia_lag1", "pct_haiti_lag1", "pct_bolivia_lag1",
           "sol_total_lag1", "sol_otorgadas_lag1", "sol_pct_otorga_lag1", "sol_share_lag1",
           "macro_inflacion_origen_lag1", "macro_crecimiento_pib_origen_lag1",
           "macro_desempleo_origen_lag1"]


def agregar_derivadas(p: pd.DataFrame) -> pd.DataFrame:
    """Agrega rate_lag2 y las variables derivadas, desde dataset_combinado.

    Ojo: `rate_lag1` ya viene en el panel; aca solo se agrega el de hace DOS anos.
    """
    dc = pd.read_csv(COMBINADO)
    dc = dc[dc["CODREGEO"] != 17]
    agg = dc.groupby(["CODREGEO", "AÑO"])["ESTIMACION"].sum().reset_index()
    tot = agg.groupby("AÑO")["ESTIMACION"].transform("sum")
    agg["rate_hist"] = agg["ESTIMACION"] / tot
    m = agg[["CODREGEO", "rate_hist"]].copy()
    m["ANIO"] = agg["AÑO"] + 2
    m = m.rename(columns={"rate_hist": "rate_lag2"})
    p = p.merge(m, on=["CODREGEO", "ANIO"], how="left")
    # control: el rate_lag1 del panel debe coincidir con el rate recalculado del
    # anio anterior (si no coinciden, las dos fuentes no son comparables)
    chk = agg[["CODREGEO", "rate_hist"]].copy()
    chk["ANIO"] = agg["AÑO"] + 1
    chk = chk.rename(columns={"rate_hist": "rate_lag1_chk"})
    p = p.merge(chk, on=["CODREGEO", "ANIO"], how="left")
    dif = (p["rate_lag1"] - p["rate_lag1_chk"]).abs().max()
    print(f"control rate_lag1 panel vs recalculado: diferencia maxima {dif:.2e}")
    assert dif < 1e-9, "rate_lag1 del panel no coincide con el recalculado"
    p = p.drop(columns=["rate_lag1_chk"])
    p["redistribucion_lag1"] = p["rate_lag1"] - p["rate_lag2"]
    p["presion_lag1"] = p["sol_share_lag1"] - p["rate_lag1"]
    p["sol_por_mil_lag1"] = p["sol_otorgadas_lag1"] / p["estimation_lag1"] * 1000
    return p


def skill(y, pred, ref) -> float:
    mse, mse_ref = ((y - pred) ** 2).mean(), ((y - ref) ** 2).mean()
    return float(1 - mse / mse_ref) if mse_ref > 0 else float("nan")


def evaluar(y_tr, y_te, X_tr, X_te, modelo, cols, cv_ok=True) -> dict:
    modelo.fit(X_tr, y_tr)
    pred = modelo.predict(X_te)
    ref = np.zeros_like(y_te)
    out = {
        "rmse": float(np.sqrt(((y_te - pred) ** 2).mean())),
        "r2": float(r2_score(y_te, pred)),
        "skill": skill(y_te, pred, ref),
        "corr_pred_real": float(np.corrcoef(pred, y_te)[0, 1]),
        "aciertos_direccion": int((np.sign(pred) == np.sign(y_te)).sum()),
        "n_vars": len(cols),
    }
    if cv_ok:
        # validacion interna SOLO sobre los anos de entrenamiento
        out["cv_rmse"] = float(-cross_val_score(modelo, X_tr, y_tr, cv=CV,
                                                scoring="neg_root_mean_squared_error").mean())
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = agregar_derivadas(cargar_panel())
    p["redistribucion"] = p["rate"] - p["rate_lag1"]
    nulos = int(p[["rate_lag2", "redistribucion_lag1", "presion_lag1",
                   "sol_por_mil_lag1"]].isna().sum().sum())
    print(f"nulos en las derivadas: {nulos}")
    assert nulos == 0, "derivadas con nulos"

    configs = []
    GB = lambda: GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=2,
                                           min_samples_leaf=2, random_state=SEED)
    RF = lambda: RandomForestRegressor(n_estimators=500, max_depth=3, min_samples_leaf=2,
                                       random_state=SEED)
    LIN = lambda: Pipeline([("e", StandardScaler()), ("m", LinearRegression())])
    RIDGE = lambda: Pipeline([("e", StandardScaler()), ("m", Ridge(alpha=1.0))])
    LASSO = lambda: Pipeline([("e", StandardScaler()), ("m", Lasso(alpha=0.0001))])
    ENET = lambda: Pipeline([("e", StandardScaler()),
                             ("m", ElasticNet(alpha=0.0005, l1_ratio=0.5))])

    configs += [
        ("ronda1: GB base5", GB, BASE5),
        ("base5 + derivadas", GB, BASE5 + DERIVADAS),
        ("solo derivadas", GB, DERIVADAS),
        ("base5 + redist_lag1", GB, BASE5 + ["redistribucion_lag1"]),
        ("base5 + presion", GB, BASE5 + ["presion_lag1"]),
        ("base5 + rate_lag2", GB, BASE5 + ["rate_lag2"]),
        ("1 var + derivadas", GB, ["sol_otorgadas_lag1"] + DERIVADAS),
        ("base5, RF", RF, BASE5),
        ("base5 + derivadas, RF", RF, BASE5 + DERIVADAS),
        ("base5, lineal", LIN, BASE5),
        ("1 var, lineal", LIN, ["sol_otorgadas_lag1"]),
        ("15 vars, Ridge", RIDGE, TODAS15),
        ("15 vars, Lasso", LASSO, TODAS15),
        ("15 vars, ElasticNet", ENET, TODAS15),
        ("todas 19, GB", GB, TODAS15 + DERIVADAS),
        ("todas 19, Ridge", RIDGE, TODAS15 + DERIVADAS),
    ]

    filas = []
    for nombre, ctor, cols in configs:
        for anio in (2023, 2022):
            tr, te = p[p["ANIO"] < anio], p[p["ANIO"] == anio]
            y_tr, y_te = tr["redistribucion"].to_numpy(), te["redistribucion"].to_numpy()
            m = evaluar(y_tr, y_te, tr[cols].to_numpy(), te[cols].to_numpy(), ctor(), cols)
            filas.append({"config": nombre, "anio_test": anio, **m})
    df = pd.DataFrame(filas)

    # --- target multiplicativo: log(rate_t / rate_lag1), se reconstruye el nivel
    filas_mult = []
    for nombre, cols in (("base5", BASE5), ("base5 + derivadas", BASE5 + DERIVADAS)):
        for anio in (2023, 2022):
            tr, te = p[p["ANIO"] < anio], p[p["ANIO"] == anio]
            y_tr = np.log(tr["rate"] / tr["rate_lag1"]).to_numpy()
            m = GB().fit(tr[cols].to_numpy(), y_tr)
            rate_pred = te["rate_lag1"].to_numpy() * np.exp(m.predict(te[cols].to_numpy()))
            pred = rate_pred - te["rate_lag1"].to_numpy()
            y_te = te["redistribucion"].to_numpy()
            filas_mult.append({
                "config": f"target multiplicativo, {nombre}", "anio_test": anio,
                "n_vars": len(cols), "rmse": float(np.sqrt(((y_te - pred) ** 2).mean())),
                "r2": float(r2_score(y_te, pred)), "skill": skill(y_te, pred, np.zeros_like(y_te)),
                "corr_pred_real": float(np.corrcoef(pred, y_te)[0, 1]),
                "aciertos_direccion": int((np.sign(pred) == np.sign(y_te)).sum()),
                "cv_rmse": np.nan})
    df = pd.concat([df, pd.DataFrame(filas_mult)], ignore_index=True)

    # --- el techo: ajustar en el propio ano de prueba (oraculo, NO es resultado)
    techos = {}
    for anio in (2023, 2022):
        sub = p[p["ANIO"] == anio]
        y = sub["redistribucion"].to_numpy()
        for etq, cols in (("base5", BASE5), ("base5+derivadas", BASE5 + DERIVADAS),
                          ("15 vars", TODAS15)):
            m = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=2,
                                          min_samples_leaf=2, random_state=SEED)
            m.fit(sub[cols].to_numpy(), y)
            techos[(anio, etq)] = float(r2_score(y, m.predict(sub[cols].to_numpy())))
    # y el techo de una recta: con 5 variables y 16 filas, cuanto explica una
    # combinacion lineal ajustada en el propio ano (techo de la familia lineal)
    sub = p[p["ANIO"] == 2023]
    y = sub["redistribucion"].to_numpy()
    lin = Pipeline([("e", StandardScaler()), ("m", LinearRegression())])
    lin.fit(sub[BASE5].to_numpy(), y)
    techo_lineal_2023 = float(r2_score(y, lin.predict(sub[BASE5].to_numpy())))

    # --- persistencia de la redistribucion (la variable clave de la ronda 2)
    corr_redist = float(np.corrcoef(p["redistribucion_lag1"], p["redistribucion"])[0, 1])

    df.to_csv(OUT_DIR / "ronda2_metricas.csv", index=False)

    # --- informe
    L = ["# Ronda 2 del modelo 4: hasta donde se puede subir el R2", "",
         "Generado por `src/modelo_4_variables/probar_ronda2.py`.", "",
         "**Pregunta:** ¿es alcanzable un R2 mas alto, o el modelo ya llego a su limite?",
         "",
         "El target sigue siendo `redistribucion = rate_t - rate_t-1`. Se agregan "
         "variables derivadas con mecanismo, transformaciones y regularizacion, y se "
         "calcula el TECHO de la familia de modelos.", "",
         f"**Correlacion entre la redistribucion de un ano y la del siguiente: "
         f"{corr_redist:+.3f}.** Es el dato que decide si hay margen: si fuera alta, la "
         "variable `redistribucion_lag1` por si sola explicaria el fenomeno.", "",
         "## Resultados por configuracion", "",
         "| configuracion | vars | test | RMSE | R2 | corr(pred,real) | direccion | "
         "CV interno (RMSE) |", "|---|---|---|---|---|---|---|---|"]
    for _, f in df.sort_values(["config", "anio_test"], ascending=[True, False]).iterrows():
        L.append(f"| {f['config']} | {f['n_vars']} | {f['anio_test']} | {f['rmse']:.5f} | "
                 f"**{f['r2']:+.3f}** | {f['corr_pred_real']:+.3f} | "
                 f"{f['aciertos_direccion']}/16 | {f['cv_rmse']:.4f} |")

    L += ["", "## El techo de la familia de modelos", "",
          "Se ajusta el mismo modelo **dentro del ano de prueba** (oraculo). No es un "
          "resultado: es el maximo que esa familia de modelos podria dar si hubiera "
          "aprendido la relacion perfectamente. Si el desempeno fuera de muestra esta "
          "cerca del techo, no hay ajuste que lo mejore.", "",
          "| conjunto | R2 oraculo 2023 | R2 oraculo 2022 | R2 fuera de muestra 2023 |", "|---|---|---|---|"]
    for etq in ("base5", "base5+derivadas", "15 vars"):
        fuera = df[(df.config.str.contains("GB")) & (df.anio_test == 2023)]
        L.append(f"| {etq} | {techos[(2023, etq)]:+.3f} | {techos[(2022, etq)]:+.3f} | — |")
    L += ["", f"Techo de una combinacion lineal (base5, ajustada en 2023): "
          f"**{techo_lineal_2023:+.3f}**.", ""]

    (OUT_DIR / "ronda2_observaciones.md").write_text("\n".join(L))
    print(df.pivot_table(index="config", columns="anio_test", values="r2").to_string())
    print(f"\ncorrelacion redistribucion t vs t-1: {corr_redist:+.3f}")
    print(f"techo oraculo 2023: base5 {techos[(2023,'base5')]:+.3f} | "
          f"base5+der {techos[(2023,'base5+derivadas')]:+.3f} | "
          f"lineal base5 {techo_lineal_2023:+.3f}")
    print(f"-> {OUT_DIR / 'ronda2_observaciones.md'}")


if __name__ == "__main__":
    main()
