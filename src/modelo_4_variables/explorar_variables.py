"""Modelo 4 (exploracion): que variables conviene usar, y para que pregunta.

Motivo. Los modelos 1-3 se evaluan sobre el NIVEL de la serie (cuantas personas
o que participacion), y ahi la persistencia domina: el R2 sale 0,9966 y no es
un logro del modelo sino una propiedad de los datos (el reparto regional es muy
estable). Ese diagnostico esta documentado en docs/glosario_hiperparametros_y_metricas.md.

Este script explora tres cosas, en este orden:
  1. Cuanta senal lleva cada variable candidata sobre cada pregunta posible.
  2. Cual de las preguntas tiene sentido hacerse: donde la persistencia NO
     regala el resultado.
  3. Que combinacion de variables conviene para la pregunta elegida.

Regla que se respeta: el ranking de variables se arma con los anios de
ENTRENAMIENTO (2021-2022) y el test (2023) se usa una sola vez, al final, para
reportar. Elegir variables mirando el anio de prueba seria fuga de datos.

Uso:  python src/modelo_4_variables/explorar_variables.py
      -> outputs/exploracion_variables.md
      -> outputs/ranking_variables_<pregunta>.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parents[1]
sys.path.insert(0, str(ROOT / "src" / "common"))
sys.path.insert(0, str(ROOT / "src" / "modelo_3_conteo"))
from config import SEED, TEST_YEAR  # noqa: E402

PANEL_V2 = ROOT / "src" / "modelo_2_regional_v2" / "outputs" / "dataset_region_v2.csv"
COMBINADO = ROOT / "dataset_combinado.csv"
OUT_DIR = AQUI / "outputs"

# Variables del censo: son de 2024, o sea POSTERIORES al anio de prueba.
# Sirven para describir, pero NO se pueden usar para predecir 2023.
NO_PREDICTIVAS = ["censo_n_migrantes", "censo_mean_edad", "censo_mean_escolaridad",
                  "censo_pct_mujer", "censo_pct_urbano", "censo_pct_llegada_reciente",
                  "censo_pct_ocupado", "censo_pct_venezuela"]


def estimation_lag1() -> pd.DataFrame:
    """ESTIMACION agregada por region-anio, rezagada a t-1."""
    dc = pd.read_csv(COMBINADO)
    dc = dc[dc["CODREGEO"] != 17]
    agg = dc.groupby(["CODREGEO", "AÑO"])["ESTIMACION"].sum().reset_index()
    agg["AÑO"] = agg["AÑO"] + 1
    return agg.rename(columns={"ESTIMACION": "estimation_lag1", "AÑO": "ANIO"})


def cargar_panel() -> pd.DataFrame:
    p = pd.read_csv(PANEL_V2)
    n0 = len(p)
    p = p.merge(estimation_lag1(), on=["CODREGEO", "ANIO"], how="left")
    assert len(p) == n0 and p["estimation_lag1"].notna().all()
    return p


def preguntas(p: pd.DataFrame) -> dict:
    """Las cuatro preguntas posibles, con su target y su referencia (persistencia).

    La referencia es "lo que dijo el ano anterior", en la misma escala del target:
      - nivel (rate / estimation)  -> el valor del ano anterior
      - crecimiento               -> 0 (no crecer)
      - redistribucion            -> 0 (no cambiar de peso)
    """
    return {
        "nivel_rate": (
            "Cuanto pesa cada region del pais",
            p["rate"], p["rate_lag1"],
        ),
        "nivel_estimation": (
            "Cuantas personas hay en cada region",
            p["estimation"], p["estimation_lag1"],
        ),
        "crecimiento": (
            "Cuanto crece cada region respecto del ano anterior",
            np.log(p["estimation"] / p["estimation_lag1"]),
            pd.Series(0.0, index=p.index),
        ),
        "redistribucion": (
            "Cuanto cambia el peso de cada region (rate t - rate t-1)",
            p["rate"] - p["rate_lag1"],
            pd.Series(0.0, index=p.index),
        ),
    }


def skill(y: np.ndarray, pred: np.ndarray, ref: np.ndarray) -> float:
    """1 - MSE_modelo / MSE_referencia (Murphy 1988). 0 = empata con no hacer nada."""
    mse, mse_ref = ((y - pred) ** 2).mean(), ((y - ref) ** 2).mean()
    return float(1 - mse / mse_ref) if mse_ref > 0 else float("nan")


def medir_pregunta(p: pd.DataFrame, nombre: str, desc: str, y: pd.Series,
                   ref: pd.Series, candidatas: list, usar_censo: bool) -> dict:
    """Evalua la pregunta: baseline, senal de cada variable y importancia."""
    tr, te = p["ANIO"] < TEST_YEAR, p["ANIO"] == TEST_YEAR
    y_tr, y_te = y[tr].to_numpy(), y[te].to_numpy()
    ref_te = ref[te].to_numpy()

    # --- linea base: que pasa si no se usa ninguna variable
    base = {
        "skill_persistencia": 0.0,
        "rmse_persistencia": float(np.sqrt(((y_te - ref_te) ** 2).mean())),
        "r2_persistencia": float(r2_score(y_te, ref_te)) if y_te.std() > 0 else float("nan"),
        "desv_target": float(y_te.std()),
    }

    cols = [c for c in candidatas if usar_censo or c not in NO_PREDICTIVAS]

    # --- senal de cada variable por separado, medida en ENTRENAMIENTO
    filas = []
    for c in cols:
        x_tr = p.loc[tr, [c]].to_numpy()
        x_te = p.loc[te, [c]].to_numpy()
        # correlacion con el target (en entrenamiento)
        corr = float(np.corrcoef(p.loc[tr, c], y_tr)[0, 1]) if p.loc[tr, c].std() > 0 else np.nan
        # informacion mutua (capta relaciones no lineales)
        mi = float(mutual_info_regression(x_tr, y_tr, random_state=SEED)[0])
        # modelo de una sola variable, medido en el test
        m = Pipeline([("esc", StandardScaler()), ("m", LinearRegression())]).fit(x_tr, y_tr)
        pred_te = m.predict(x_te)
        filas.append({
            "variable": c, "correlacion": corr, "info_mutua": mi,
            "r2_test_1var": float(r2_score(y_te, pred_te)) if y_te.std() > 0 else np.nan,
            "skill_test_1var": skill(y_te, pred_te, ref_te),
        })

    # --- importancia por permutacion: que variables usa de verdad un bosque
    X_tr = p.loc[tr, cols].to_numpy()
    X_te = p.loc[te, cols].to_numpy()
    rf = RandomForestRegressor(n_estimators=500, max_depth=3, min_samples_leaf=2,
                               random_state=SEED).fit(X_tr, y_tr)
    imp = permutation_importance(rf, X_te, y_te, n_repeats=30, random_state=SEED)
    df = pd.DataFrame(filas)
    df["importancia_perm"] = imp.importances_mean
    df["importancia_std"] = imp.importances_std
    df["pregunta"] = nombre
    df["usable_para_predecir"] = ~df["variable"].isin(NO_PREDICTIVAS)
    df = df.sort_values("importancia_perm", ascending=False)

    return {"nombre": nombre, "desc": desc, "base": base, "tabla": df,
            "skill_bosque_test": skill(y_te, rf.predict(X_te), ref_te)}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = cargar_panel()
    candidatas = [c for c in p.columns if c.endswith("_lag1") and c not in
                  ("rate_lag1", "estimation_lag1")] + NO_PREDICTIVAS

    print(f"Panel: {len(p)} filas | {len(candidatas)} variables candidatas | "
          f"entrenamiento {sorted(p[p['ANIO'] < TEST_YEAR]['ANIO'].unique())} | "
          f"test {TEST_YEAR}")

    resultados = []
    for nombre, (desc, y, ref) in preguntas(p).items():
        r = medir_pregunta(p, nombre, desc, y, ref, candidatas, usar_censo=True)
        resultados.append(r)
        print(f"\n--- {nombre}: {desc}")
        print(f"    no hacer nada: RMSE {r['base']['rmse_persistencia']:.4f}  "
              f"R2 {r['base']['r2_persistencia']:.4f}")
        print(f"    bosque con todo: skill {r['skill_bosque_test']:+.3f}")
        top = r["tabla"].head(5)
        for _, f in top.iterrows():
            print(f"      {f['variable']:<32} imp {f['importancia_perm']:+.4f} "
                  f"corr {f['correlacion']:+.3f} skill_1var {f['skill_test_1var']:+.3f}")
        r["tabla"].to_csv(OUT_DIR / f"ranking_variables_{nombre}.csv", index=False)

    # --- informe
    lineas = [
        "# Exploracion de variables y de preguntas posibles", "",
        "Generado por `src/modelo_4_variables/explorar_variables.py`.", "",
        f"Panel: {len(p)} filas (16 regiones x {p['ANIO'].nunique()} anios). "
        f"Entrenamiento: {sorted(p[p['ANIO'] < TEST_YEAR]['ANIO'].unique())}. "
        f"Test: {TEST_YEAR}.", "",
        "**El ranking de variables se arma con los anios de entrenamiento y el test "
        "se usa una sola vez para reportar.** Elegir variables mirando el anio de "
        "prueba seria fuga de datos.", "",
        "## Resumen: cual pregunta tiene sentido", "",
        "| pregunta | que responde | RMSE de no hacer nada | R2 de no hacer nada | "
        "skill del bosque (con todas las variables) |", "|---|---|---|---|---|",
    ]
    for r in resultados:
        b = r["base"]
        lineas.append(
            f"| `{r['nombre']}` | {r['desc']} | {b['rmse_persistencia']:.4f} | "
            f"{b['r2_persistencia']:.4f} | **{r['skill_bosque_test']:+.3f}** |")
    lineas += ["", "Lectura: donde el skill del bosque es cercano a 0, las variables no "
               "aportan nada sobre copiar el ano anterior; esa pregunta esta regalada por "
               "la persistencia. Donde el skill es negativo, el bosque **pierde** contra "
               "no hacer nada. La pregunta con skill mas alto es la unica donde el modelo "
               "aporta algo real.", ""]

    for r in resultados:
        b = r["base"]
        lineas += [f"## Pregunta `{r['nombre']}`", "", f"*{r['desc']}*", "",
                   f"- No hacer nada: RMSE {b['rmse_persistencia']:.4f}, "
                   f"R2 {b['r2_persistencia']:.4f}, desviacion del target en test "
                   f"{b['desv_target']:.4f}",
                   f"- Bosque con todas las variables: skill "
                   f"**{r['skill_bosque_test']:+.3f}**", "",
                   "| variable | correlacion | info. mutua | skill sola | importancia | "
                   "se puede usar para predecir |", "|---|---|---|---|---|---|"]
        for _, f in r["tabla"].iterrows():
            usable = "si" if f["usable_para_predecir"] else "**NO** (censo 2024)"
            lineas.append(
                f"| `{f['variable']}` | {f['correlacion']:+.3f} | {f['info_mutua']:.4f} | "
                f"{f['skill_test_1var']:+.3f} | {f['importancia_perm']:+.4f} "
                f"+- {f['importancia_std']:.4f} | {usable} |")
        lineas.append("")

    (OUT_DIR / "exploracion_variables.md").write_text("\n".join(lineas))
    print(f"\n-> {OUT_DIR / 'exploracion_variables.md'}")


if __name__ == "__main__":
    main()
