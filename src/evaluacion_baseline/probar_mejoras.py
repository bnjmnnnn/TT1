"""Prueba de mejoras candidatas para los modelos del TT.

Cada candidato ataca un problema concreto que quedo identificado en
docs/evaluacion_baseline.md:

  P1. El nivel de la serie no se puede pronosticar con arboles (no extrapolan).
  P2. El target `rate` es una participacion: tiene que sumar 1 por definicion.
  P3. El anio de test (2023) crecio 13% sobre 2022: un shock de nivel que
      castiga a todos los modelos en RMSE y esconde quien reparte mejor.
  P4. n es chico (32 filas de entrenamiento): modelos con 8 features + intercepto
      estan al limite de grados de libertad.

Candidatos (todos leakage-free: solo usan anios < anio de test):

  persistencia          y_hat = y_{t-1}
  deriva_nacional       y_hat = y_{t-1} * crecimiento nacional promedio
  deriva_regional       y_hat = y_{t-1} * crecimiento propio promedio de la region
  lineal / ridge / lasso  misma matriz de features, distinta regularizacion
  combinacion           lambda * modelo + (1 - lambda) * persistencia  (lambda por CV interno)
  crecimiento           se modela el log-crecimiento (t-1 -> t) en vez del nivel
  jerarquico            total nacional (deriva) x participaciones (persistencia o logit-OLS)
  logit_rate            (solo rate) logit de la participacion + renormalizacion a suma 1

Reporta skill score, MAPE y que fraccion del error cuadratico aporta la RM, mas
un bootstrap pareado por regiones para saber si la diferencia es distinguible del
ruido con n=16.

Uso:  python src/evaluacion_baseline/probar_mejoras.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from evaluar_baseline import (  # noqa: E402
    COMBINADO, FEATURES, FEATURES_RATE, GB_PARAMS, PANEL_V2, OUT_DIR, RF_PARAMS,
    TEST_YEAR, estimation_lag1, medir,
)

ALPHA_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
LAMBDA_GRID = np.linspace(0, 1, 21)


# ---------------------------------------------------------------- utilidades
def crecimiento_nacional(total: pd.Series, hasta: int) -> float:
    """Crecimiento nacional promedio anual usando solo anios <= `hasta`."""
    anios = [a for a in sorted(total.index) if a <= hasta and a - 1 in total.index]
    return float(np.mean([total[a] / total[a - 1] for a in anios[-3:]]))


def error_cuadratico(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return (y - pred) ** 2


def fraccion_mse_rm(panel_test: pd.DataFrame, y: np.ndarray, pred: np.ndarray) -> float:
    ec = error_cuadratico(y, pred)
    return float(100 * ec[panel_test["CODREGEO"].to_numpy() == 7].sum() / ec.sum())


def bootstrap_skill(y, pred, base, n_iter: int = 5000, semilla: int = 42) -> tuple:
    """IC 95% de la diferencia de skill por bootstrap pareado sobre las 16 regiones.
    Es mas honesto que un test asintotico con n=16."""
    rng = np.random.default_rng(semilla)
    n = len(y)
    difs = np.empty(n_iter)
    ec_m, ec_b = error_cuadratico(y, pred), error_cuadratico(y, base)
    for i in range(n_iter):
        idx = rng.integers(0, n, n)
        difs[i] = 1 - ec_m[idx].mean() / ec_b[idx].mean()
    return float(np.percentile(difs, 2.5)), float(np.percentile(difs, 97.5))


def diebold_mariano(y, pred1, pred2, h: int = 1) -> tuple:
    """DM con correccion Harvey-Leybourne-Newbold (muestras chicas).
    Devuelve (estadistico, p bilateral). Advertencia: las 16 observaciones son
    regiones en un mismo anio, no una serie de tiempo; la correlacion espacial
    viola la independencia y hace que el p-valor sea optimista. Se reporta por
    convencion, pero la evidencia fuerte es el bootstrap de arriba."""
    from scipy import stats  # scipy viene con scikit-learn
    d = error_cuadratico(y, pred1) - error_cuadratico(y, pred2)
    n = len(d)
    d_barra = d.mean()
    var = d.var(ddof=1) / n
    if var <= 0:
        return 0.0, 1.0
    dm = d_barra / np.sqrt(var)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = dm * hln
    p = 2 * (1 - stats.t.cdf(abs(dm_hln), df=n - 1))
    return float(dm_hln), float(p)


# ---------------------------------------------------------------- candidatos
def candidato_regularizado(X_tr, y_tr, X_te, clase):
    mejor, mejor_error = None, np.inf
    for alpha in ALPHA_GRID:
        # validacion temporal interna: ultimo anio de entrenamiento como validacion
        anios = sorted(X_tr.index.get_level_values("ANIO").unique())
        if len(anios) >= 2:
            m = X_tr.index.get_level_values("ANIO") < anios[-1]
            modelo = clase(alpha=alpha).fit(X_tr[m], y_tr[m])
            error = np.sqrt(np.mean((y_tr[~m] - modelo.predict(X_tr[~m])) ** 2))
        else:
            modelo = clase(alpha=alpha).fit(X_tr, y_tr)
            error = np.sqrt(np.mean((y_tr - modelo.predict(X_tr)) ** 2))
        if error < mejor_error:
            mejor, mejor_error = alpha, error
    return clase(alpha=mejor).fit(X_tr, y_tr), mejor


def candidato_combinacion(pred_modelo, pred_persistencia, y_tr, pred_tr_modelo, pred_tr_persist):
    """lambda optimo sobre el anio de validacion interno (el ultimo de train)."""
    mejor_lambda, mejor_error = 0.0, np.inf
    for lam in LAMBDA_GRID:
        error = np.sqrt(np.mean((y_tr - (lam * pred_tr_modelo + (1 - lam) * pred_tr_persist)) ** 2))
        if error < mejor_error:
            mejor_lambda, mejor_error = lam, error
    return mejor_lambda * pred_modelo + (1 - mejor_lambda) * pred_persistencia, mejor_lambda


def evaluar(panel: pd.DataFrame, total: pd.Series, target: str, test_year: int) -> pd.DataFrame:
    features = FEATURES_RATE if target == "rate" else FEATURES
    tr = panel[panel["ANIO"] < test_year].copy()
    te = panel[panel["ANIO"] == test_year].reset_index(drop=True)
    base_col = "rate_lag1" if target == "rate" else "estimation_lag1"
    y_te, base = te[target].to_numpy(), te[base_col].to_numpy()

    # indice (region, anio) para poder separar el anio de validacion interna
    tr = tr.set_index(["CODREGEO", "ANIO"])
    te_idx = te.set_index(["CODREGEO", "ANIO"])
    X_tr, y_tr = tr[features], tr[target].to_numpy()
    X_te = te_idx[features]
    anios_tr = sorted(tr.index.get_level_values("ANIO").unique())
    ultimo = anios_tr[-1]
    m_val = tr.index.get_level_values("ANIO") == ultimo

    resultados, predicciones = {}, {}

    def agregar(nombre, pred, nota=""):
        predicciones[nombre] = pred
        resultados[nombre] = {**medir(y_te, pred, base),
                              "pct_mse_rm": fraccion_mse_rm(te, y_te, pred), "nota": nota}

    # --- baselines con deriva
    agregar("persistencia", base)
    g_nac = crecimiento_nacional(total, test_year - 1)
    agregar("deriva_nacional", base * g_nac, f"g={g_nac-1:+.2%}")
    g_reg = {}
    for cod in te["CODREGEO"]:
        serie = panel[(panel["CODREGEO"] == cod) & (panel["ANIO"] < test_year)].sort_values("ANIO")
        valores = serie[target].to_numpy()
        crec = [valores[i] / valores[i - 1] for i in range(1, len(valores)) if valores[i - 1] > 0]
        g_reg[cod] = float(np.mean(crec)) if crec else 1.0
    agregar("deriva_regional", np.array([base[i] * g_reg[c] for i, c in enumerate(te["CODREGEO"])]))

    # --- regularizacion
    for nombre, clase in (("ridge", Ridge), ("lasso", Lasso)):
        modelo, alpha = candidato_regularizado(X_tr, y_tr, X_te, clase)
        agregar(nombre, modelo.predict(X_te), f"alpha={alpha}")

    # --- combinacion con la persistencia
    lineal = LinearRegression().fit(X_tr, y_tr)
    pred_lineal = lineal.predict(X_te)
    pred_lineal_val = lineal.predict(X_tr[m_val])
    persist_val = tr.loc[m_val, base_col].to_numpy()
    pred_comb, lam = candidato_combinacion(pred_lineal, base, y_tr[m_val],
                                           pred_lineal_val, persist_val)
    agregar("combinacion_lineal", pred_comb, f"lambda={lam:.2f}")
    agregar("promedio_simple", 0.5 * pred_lineal + 0.5 * base)

    # --- modelar el crecimiento en vez del nivel
    if target == "estimation":
        y_crec = np.log(y_tr / tr[base_col].to_numpy())
        modelo_crec = LinearRegression().fit(X_tr, y_crec)
        agregar("crecimiento_log", base * np.exp(modelo_crec.predict(X_te)))
    else:
        delta = y_tr - tr[base_col].to_numpy()
        modelo_delta = LinearRegression().fit(X_tr, delta)
        agregar("delta_tasa", base + modelo_delta.predict(X_te))

    # --- top-down: total nacional x participaciones
    total_tr = float(panel[panel["ANIO"] == test_year - 1][target].sum())
    total_hat = total_tr * crecimiento_nacional(total, test_year - 1) if target == "estimation" else 1.0
    share_persist = base / base.sum()
    agregar("jerarquico_shares_persistencia", total_hat * share_persist)

    # participaciones = y_t / total_t, modeladas en escala logit y renormalizadas.
    # Es la version correcta del enfoque jerarquico: modela el reparto, no el nivel.
    if target == "estimation":
        share_tr = y_tr / tr.groupby(level="ANIO")[target].transform("sum").to_numpy()
    else:
        share_tr = y_tr
    share_tr = np.clip(share_tr, 1e-6, 1 - 1e-6)
    modelo_share = LinearRegression().fit(X_tr, np.log(share_tr / (1 - share_tr)))
    logit_hat = modelo_share.predict(X_te)
    share_hat = 1 / (1 + np.exp(-logit_hat))
    share_hat = share_hat / share_hat.sum()
    agregar("jerarquico_shares_logit", total_hat * share_hat)

    # --- el mismo target de crecimiento aplicado a los arboles:
    #     si el arbol predice el crecimiento en vez del nivel, deja de estar
    #     obligado a subestimar (el crecimiento si cae dentro de su rango visto).
    if target == "estimation":
        y_crec = np.log(y_tr / tr[base_col].to_numpy())
        for nombre, modelo in (
            ("random_forest_crecimiento", RandomForestRegressor(**RF_PARAMS)),
            ("gradient_boosting_crecimiento", GradientBoostingRegressor(**GB_PARAMS)),
        ):
            modelo.fit(X_tr, y_crec)
            agregar(nombre, base * np.exp(modelo.predict(X_te)))

    # --- tabla
    filas = []
    for nombre, r in resultados.items():
        ic = bootstrap_skill(y_te, predicciones[nombre], base)
        filas.append({"modelo": nombre, **r,
                      "skill_ic95_bajo": ic[0], "skill_ic95_alto": ic[1]})
    tabla = pd.DataFrame(filas).sort_values("rmse")
    tabla.insert(0, "target", target)
    tabla.insert(1, "anio_test", test_year)
    tabla.insert(2, "n_train", len(tr))
    tabla.to_csv(OUT_DIR / f"mejoras_{target}_{test_year}.csv", index=False)

    # --- significancia contra persistencia y contra el modelo lineal
    print(f"\n=== {target} / test {test_year} ===")
    print(tabla[["modelo", "rmse", "mape", "skill_vs_persistencia", "pct_mse_rm",
                 "skill_ic95_bajo", "skill_ic95_alto", "nota"]]
          .to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    for nombre in predicciones:
        if nombre == "persistencia":
            continue
        _, p_pers = diebold_mariano(y_te, predicciones[nombre], base)
        _, p_lin = diebold_mariano(y_te, predicciones[nombre], pred_lineal)
        print(f"  DM {nombre:32s} vs persistencia p={p_pers:.3f} | vs lineal p={p_lin:.3f}")
    return tabla


def main() -> None:
    panel = pd.read_csv(PANEL_V2).merge(estimation_lag1(), on=["CODREGEO", "ANIO"], how="left")
    dc = pd.read_csv(COMBINADO)
    total = dc[dc["CODREGEO"] != 17].groupby("AÑO")["ESTIMACION"].sum()
    print("suma de rate por anio: " + ", ".join(
        f"{a}={panel[panel['ANIO'] == a]['rate'].sum():.4f}" for a in sorted(panel['ANIO'].unique())))
    print(f"Panel {sorted(int(a) for a in panel['ANIO'].unique())} | "
          f"total nacional: " + ", ".join(f"{a}={total[a]:,.0f}" for a in sorted(total.index)[-4:]))
    for target in ("estimation", "rate"):
        for test_year in (TEST_YEAR, 2022):
            evaluar(panel, total, target, test_year)
    print(f"\n-> {OUT_DIR}")


if __name__ == "__main__":
    main()
