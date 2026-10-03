"""Proyeccion 2024-2028 con el MODELO de conteo (modo crecimiento).

Para que sirve
--------------
El mapa 3D hoy proyecta con una tendencia log-lineal (sin validacion). El objetivo
especifico 4 del anteproyecto pide que el mapa visualice "los resultados del modelo
predictivo". Este script genera esa proyeccion, para que el mapa pueda mostrar las
dos vistas: tendencia y modelo.

Como funciona
-------------
1. RECONSTRUYE las variables de 2023 (las que alimentan el pronostico de 2024)
   desde dataset_combinado.csv + el xlsx de solicitudes + la macro, replicando las
   formulas de src/modelo_1_deprecado/features.py y de
   src/modelo_2_regional_v2/build_dataset_v2.py.
   -> Se VALIDA: se reconstruyen tambien 2020-2022 y se comparan contra las
      columnas _lag1 del panel v2 (que son exactamente esos valores). Si no
      coinciden, el script falla.

2. REENTRENA el modelo en modo crecimiento con los 48 registros (2021-2023).
   El protocolo de evaluacion (entrenar hasta 2022 y probar en 2023) ya se hizo en
   train_conteo.py; para la proyeccion final lo estandar es reentrenar con todo.

3. PROYECTA recursivamente, region por region:
       y_{t+1} = y_t * exp(g_hat)
   donde g_hat es el crecimiento que predice el modelo. En cada paso se actualiza
   `estimation_lag1` con la propia prediccion (como corresponde a un pronostico
   recursivo) y el resto de las variables se mantiene en su valor de 2023.
   SUPUESTO DECLARADO: las condiciones (solicitudes, composicion, macro) se
   mantienen en los valores del ultimo anio observado.

Salidas (en outputs/)
---------------------
  proyeccion_modelo_2024_2028.csv     <- mismo formato que la tendencia (para el mapa)
  proyeccion_modelo_comparacion.csv   <- tendencia vs modelo, por region
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parents[1]
sys.path.insert(0, str(ROOT / "src" / "common"))
sys.path.insert(0, str(ROOT / "src" / "modelo_2_regional_v2"))
sys.path.insert(0, str(AQUI))
from config import SEED  # noqa: E402
from build_dataset_v2 import features_macro, features_solicitudes  # noqa: E402
# Se importan del propio modelo 3 para que no haya dos definiciones distintas:
# FEATURES es la lista exacta de variables y estimation_lag1() construye el rezago.
from train_conteo import FEATURES, estimation_lag1 as lag_conteo  # noqa: E402

PANEL_V2 = ROOT / "src" / "modelo_2_regional_v2" / "outputs" / "dataset_region_v2.csv"
COMBINADO = ROOT / "dataset_combinado.csv"
TENDENCIA = ROOT / "src" / "proyeccion_regional" / "outputs" / \
    "proyeccion_regional_combinado_2024_2028.csv"
HIPER = AQUI / "outputs" / "hiperparametros_conteo_crecimiento.json"
OUT_DIR = AQUI / "outputs"

# FEATURES se importa de train_conteo.py: misma lista, mismo orden que el modelo.
ANIOS_PROY = range(2024, 2029)

EDAD_MAP = {
    "00 A 04": 2, "05 A 09": 7, "10 A 14": 12, "15 A 19": 17, "20 A 24": 22,
    "25 A 29": 27, "30 A 34": 32, "35 A 39": 37, "40 A 44": 42, "45 A 49": 47,
    "50 A 54": 52, "55 A 59": 57, "60 A 64": 62, "65 A 69": 67, "70 A 74": 72,
    "75 A 79": 77, "80 O MÁS": 85, "IGNORADA": -1,
}


def features_demograficas() -> pd.DataFrame:
    """Replica features.py (modelo 1) sobre dataset_combinado.csv, todos los anios.

    Devuelve estimation, pct_women, mean_age, pct_irregular, pct_venezuela por
    region-anio. Se reconstruyen TODOS los anios a proposito: asi se puede
    validar contra las columnas _lag1 del panel v2.
    """
    dc = pd.read_csv(COMBINADO)
    dc = dc[dc["CODREGEO"] != 17].copy()          # sin region asignada, fuera
    dc = dc.rename(columns={"AÑO": "ANIO"})       # el CSV usa AÑO con tilde
    dc["SEXO_N"] = dc["SEXO"].map({"H": 1, "M": 0})
    dc["EDAD_N"] = dc["EDAD"].map(EDAD_MAP).fillna(-1)
    for c in ("ESTIMACION", "RRAA_IRREGULAR"):
        dc[c] = pd.to_numeric(dc[c], errors="coerce").fillna(0)

    g = dc.groupby(["CODREGEO", "ANIO"])
    out = pd.DataFrame({
        "estimation": g["ESTIMACION"].sum(),
        "rraa_irregular": g["RRAA_IRREGULAR"].sum(),
    }).reset_index()

    # % de mujeres (SEXO 0 = M, igual que limpieza_sermig.py)
    w = (dc[dc["SEXO_N"] == 0].groupby(["CODREGEO", "ANIO"])["ESTIMACION"].sum()
         .rename("women_est"))
    out = out.merge(w, on=["CODREGEO", "ANIO"], how="left")
    out["pct_women"] = out["women_est"] / out["estimation"]

    # edad media ponderada (excluye EDAD IGNORADA = -1)
    wa = dc[dc["EDAD_N"] != -1].copy()
    wa["wage"] = wa["EDAD_N"] * wa["ESTIMACION"]
    a = wa.groupby(["CODREGEO", "ANIO"]).agg(age_sum=("wage", "sum"),
                                             est_sum=("ESTIMACION", "sum"))
    out = out.merge((a["age_sum"] / a["est_sum"]).rename("mean_age"),
                    on=["CODREGEO", "ANIO"], how="left")

    out["pct_irregular"] = out["rraa_irregular"] / out["estimation"]

    ve = (dc[dc["PAIS"].str.upper().str.strip() == "VENEZUELA"]
          .groupby(["CODREGEO", "ANIO"])["ESTIMACION"].sum().rename("est_ve"))
    out = out.merge(ve, on=["CODREGEO", "ANIO"], how="left")
    out["pct_venezuela"] = out["est_ve"].fillna(0) / out["estimation"]

    return out.drop(columns=["women_est", "rraa_irregular", "est_ve"])


def validar_contra_panel(mias: pd.DataFrame) -> None:
    """Comprueba que las features reconstruidas coinciden con el panel v2.

    El panel del anio t tiene las features del anio t-1, asi que:
      panel[2021].<f>_lag1  debe ser igual a  mis[2020].<f>
      panel[2022].<f>_lag1  debe ser igual a  mis[2021].<f>
      panel[2023].<f>_lag1  debe ser igual a  mis[2022].<f>
    """
    panel = pd.read_csv(PANEL_V2)
    cols = ["pct_women", "mean_age", "pct_irregular", "pct_venezuela"]
    print("\nValidacion de la reconstruccion contra el panel v2:")
    for f in cols:
        difs = []
        for anio in (2021, 2022, 2023):
            mio = mias[mias["ANIO"] == anio - 1].set_index("CODREGEO")[f]
            pan = panel[panel["ANIO"] == anio].set_index("CODREGEO")[f"{f}_lag1"]
            difs.append((mio - pan).abs().max())
        peor = max(difs)
        estado = "OK" if peor < 1e-9 else "DIFERENCIA"
        print(f"  {f:16s} diferencia maxima {peor:.2e}  {estado}")
        assert peor < 1e-9, f"{f} no reproduce el panel v2"
    # estimation tambien debe calzar con la columna estimation del panel
    mio = mias.set_index(["CODREGEO", "ANIO"])["estimation"]
    pan = panel.set_index(["CODREGEO", "ANIO"])["estimation"]
    peor = (mio - pan).abs().max()
    print(f"  {'estimation':16s} diferencia maxima {peor:.2e}  "
          f"{'OK' if peor == 0 else 'DIFERENCIA'}")
    assert peor == 0


def condiciones(mias: pd.DataFrame, sol: pd.DataFrame, macro: pd.DataFrame,
                anios_cond, anio_nivel: int = 2023) -> pd.DataFrame:
    """Arma las variables que alimentan la proyeccion (una fila por region).

    El NIVEL (`estimation_lag1`) es siempre el observado en `anio_nivel`: es dato
    real y desde ahi arranca la recursion. Las demas variables son las
    "condiciones", y `anios_cond` decide de que anios se toman (promediadas).
    """
    demos = ["pct_women", "mean_age", "pct_irregular", "pct_venezuela"]
    dm = mias[mias["ANIO"].isin(anios_cond)].groupby("CODREGEO")[demos].mean()
    dm.columns = [f"{c}_lag1" for c in demos]
    dm["estimation_lag1"] = mias[mias["ANIO"] == anio_nivel].set_index("CODREGEO")["estimation"]

    s = (sol[sol["ANIO"].isin(anios_cond)].groupby("CODREGEO")[["sol_share", "sol_pct_otorga"]]
         .mean().rename(columns={"sol_share": "sol_share_lag1",
                                 "sol_pct_otorga": "sol_pct_otorga_lag1"}))
    m = (macro[macro["ANIO"].isin(anios_cond)].groupby("CODREGEO")[["macro_desempleo_origen"]]
         .mean().rename(columns={"macro_desempleo_origen": "macro_desempleo_origen_lag1"}))

    out = dm.join(s).join(m).reset_index()
    assert out[FEATURES].isna().sum().sum() == 0, "condiciones con nulos"
    return out[["CODREGEO"] + FEATURES]


def proyectar(modelo, filas: pd.DataFrame) -> pd.DataFrame:
    """Proyeccion recursiva 2024-2028 para las 16 regiones.

    El crecimiento se predice con las condiciones fijas; el nivel (que es una de
    las variables) se actualiza con la propia prediccion en cada paso.
    """
    registros = []
    for _, r in filas.iterrows():
        feats = r[FEATURES].astype(float).to_dict()
        y = float(feats["estimation_lag1"])          # nivel observado en 2023
        for anio in ANIOS_PROY:
            g = float(modelo.predict(pd.DataFrame([feats])[FEATURES])[0])
            y = y * np.exp(g)
            feats["estimation_lag1"] = y             # el rezago se actualiza
            registros.append({"CODREGEO": int(r["CODREGEO"]), "ANIO": anio,
                              "estimacion_proyectada": y})
    return pd.DataFrame(registros)


def modelo_rf():
    """El Random Forest del informe, con los hiperparametros ya elegidos."""
    import json
    params = {"n_estimators": 300, "max_depth": 3, "min_samples_leaf": 2}
    if HIPER.exists():
        mejores = json.loads(HIPER.read_text()).get("mejores_params", {})
        for clave in ("random_forest_conteo", "random_forest"):
            if mejores.get(clave):
                params = mejores[clave]
                break
    print(f"  random_forest con hiperparametros: {params}")
    return RandomForestRegressor(random_state=SEED, **params), params


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(PANEL_V2)
    # el rezago en personas lo construye el propio modelo 3 (misma funcion)
    panel = panel.merge(lag_conteo(), on=["CODREGEO", "ANIO"], how="left")
    assert panel["estimation_lag1"].notna().all(), "estimation_lag1 con nulos"

    # --- 1. variables reconstruidas + validacion ---
    mias = features_demograficas()
    validar_contra_panel(mias)
    sol, macro = features_solicitudes(), features_macro()

    # --- 2. reentrenar en modo crecimiento con los 48 registros ---
    p = panel.copy()
    p["crecimiento"] = np.log(p["estimation"] / p["estimation_lag1"])
    modelos = {
        "random_forest": modelo_rf()[0],
        "regresion_lineal": LinearRegression(),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=2,
            min_samples_leaf=2, random_state=SEED),
    }
    for nombre, m in modelos.items():
        m.fit(p[FEATURES], p["crecimiento"])
        sesgo = float((p["crecimiento"] - m.predict(p[FEATURES])).mean())
        print(f"  {nombre:18s} ajustado (sesgo medio {sesgo:+.4f})")

    # --- 3. proyectar bajo los tres supuestos de condiciones ---
    # El resultado depende mucho de cual se elija: por eso se guardan los tres.
    ESCENARIOS = {
        "promedio 2021-2023": [2021, 2022, 2023],
        "ultimo anio (2023)": [2023],
        "solo 2022": [2022],
    }
    real23 = panel[panel["ANIO"] == 2023].set_index("CODREGEO")["estimation"]
    proyecciones = {}          # (escenario, modelo) -> proyeccion
    for esc, anios in ESCENARIOS.items():
        filas_cond = condiciones(mias, sol, macro, anios)
        for nombre, m in modelos.items():
            proyecciones[(esc, nombre)] = proyectar(m, filas_cond)

    # --- salida principal: escenario recomendado (condiciones promedio) con el RF ---
    proy = proyecciones[("promedio 2021-2023", "random_forest")].merge(
        panel[["CODREGEO", "REGION"]].drop_duplicates(), on="CODREGEO", how="left")
    proy = proy[["CODREGEO", "REGION", "ANIO", "estimacion_proyectada"]]
    proy["estimacion_proyectada"] = proy["estimacion_proyectada"].round()
    proy.to_csv(OUT_DIR / "proyeccion_modelo_2024_2028.csv", index=False)

    # --- comparacion larga: 3 escenarios x 3 algoritmos + la tendencia ---
    tend = pd.read_csv(TENDENCIA)
    col_t = [c for c in tend.columns if "proyect" in c.lower() or "estim" in c.lower()][0]
    larga = []
    for (esc, nombre), pr in proyecciones.items():
        t = pr.groupby("ANIO")["estimacion_proyectada"].sum()
        for anio in ANIOS_PROY:
            larga.append({"escenario": esc, "modelo": nombre, "ANIO": anio,
                          "total_nacional": round(t[anio])})
    t_tend = tend.groupby("ANIO")[col_t].sum()
    for anio in ANIOS_PROY:
        larga.append({"escenario": "tendencia (actual)", "modelo": "log_lineal",
                      "ANIO": anio, "total_nacional": round(t_tend[anio])})
    pd.DataFrame(larga).to_csv(OUT_DIR / "proyeccion_modelo_comparacion.csv", index=False)

    # --- comparacion por region: escenario recomendado vs tendencia ---
    comp = proyecciones[("promedio 2021-2023", "random_forest")]
    comp = comp[comp["ANIO"] == 2028][["CODREGEO", "estimacion_proyectada"]].rename(
        columns={"estimacion_proyectada": "modelo_2028"})
    comp = comp.merge(tend[tend["ANIO"] == 2028][["CODREGEO", col_t]].rename(
        columns={col_t: "tendencia_2028"}), on="CODREGEO", how="left")
    comp = comp.merge(panel[panel["ANIO"] == 2023][["CODREGEO", "REGION", "estimation"]],
                      on="CODREGEO", how="left")
    comp["cagr_modelo"] = (comp["modelo_2028"] / comp["estimation"]) ** 0.2 - 1
    comp["cagr_tendencia"] = (comp["tendencia_2028"] / comp["estimation"]) ** 0.2 - 1
    comp = comp.sort_values("estimation", ascending=False)
    comp.to_csv(OUT_DIR / "proyeccion_modelo_por_region.csv", index=False)

    # --- resumen en pantalla ---
    print("\n== Totales nacionales al 2028 (suma de las 16 regiones) ==")
    for (esc, nombre), pr in proyecciones.items():
        t28 = pr[pr["ANIO"] == 2028]["estimacion_proyectada"].sum()
        print(f"  {esc:20s} {nombre:18s} {t28:>10,.0f}  ({t28/real23.sum()-1:+.1%} vs 2023)")
    print(f"  {'tendencia (actual)':39s} {t_tend[2028]:>10,.0f}  "
          f"({t_tend[2028]/real23.sum()-1:+.1%} vs 2023)")
    print(f"  {'2023 real':39s} {real23.sum():>10,.0f}")
    print("\n== 2028 por region: modelo (promedio, RF) vs tendencia ==")
    print(comp[["REGION", "estimation", "modelo_2028", "tendencia_2028",
                "cagr_modelo", "cagr_tendencia"]].to_string(
        index=False, formatters={"estimation": "{:>9,.0f}".format,
                                 "modelo_2028": "{:>10,.0f}".format,
                                 "tendencia_2028": "{:>12,.0f}".format,
                                 "cagr_modelo": "{:+.1%}".format,
                                 "cagr_tendencia": "{:+.1%}".format}))
    print(f"\n-> {OUT_DIR / 'proyeccion_modelo_2024_2028.csv'}")
    print(f"-> {OUT_DIR / 'proyeccion_modelo_comparacion.csv'}")


if __name__ == "__main__":
    main()
