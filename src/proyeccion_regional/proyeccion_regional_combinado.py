"""Proyeccion regional 2024-2028 sin censo (solo dataset_combinado).

Metodo: tendencia log-lineal de ESTIMACION por region (regresion univariada
log(ESTIMACION) ~ ANIO, una por region), NO un modelo multivariado con un
R2 agregado. Motivo documentado en el informe final de TT I, seccion 6.6:
evitar un R2 dificil de defender y, de paso, no depender del censo (que solo
aporta reparto provincial, no region).

Anios de ajuste: 2020-2023 (4 puntos por region). El total nacional no se
modela aparte: emerge de sumar las 16 proyecciones regionales, asi que no
puede haber inconsistencia entre el total y la suma de las partes.

Uso:  python src/proyeccion_regional/proyeccion_regional_combinado.py
      -> outputs/tendencia_r2_por_region_combinado.csv
      -> outputs/proyeccion_regional_combinado_2024_2028.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parents[1]
sys.path.insert(0, str(ROOT / "src" / "modelo_combinado"))
from limpiar_datos import limpiar_datos  # noqa: E402

COMBINADO = ROOT / "data" / "raw" / "dataset_combinado.csv"
OUT_DIR = AQUI / "outputs"
ANIOS_AJUSTE = range(2020, 2024)  # 2020-2023
ANIOS_PROYECCION = range(2024, 2029)  # 2024-2028
REGION_IGNORADA = 17


def ajustar_region(sub: pd.DataFrame) -> pd.Series:
    """Regresion log-lineal de ESTIMACION vs ANIO para una region."""
    x = sub["AÑO"].to_numpy().reshape(-1, 1)
    y_log = np.log(sub["ESTIMACION"].to_numpy())
    modelo = LinearRegression().fit(x, y_log)
    r2 = r2_score(y_log, modelo.predict(x))
    cagr = float(np.exp(modelo.coef_[0]) - 1)
    return pd.Series({
        "intercepto": modelo.intercept_, "pendiente": modelo.coef_[0],
        "r2_tendencia": r2, "cagr_anual": cagr,
    })


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dc = pd.read_csv(COMBINADO)
    dc = limpiar_datos(dc)
    dc = dc[dc["CODREGEO"] != REGION_IGNORADA]

    panel = (dc[dc["AÑO"].isin(ANIOS_AJUSTE)]
               .groupby(["CODREGEO", "REGION", "AÑO"])["ESTIMACION"]
               .sum().reset_index())
    print(f"Panel de ajuste: {len(panel)} filas "
          f"({panel['CODREGEO'].nunique()} regiones x "
          f"{panel['AÑO'].nunique()} anios)")

    nombres = panel.drop_duplicates("CODREGEO").set_index("CODREGEO")["REGION"]
    ajuste = (panel.groupby("CODREGEO")
                   .apply(ajustar_region, include_groups=False)
                   .reset_index())
    ajuste["REGION"] = ajuste["CODREGEO"].map(nombres)

    peor = ajuste.loc[ajuste["r2_tendencia"].idxmin()]
    mejor = ajuste.loc[ajuste["r2_tendencia"].idxmax()]
    print(f"Peor ajuste: {peor['REGION']} R2={peor['r2_tendencia']:.3f} "
          f"CAGR={peor['cagr_anual']:.1%}")
    print(f"Mejor ajuste: {mejor['REGION']} R2={mejor['r2_tendencia']:.3f} "
          f"CAGR={mejor['cagr_anual']:.1%}")

    ajuste[["CODREGEO", "REGION", "r2_tendencia", "cagr_anual"]].to_csv(
        OUT_DIR / "tendencia_r2_por_region_combinado.csv", index=False)

    filas = []
    for _, row in ajuste.iterrows():
        for anio in ANIOS_PROYECCION:
            estim = float(np.exp(row["intercepto"] + row["pendiente"] * anio))
            filas.append({"CODREGEO": row["CODREGEO"], "REGION": row["REGION"],
                          "ANIO": anio, "estimacion_proyectada": round(estim)})
    proy = pd.DataFrame(filas)
    proy.to_csv(OUT_DIR / "proyeccion_regional_combinado_2024_2028.csv", index=False)

    total_2024 = proy.query("ANIO == 2024")["estimacion_proyectada"].sum()
    total_2028 = proy.query("ANIO == 2028")["estimacion_proyectada"].sum()
    print(f"\nTotal nacional (suma de regiones): "
          f"2024={total_2024:,.0f}  2028={total_2028:,.0f}  "
          f"crecimiento={(total_2028 / total_2024 - 1):.1%}")
    top = (proy.pivot(index="REGION", columns="ANIO", values="estimacion_proyectada")
              [[2024, 2028]].sort_values(2024, ascending=False).head(3))
    print("\nTop 3 regiones (2024 -> 2028):")
    print(top.to_string())

    print(f"\n-> {OUT_DIR / 'tendencia_r2_por_region_combinado.csv'}")
    print(f"-> {OUT_DIR / 'proyeccion_regional_combinado_2024_2028.csv'}")


if __name__ == "__main__":
    main()
