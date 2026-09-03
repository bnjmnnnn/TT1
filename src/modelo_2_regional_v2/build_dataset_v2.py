"""Dataset v2: panel region-anio con las 3 fuentes del documento de TT1.

  1. SERMIG estimaciones  -> ../modelo_1_deprecado/outputs/dataset_region.csv
     (NO se regenera: es el insumo del modelo 1 y se lee tal cual para
     mantener comparabilidad)
  2. SERMIG solicitudes   -> RD-Resueltas-2o-semestre-2025.xlsx (2000-2025)
  3. Censo 2024           -> outputs/censo_features_region.csv
"""
from pathlib import Path
import pandas as pd

from homologacion_v2 import region_a_codregeo

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parents[1]
PANEL_M1 = ROOT / "src" / "modelo_1_deprecado" / "outputs" / "dataset_region.csv"
SOLICITUDES = ROOT / "RD-Resueltas-2o-semestre-2025.xlsx"
COMBINADO = ROOT / "dataset_combinado.csv"
CENSO_FEATS = AQUI / "outputs" / "censo_features_region.csv"
OUT = AQUI / "outputs" / "dataset_region_v2.csv"


def features_solicitudes() -> pd.DataFrame:
    """Solicitudes de residencia resueltas por region-anio.

    sol_share: fraccion de las solicitudes otorgadas del anio que corresponde
    a la region (misma naturaleza distributiva que el target `rate`).
    """
    rd = pd.read_excel(SOLICITUDES)
    rd.columns = ["sexo", "rango_etario", "pais", "actividad", "estudios",
                  "region", "anio", "tipo_resuelto", "total"]
    rd["total"] = pd.to_numeric(rd["total"], errors="coerce")
    n_bad = rd["total"].isna().sum()
    print(f"Solicitudes: {len(rd):,} filas, {n_bad} totales no numericos (se excluyen)")
    rd = rd.dropna(subset=["total"])

    rd["CODREGEO"] = rd["region"].map(region_a_codregeo)
    sin_region = rd["CODREGEO"].isna()
    print(f"Solicitudes sin region homologable (Anonimizada/Sin Informacion): "
          f"{rd.loc[sin_region, 'total'].sum():,.0f} de {rd['total'].sum():,.0f}")
    rd = rd.dropna(subset=["CODREGEO"])
    rd["CODREGEO"] = rd["CODREGEO"].astype(int)

    g = rd.groupby(["CODREGEO", "anio"])
    out = pd.DataFrame({
        "sol_total": g["total"].sum(),
        "sol_otorgadas": g.apply(
            lambda s: s.loc[s["tipo_resuelto"] == "Otorga", "total"].sum(),
            include_groups=False),
    }).reset_index()
    out["sol_pct_otorga"] = out["sol_otorgadas"] / out["sol_total"]
    out["sol_share"] = (out["sol_otorgadas"]
                        / out.groupby("anio")["sol_otorgadas"].transform("sum"))
    return out.rename(columns={"anio": "ANIO"})


def features_macro() -> pd.DataFrame:
    """Macro del pais de origen (inflacion, PIB, desempleo) promediadas por
    region-anio, ponderadas por el stock estimado de cada nacionalidad."""
    dc = pd.read_csv(COMBINADO)
    dc = dc[dc["CODREGEO"] != 17]  # REGION IGNORADA, fuera del panel
    dc = dc.dropna(subset=["ESTIMACION", "DESEMPLEO"])

    def wavg(sub: pd.DataFrame) -> pd.Series:
        w = sub["ESTIMACION"]
        return pd.Series({
            f"macro_{c.lower()}_origen": (sub[c] * w).sum() / w.sum()
            for c in ["INFLACION", "CRECIMIENTO_PIB", "DESEMPLEO"]
        })

    out = (dc.groupby(["CODREGEO", "AÑO"])
             .apply(wavg, include_groups=False).reset_index())
    return out.rename(columns={"AÑO": "ANIO"})


def construir() -> pd.DataFrame:
    panel = pd.read_csv(PANEL_M1)  # 48 filas, target y features del modelo 1
    n0, cols0 = len(panel), panel.shape[1]

    # Rezago t-1: el panel del anio t recibe las features del anio t-1
    sol = features_solicitudes()
    macro = features_macro()
    for feats in (sol, macro):
        lag = feats.copy()
        lag["ANIO"] = lag["ANIO"] + 1
        lag = lag.rename(columns={c: f"{c}_lag1" for c in lag.columns
                                  if c not in ("CODREGEO", "ANIO")})
        panel = panel.merge(lag, on=["CODREGEO", "ANIO"], how="left")

    censo = pd.read_csv(CENSO_FEATS)  # estaticas (foto 2024, candidatas)
    panel = panel.merge(censo, on="CODREGEO", how="left")

    # --- Checks (el script falla si no se cumplen) ---
    assert len(panel) == n0 == 48, "El merge altero el numero de filas del panel"
    nuevas = [c for c in panel.columns if c.startswith(("sol_", "macro_", "censo_"))]
    # Las 3 fuentes nuevas deben haber aportado columnas: solicitudes, macro y censo
    for prefijo in ("sol_", "macro_", "censo_"):
        assert any(c.startswith(prefijo) for c in nuevas), \
            f"El merge no aporto ninguna columna {prefijo}*"
    assert panel.shape[1] == cols0 + len(nuevas), "Columnas inesperadas tras el merge"
    nulos = panel[nuevas].isna().sum()
    assert (nulos == 0).all(), f"Features nuevas con nulos:\n{nulos[nulos > 0]}"
    # El panel del modelo 1 debe quedar identico en sus columnas originales
    assert panel[pd.read_csv(PANEL_M1).columns].equals(pd.read_csv(PANEL_M1))
    # Rezago verificado: sol_share de Valparaiso 2022 = share calculado en 2021
    v22 = panel.query("CODREGEO == 6 and ANIO == 2022")["sol_share_lag1"].iloc[0]
    v21 = sol.query("CODREGEO == 6 and ANIO == 2021")["sol_share"].iloc[0]
    assert abs(v22 - v21) < 1e-12, "Rezago sol_share mal construido"

    print(f"Panel v2: {len(panel)} filas, {panel.shape[1]} columnas "
          f"({len(nuevas)} features nuevas: {nuevas})")
    return panel


if __name__ == "__main__":
    panel = construir()
    panel.to_csv(OUT, index=False)
    print(f"-> {OUT}")
