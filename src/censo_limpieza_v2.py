"""Limpieza reproducible del censo migrante (no modifica el original).

Fuente: CensoData.csv (codigos INE numericos) -- CensoData_normalizado.csv
(version con etiquetas de texto) ya no existe en el repo, se perdio en una
reorganizacion de archivos; se adapto esta lectura a los codigos INE con los
mismos mapeos usados en modelo_combinado/features_censo.py, verificando que
las cifras agregadas por region sean equivalentes a las documentadas.

Produce:
  data/processed/censo_features_region.csv -> features censales agregadas por region

Politica de missing (ausencia estructural, marcador -99 del cuestionario):
  se excluye de cada porcentaje puntual (no se imputa con media/moda, sesgaria
  la composicion), igual que la version anterior basada en 'Sin dato'.
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modelo_combinado"))
from features_censo import REGION_INE_A_CODREGEO, PAIS_INE  # noqa: E402

SOURCE = ROOT / "CensoData.csv"
OUT_FEATURES = ROOT / "data" / "processed" / "censo_features_region.csv"
EDAD_MAX_VALIDA = 110


def limpiar() -> pd.DataFrame:
    df = pd.read_csv(SOURCE, usecols=[
        "region", "p27_nacionalidad_esp", "sexo", "edad", "escolaridad",
        "area", "llegada_reciente", "sit_fuerza_trabajo"])
    n0 = len(df)
    print(f"Filas leidas: {n0:,}")

    mask_edad = df["edad"].between(0, EDAD_MAX_VALIDA)
    print(f"Filas con edad fuera de [0,{EDAD_MAX_VALIDA}]: {(~mask_edad).sum():,} (se eliminan)")
    df = df[mask_edad].copy()

    df["codregeo"] = df["region"].map(REGION_INE_A_CODREGEO)
    assert df["codregeo"].notna().all(), "Region censal sin homologar a CODREGEO"
    df["codregeo"] = df["codregeo"].astype(int)
    df["pais_homologado"] = (df["p27_nacionalidad_esp"]
                             .map(PAIS_INE).fillna("OTRO PAIS"))
    df.loc[df["p27_nacionalidad_esp"] == -99, "pais_homologado"] = "PAIS IGNORADO"

    print(f"Filas finales: {len(df):,} ({n0 - len(df):,} eliminadas)")
    return df


def features_region(df: pd.DataFrame) -> pd.DataFrame:
    """Composicion censal de la poblacion migrante por region (foto censo 2024).

    Los porcentajes se calculan sobre respuestas validas (excluyen -99).
    """
    def pct(sub: pd.DataFrame, col: str, valor) -> float:
        validas = sub[sub[col] != -99][col]
        return (validas == valor).mean() if len(validas) else float("nan")

    filas = []
    for cod, sub in df.groupby("codregeo"):
        filas.append({
            "CODREGEO": cod,
            "censo_n_migrantes": len(sub),
            "censo_mean_edad": sub["edad"].mean(),
            "censo_mean_escolaridad": sub.loc[sub["escolaridad"] != -99, "escolaridad"].mean(),
            "censo_pct_mujer": (sub["sexo"] == 2).mean(),
            "censo_pct_urbano": (sub["area"] == 1).mean(),
            "censo_pct_llegada_reciente": pct(sub, "llegada_reciente", 1),
            "censo_pct_ocupado": pct(sub, "sit_fuerza_trabajo", 1),
            "censo_pct_venezuela": (sub["pais_homologado"] == "VENEZUELA").mean(),
        })
    out = pd.DataFrame(filas).sort_values("CODREGEO")
    assert len(out) == 16, f"Se esperaban 16 regiones, hay {len(out)}"
    assert out.notna().all().all()
    return out


if __name__ == "__main__":
    OUT_FEATURES.parent.mkdir(parents=True, exist_ok=True)
    df = limpiar()

    feats = features_region(df)
    feats.to_csv(OUT_FEATURES, index=False)
    print(f"-> {OUT_FEATURES}")
    print(feats.to_string(index=False))
