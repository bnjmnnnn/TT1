from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parents[1]
CENSO = ROOT / "CensoData.csv"
COMBINADO = ROOT / "dataset_combinado.csv"
SALIDA = AQUI / "dataset_combinado_enriquecido.csv"

# Codigo de region INE (censo) -> CODREGEO (dataset_combinado, orden geografico)
REGION_INE_A_CODREGEO = {
    1: 2,   # Tarapaca
    2: 3,   # Antofagasta
    3: 4,   # Atacama
    4: 5,   # Coquimbo
    5: 6,   # Valparaiso
    6: 8,   # O'Higgins
    7: 9,   # Maule
    8: 11,  # Biobio
    9: 12,  # La Araucania
    10: 14,  # Los Lagos
    11: 15,  # Aysen
    12: 16,  # Magallanes
    13: 7,   # Metropolitana
    14: 13,  # Los Rios
    15: 1,   # Arica y Parinacota
    16: 10,  # Nuble
}

# p27_nacionalidad_esp (codigo INE de pais) -> pais censal homologado
PAIS_INE = {32: "ARGENTINA", 68: "BOLIVIA", 170: "COLOMBIA",
            332: "HAITI", 604: "PERU", 862: "VENEZUELA"}

# Paises que el censo distingue; el resto del dataset_combinado se compara
# contra el agregado "OTRO PAIS" del censo.
PAISES_CENSO = {"ARGENTINA": "ARGENTINA", "BOLIVIA": "BOLIVIA",
                "COLOMBIA": "COLOMBIA", "HAITÍ": "HAITI", "PERÚ": "PERU",
                "VENEZUELA": "VENEZUELA"}


def features_desde_censo() -> pd.DataFrame:
    censo = pd.read_csv(CENSO, usecols=[
        "region", "p27_nacionalidad_esp", "sexo", "edad", "escolaridad",
        "area", "llegada_reciente", "sit_fuerza_trabajo"])
    censo["codregeo"] = censo["region"].map(REGION_INE_A_CODREGEO)
    assert censo["codregeo"].notna().all()
    censo["pais_homologado"] = (censo["p27_nacionalidad_esp"]
                                .map(PAIS_INE).fillna("OTRO PAIS"))
    censo.loc[censo["p27_nacionalidad_esp"] == -99,
              "pais_homologado"] = "PAIS IGNORADO"
    censo["sexo_txt"] = censo["sexo"].map({1: "Hombre", 2: "Mujer"})
    censo["escolaridad"] = censo["escolaridad"].replace(-99, np.nan)

    def agregar(sub):
        laboral = sub[sub["sit_fuerza_trabajo"] != -99]["sit_fuerza_trabajo"]
        return pd.Series({
            "CENSO_N_GRUPO": len(sub),
            "CENSO_MEAN_EDAD": sub["edad"].mean(),
            "CENSO_MEAN_ESCOLARIDAD": sub["escolaridad"].mean(),
            "CENSO_PCT_URBANO": (sub["area"] == 1).mean(),
            "CENSO_PCT_LLEGADA_RECIENTE": sub["llegada_reciente"].mean(),
            "CENSO_PCT_OCUPADO": (laboral == 1).mean() if len(laboral) else pd.NA,
        })

    grupos = (censo.groupby(["codregeo", "pais_homologado", "sexo_txt"])
                   .apply(agregar, include_groups=False).reset_index())
    print(f"Grupos censales (region x pais x sexo): {len(grupos)}")
    return grupos


def enriquecer() -> pd.DataFrame:
    dc = pd.read_csv(COMBINADO)
    n0 = len(dc)

    dc["_pais"] = dc["PAIS"].map(PAISES_CENSO).fillna("OTRO PAIS")
    dc["_sexo"] = dc["SEXO"].map({"H": "Hombre", "M": "Mujer"})

    grupos = features_desde_censo()
    dc = dc.merge(grupos,
                  left_on=["CODREGEO", "_pais", "_sexo"],
                  right_on=["codregeo", "pais_homologado", "sexo_txt"],
                  how="left")

    cols_censo = [c for c in dc.columns if c.startswith("CENSO_")]
    sin_match = dc["CENSO_N_GRUPO"].isna().sum()
    print(f"Filas sin match directo: {sin_match:,} de {n0:,} "
          f"({sin_match / n0:.1%}) -> promedio nacional pais-sexo")
    nacional = grupos.groupby(["pais_homologado", "sexo_txt"])[cols_censo].mean()
    respaldo = (dc.set_index(["_pais", "_sexo"]).index
                  .map(lambda k: nacional.loc[k] if k in nacional.index else pd.Series()))
    for i, col in enumerate(cols_censo):
        dc[col] = dc[col].fillna(pd.Series([r.iloc[i] if len(r) else pd.NA
                                            for r in respaldo], index=dc.index))

    dc = dc.drop(columns=["codregeo", "pais_homologado", "sexo_txt",
                          "_pais", "_sexo"])
    assert len(dc) == n0, "El merge altero el numero de filas"
    assert dc[cols_censo].isna().sum().sum() == 0, "Quedaron features censales nulas"
    print(f"Dataset enriquecido: {len(dc):,} filas, "
          f"{len(cols_censo)} features censales nuevas")
    return dc


if __name__ == "__main__":
    dc = enriquecer()
    dc.to_csv(SALIDA, index=False)
    print(f"-> {SALIDA}")
