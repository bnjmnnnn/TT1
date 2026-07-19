"""Limpieza de dataset_combinado (codificacion a numerico + coercion),
misma estructura que la limpieza historica del equipo para este dataset.
"""
import pandas as pd

EDAD_MAP = {
    "00 A 04": 2, "05 A 09": 7, "10 A 14": 12, "15 A 19": 17,
    "20 A 24": 22, "25 A 29": 27, "30 A 34": 32, "35 A 39": 37,
    "40 A 44": 42, "45 A 49": 47, "50 A 54": 52, "55 A 59": 57,
    "60 A 64": 62, "65 A 69": 67, "70 A 74": 72, "75 A 79": 77,
    "80 O MÁS": 85, "IGNORADA": -1,
}

PAIS_MAP = {
    "ARGENTINA": 1, "BOLIVIA": 2, "BRASIL": 3, "COLOMBIA": 4, "ECUADOR": 5,
    "PERÚ": 6, "VENEZUELA": 7, "PARAGUAY": 8, "URUGUAY": 9,
    "ESTADOS UNIDOS": 10, "ESPAÑA": 11, "MÉXICO": 12, "CUBA": 13,
    "R. DOMINICANA": 14, "HAITÍ": 15, "CHINA": 16, "ALEMANIA": 17,
    "FRANCIA": 18, "ITALIA": 19, "OTRO PAÍS": 20, "PAÍS IGNORADO": 0,
}

COLUMNAS_NUMERICAS = ["CENSO AJUSTADO", "RRAA_REGULAR", "RRAA_IRREGULAR",
                      "RRAA_TOTAL", "ESTIMACION", "INFLACION",
                      "CRECIMIENTO_PIB", "DESEMPLEO"]


def limpiar_datos(df):
    if df.empty:
        print("El DataFrame esta vacio.")
        return df

    df["SEXO"] = df["SEXO"].map({"H": 1, "M": 0})
    print("Columna SEXO codificada: H->1, M->0")

    df["EDAD_NUMERICA"] = df["EDAD"].map(EDAD_MAP)
    no_mapeados = df[df["EDAD_NUMERICA"].isna()]["EDAD"].unique()
    if len(no_mapeados) > 0:
        print(f"Valores de EDAD no reconocidos: {no_mapeados}")
        df["EDAD_NUMERICA"] = df["EDAD_NUMERICA"].fillna(-1)
    print("Columna EDAD_NUMERICA creada")

    df["PAIS_CODIGO"] = df["PAIS"].map(PAIS_MAP)
    pais_no_mapeado = df[df["PAIS_CODIGO"].isna()]["PAIS"].unique()
    if len(pais_no_mapeado) > 0:
        print(f"Paises no reconocidos: {pais_no_mapeado}")
        df["PAIS_CODIGO"] = df["PAIS_CODIGO"].fillna(0)
    print("Columna PAIS_CODIGO creada")

    for col in COLUMNAS_NUMERICAS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    print("Columnas numericas convertidas y NaN rellenados con 0")

    sin_negativos = [c for c in COLUMNAS_NUMERICAS if c != "CRECIMIENTO_PIB"]
    negativos = (df[sin_negativos] < 0).any(axis=1)
    if negativos.any():
        print(f"Existen {negativos.sum()} filas con valores negativos -> valor absoluto")
        df[sin_negativos] = df[sin_negativos].abs()

    return df
