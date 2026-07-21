"""Limpieza y codificacion de CensoData_normalizado.csv (version con codigos
INE de provincia/comuna) — misma estructura que modelo_combinado/limpiar_datos.py:
mapas explicitos a numerico + reporte de valores no mapeados.

Provincias: se validan y normalizan contra el codigo oficial territorial
(basura/Codigos.xlsx, hoja con 56 provincias) y se agrega provincia_nombre.

Convencion de codificacion (consistente con modelo_combinado y
modelo_censo_encoded):
   1/0  respuestas binarias (Si/No, Hombre/Mujer, Urbano/Rural)
   0    "No respuesta" / ignorado (la persona fue censada pero no respondio)
  -1    sin dato (la pregunta no aplicaba: salto del cuestionario)

Uso:  python src/censo_limpieza_v3.py
      -> data/processed/censo_normalizado_codificado.csv
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "CensoData_normalizado.csv"
CODIGOS = ROOT / "basura" / "Codigos.xlsx"
SALIDA = ROOT / "data" / "processed" / "censo_normalizado_codificado.csv"

RANGO_EDAD_MAP = {"0-17": 0, "18-30": 1, "31-45": 2, "46-60": 3, "60+": 4}

# Punto medio del periodo de llegada (mismo criterio que CensoData_encoded)
ANIO_LLEGADA_MAP = {"Entre 2020 y 2022": 2021.0, "Entre 2023 y 2024": 2023.5}

NACIONALIDAD_MAP = {
    "Argentina": 1, "Bolivia (Estado Plurinacional de)": 2, "Colombia": 4,
    "Perú": 6, "Venezuela (República Bolivariana de)": 7, "Haití": 15,
    "América del Norte": 21, "Otros países de América Central y El Caribe": 22,
    "Otros países de América del Sur": 23, "Europa": 24, "Asia": 25,
    "África": 26, "Oceanía": 27, "No respuesta": 0,
}  # 1-15 coinciden con PAIS_MAP de modelo_combinado; 21+ son agregados censales

SITUACION_LABORAL_MAP = {"Ocupado": 1, "Desocupado": 2,
                         "Fuera de la fuerza de trabajo": 3, "No respuesta": 0}

CATEGORIA_OCUPACIONAL_MAP = {
    "Dependiente": 1, "Independiente": 2,
    "Trabajador/a familiar o personal no remunerado en un negocio de un integrante de su familia": 3,
    "No respuesta": 0,
}

MEDIO_TRANSPORTE_MAP = {
    "Caminando": 1, "Bicicleta (incluye scooter)": 2, "Motocicleta": 3,
    "Auto particular": 4,
    "Transporte público (bus, micro, metro, tren, taxi, colectivo)": 5,
    "Caballo, lancha o bote": 6, "Otro": 7, "No respuesta": 0,
}

BINARIAS = {"pertenece_pueblo_indigena": ("Sí", "No"),
            "tiene_religion": ("Sí", "No")}


def codificar_mapa(df, col, mapa, nueva=None):
    """Aplica un mapa con la misma mecanica que limpiar_datos.py:
    reporta valores no mapeados y deja NaN -> -1 (sin dato)."""
    nueva = nueva or f"{col.upper()}_CODIGO"
    df[nueva] = df[col].map(mapa)
    no_mapeados = df.loc[df[nueva].isna() & df[col].notna(), col].unique()
    if len(no_mapeados) > 0:
        print(f"  Valores de {col} no reconocidos: {no_mapeados}")
    df[nueva] = df[nueva].fillna(-1)
    n_sin_dato = (df[nueva] == -1).sum()
    print(f"Columna {nueva} creada ({n_sin_dato:,} sin dato -> -1)")
    return df


def normalizar_provincias(df):
    """Valida provincia contra el codigo territorial oficial y agrega nombre."""
    cod = pd.read_excel(CODIGOS)
    prov = (cod[cod["División Política Administrativa"] == "Provincia"]
            .set_index("Código territorial")["Territorio"])
    print(f"Codigo oficial: {len(prov)} provincias (basura/Codigos.xlsx)")

    invalidas = set(df["provincia"].unique()) - set(prov.index)
    assert not invalidas, f"Provincias fuera del codigo oficial: {invalidas}"

    df["provincia_nombre"] = df["provincia"].map(prov)
    faltantes = set(prov.index) - set(df["provincia"].unique())
    print(f"Provincias del censo: {df['provincia'].nunique()} de {len(prov)} "
          f"(sin poblacion migrante censada en: "
          f"{sorted(prov.loc[list(faltantes)])})")
    return df


def limpiar():
    df = pd.read_csv(SOURCE)
    print(f"Filas leidas: {len(df):,}")

    # --- Provincias contra el codigo oficial ---
    df = normalizar_provincias(df)

    # --- Binarias (mismo criterio que limpiar_datos.py: H->1, M->0) ---
    df["sexo"] = df["sexo"].map({"Hombre": 1, "Mujer": 0})
    print("Columna sexo codificada: Hombre->1, Mujer->0")
    df["area"] = df["area"].map({"Urbano": 1, "Rural": 0})
    print("Columna area codificada: Urbano->1, Rural->0")
    for col, (si, no) in BINARIAS.items():
        df = codificar_mapa(df, col, {si: 1, no: 0, "No respuesta": 0},
                            nueva=col.upper())

    # --- Ordinales y derivadas ---
    df["RANGO_EDAD_CODIGO"] = df["rango_edad"].map(RANGO_EDAD_MAP)
    print("Columna RANGO_EDAD_CODIGO creada (ordinal 0-4)")
    df["ANIO_LLEGADA_ESTIMADO"] = df["año"].map(ANIO_LLEGADA_MAP)
    df["ANIOS_EN_CHILE"] = 2024 - df["ANIO_LLEGADA_ESTIMADO"]
    print("Columnas ANIO_LLEGADA_ESTIMADO y ANIOS_EN_CHILE creadas")

    # --- Nominales -> codigo ---
    df = codificar_mapa(df, "nacionalidad", NACIONALIDAD_MAP)
    df = codificar_mapa(df, "situacion_laboral", SITUACION_LABORAL_MAP)
    df = codificar_mapa(df, "categoria_ocupacional", CATEGORIA_OCUPACIONAL_MAP)
    df = codificar_mapa(df, "medio_transporte", MEDIO_TRANSPORTE_MAP)

    # nivel_educativo ya trae el codigo CINE como prefijo ("24: Educacion...")
    df["NIVEL_EDUCATIVO_CODIGO"] = (
        df["nivel_educativo"].str.extract(r"^(\d+):")[0].astype(float))
    df.loc[df["nivel_educativo"] == "No respuesta", "NIVEL_EDUCATIVO_CODIGO"] = 0
    df["NIVEL_EDUCATIVO_CODIGO"] = df["NIVEL_EDUCATIVO_CODIGO"].fillna(-1)
    print(f"Columna NIVEL_EDUCATIVO_CODIGO creada (prefijo CINE; "
          f"{(df['NIVEL_EDUCATIVO_CODIGO'] == -1).sum():,} sin dato -> -1)")

    # --- Numericas restantes ---
    df["escolaridad"] = pd.to_numeric(df["escolaridad"], errors="coerce").fillna(-1)

    # --- Checks finales ---
    codificadas = ["region", "provincia", "comuna", "area", "sexo", "edad",
                   "edad_quinquenal", "escolaridad", "llegada_reciente",
                   "PERTENECE_PUEBLO_INDIGENA", "TIENE_RELIGION",
                   "RANGO_EDAD_CODIGO", "ANIO_LLEGADA_ESTIMADO",
                   "ANIOS_EN_CHILE", "NACIONALIDAD_CODIGO",
                   "SITUACION_LABORAL_CODIGO", "CATEGORIA_OCUPACIONAL_CODIGO",
                   "MEDIO_TRANSPORTE_CODIGO", "NIVEL_EDUCATIVO_CODIGO"]
    assert df[codificadas].isna().sum().sum() == 0, "Quedaron nulos sin marcar"
    assert df["edad"].between(0, 110).all()
    print(f"\nFilas finales: {len(df):,} | columnas codificadas: {len(codificadas)}")
    return df


if __name__ == "__main__":
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    df = limpiar()
    df.to_csv(SALIDA, index=False)
    print(f"-> {SALIDA}")
