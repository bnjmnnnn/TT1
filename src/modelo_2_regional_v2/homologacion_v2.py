"""Homologacion de llaves entre fuentes (censo, SERMIG, dataset_combinado).

Todas las fuentes usan nombres/codigos distintos para region y nacionalidad.
Este modulo define UNA llave canonica por dimension:
  - Region  -> CODREGEO (orden geografico norte-sur, el usado en dataset_region.csv)
  - Pais    -> nombre de pais del dataset_combinado (mayusculas)
"""
import unicodedata

def normalizar(texto: str) -> str:
    """Llave de comparacion: sin tildes, mayusculas, espacios colapsados."""
    if texto is None:
        return ""
    t = unicodedata.normalize("NFKD", str(texto))
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    return " ".join(t.upper().split())

# CODREGEO segun dataset_region.csv / dataset_combinado.csv (orden geografico).
# Nota: NO es el codigo INE (Tarapaca=1 en INE, pero aqui Arica=1).
REGION_A_CODREGEO = {
    "ARICA Y PARINACOTA": 1,
    "TARAPACA": 2,
    "ANTOFAGASTA": 3,
    "ATACAMA": 4,
    "COQUIMBO": 5,
    "VALPARAISO": 6,
    "METROPOLITANA DE SANTIAGO": 7,
    "LIBERTADOR GENERAL BERNARDO O'HIGGINS": 8,
    "MAULE": 9,
    "NUBLE": 10,
    "BIOBIO": 11,
    "LA ARAUCANIA": 12,
    "LOS RIOS": 13,
    "LOS LAGOS": 14,
    "AYSEN DEL GENERAL CARLOS IBANEZ DEL CAMPO": 15,
    "MAGALLANES Y DE LA ANTARTICA CHILENA": 16,
}

def region_a_codregeo(nombre: str):
    """Devuelve CODREGEO o None si la region no es homologable
    (ej. 'REGION IGNORADA', 'Anonimizada', 'Sin Informacion')."""
    return REGION_A_CODREGEO.get(normalizar(nombre))

# Nacionalidad del censo (p27_nacionalidad_esp recodificada) -> PAIS de
# dataset_combinado. Las categorias continentales del censo no distinguen
# pais, por lo que van a OTRO PAIS (sin variables macro especificas).
NACIONALIDAD_CENSO_A_PAIS = {
    "ARGENTINA": "ARGENTINA",
    "BOLIVIA (ESTADO PLURINACIONAL DE)": "BOLIVIA",
    "COLOMBIA": "COLOMBIA",
    "HAITI": "HAITI",
    "PERU": "PERU",
    "VENEZUELA (REPUBLICA BOLIVARIANA DE)": "VENEZUELA",
    "AMERICA DEL NORTE": "OTRO PAIS",
    "ASIA": "OTRO PAIS",
    "EUROPA": "OTRO PAIS",
    "AFRICA": "OTRO PAIS",
    "OCEANIA": "OTRO PAIS",
    "OTROS PAISES DE AMERICA CENTRAL Y EL CARIBE": "OTRO PAIS",
    "OTROS PAISES DE AMERICA DEL SUR": "OTRO PAIS",
    "NO RESPUESTA": "PAIS IGNORADO",
}

def nacionalidad_a_pais(nacionalidad: str):
    return NACIONALIDAD_CENSO_A_PAIS.get(normalizar(nacionalidad), "PAIS IGNORADO")
