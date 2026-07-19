# Fuentes de datos — TT1

Documenta el origen, formato y transformación de cada dataset usado en el
proyecto, para citar en el Capítulo 2 (Marco Teórico) y el Capítulo 5
(Implementación) del informe. Complementa la Capa 1 descrita en
`Avance_1_Trabajo_Titulo_I_2026.md` (Cap. 4.2).

## Resumen — 4 fuentes oficiales

| # | Fuente | Organismo | Nivel | Periodo | Archivo crudo |
|---|---|---|---|---|---|
| 1 | Microdatos Censo 2024 | INE | Individuo | Foto 2024 (declara periodo de llegada) | `personas_censo2024.csv` |
| 2 | Estimaciones de extranjeros residentes | SERMIG | Región × país × sexo × edad × año | 2018–2023 | `8. baseregiones.csv` |
| 3 | Solicitudes de residencia resueltas | SERMIG | Región × país × sexo × rango etario × año | 2000–2025 | `RD-Resueltas-2o-semestre-2025.xlsx` |
| 4 | Indicadores macroeconómicos por país de origen | Combinados en `dataset_combinado.csv` (INFLACION, CRECIMIENTO_PIB, DESEMPLEO) | País × año | 2018–2023 | *(ver nota de trazabilidad más abajo)* |

---

## 1. Microdatos Censo 2024 (INE)

- **Origen:** Instituto Nacional de Estadísticas de Chile. Descarga pública:
  https://censo2024.ine.gob.cl/resultados/
- **Archivo crudo:** `personas_censo2024.csv` — 18.480.432 registros × 63
  variables (~2.400 MB), delimitador `;`, codificación UTF-8. No se versiona
  en git por tamaño (`.gitignore`).
- **Diccionario de variables:** `diccionario_variables_censo2024 (2).xlsx`
  (INE) — define cada variable, sus categorías y universo de respuesta;
  hojas `tabla_viviendas`, `tabla_hogares`, `tabla_personas`,
  `cod_territoriales_especificos`, `codigos_territoriales`.
- **Códigos territoriales y de país:** `Codigos.xlsx` (INE/BCN) — 419 filas:
  códigos de país/continente de origen (columna "Código específico") y
  códigos de Región/Provincia/Comuna con nombre oficial (56 provincias).
  Usado para validar y homologar `provincia` en `src/censo_limpieza_v3.py`.

### Transformación

```
personas_censo2024.csv (18,48M filas, 63 cols)
        │  src/etl_censo.py
        │  Filtro: p27_nacionalidad==3 AND p27_nacionalidad_rec==2
        │          AND p26_llegada_periodo in (1,2)
        │          → población nacida en el extranjero, nacionalidad no
        │            chilena, con periodo de llegada declarado
        │  Columnas: 63 → 20 (pertinencia analítica)
        │  Missing -99/-66 → vacío
        ▼
CensoData.csv (540.291 filas, 20 cols, códigos numéricos)
        │  normalización de categorías (códigos → etiquetas legibles:
        │  "8" → "Biobío", "3" → "Venezuela (República Bolivariana de)", etc.)
        ▼
CensoData_normalizado.csv (540.291 filas, 19 cols)
        │  src/censo_limpieza_v3.py
        │  - provincia validada contra Codigos.xlsx (56 provincias oficiales)
        │  - variables codificadas a numérico (convención unificada:
        │    1/0 binarias, 0 "No respuesta", -1 "sin dato"/no aplica)
        ▼
data/processed/censo_normalizado_codificado.csv (540.291 filas, listo para modelar)
```

> **Nota:** el paso `CensoData.csv → CensoData_normalizado.csv` (traducción
> de códigos a etiquetas) no tiene script propio en este repo — se
> desconoce el mecanismo exacto de generación. Si el archivo se generó con
> otra herramienta o notebook, documentarlo aquí antes de la entrega final.

### Reducción y validación (citar en Cap. 5.3)

18.480.432 → 540.291 filas (reducción 97,08%), validado en su momento con
checksum MD5, conteo de filas y muestreo aleatorio (ver
`Avance_1_Trabajo_Titulo_I_2026.md`, §5.3).

---

## 2. Estimaciones de extranjeros residentes (SERMIG)

- **Origen:** Servicio Nacional de Migraciones.
  https://serviciomigraciones.cl/estudios-migratorios/estimaciones-de-extranjeros/
- **Archivo crudo:** `8. baseregiones.csv` — stock estimado de población
  extranjera por región, con desagregación adicional (`CENSO AJUSTADO`,
  `RRAA_REGULAR/IRREGULAR/TOTAL`, `ESTIMACION`), 2018–2023.

### Transformación

```
8. baseregiones.csv
        │  limpieza_sermig.py
        │  SEXO H/M → 1/0 · EDAD (rango quinquenal) → punto medio
        │  PAIS → PAIS_CODIGO + PAIS_ISO (mapeo manual a 21 categorías)
        │  columnas numéricas → coerción + NaN→0 · negativos → abs()
        │  (excepto CRECIMIENTO_PIB, puede ser negativo por recesión)
        ▼
8. baseregiones_limpio.csv
        │  src/build_datasets.py
        │  target: rate_{r,t} = ESTIMACION_{r,t} / Σ_r ESTIMACION_{r,t}
        │  (share regional del stock nacional regionalizado)
        │  filtro año ≥ 2020 · features rezagadas a t-1 (anti-leakage)
        ▼
data/processed/dataset_region.csv (48 filas: 16 regiones × 2021-2023)
```

Ver `docs/decisiones_modelo_regional.md` para el detalle de por qué se
excluye la Región 17 ("Región Ignorada") y la regla de rezago.

---

## 3. Solicitudes de residencia resueltas (SERMIG)

- **Origen:** Servicio Nacional de Migraciones — registros administrativos
  de solicitudes de regularización migratoria.
- **Archivo:** `RD-Resueltas-2o-semestre-2025.xlsx` — 54.200 filas,
  columnas `SEXO, RANGO_ETARIO, PAÍS, ACTIVIDAD, ESTUDIOS, REGIÓN, AÑO,
  TIPO_RESUELTO, Total`. Pese al nombre del archivo (alude a la fecha de
  *publicación*, 2° semestre 2025), la columna `AÑO` cubre **2000–2025**.
  `TIPO_RESUELTO` ∈ {Otorga, Archiva, Rechaza con Rt, Rechaza con abandono}.

### Transformación

```
RD-Resueltas-2o-semestre-2025.xlsx
        │  src/build_dataset_v2.py → features_solicitudes()
        │  Total → numérico (172 filas no numéricas excluidas)
        │  REGIÓN → CODREGEO (src/homologacion_v2.py; "Anonimizada" y
        │  "Sin Información" sin homologación, ~1,8% del total, se excluyen)
        │  agregación región-año: sol_total, sol_otorgadas, sol_pct_otorga,
        │  sol_share (participación regional en las solicitudes otorgadas
        │  del año — misma naturaleza distributiva que el target `rate`)
        ▼
features rezagadas a t-1, unidas al panel región-año
```

---

## 4. Indicadores macroeconómicos y `dataset_combinado.csv`

`dataset_combinado.csv` (62.860 filas: 21 países × 17 CODREGEO × sexo ×
banda etaria quinquenal × 2018–2023) integra, por fila:

| Columna | Contenido |
|---|---|
| `CENSO AJUSTADO`, `RRAA_REGULAR/IRREGULAR/TOTAL`, `ESTIMACION` | Estimaciones SERMIG (misma familia que la fuente 2) |
| `INFLACION`, `CRECIMIENTO_PIB`, `DESEMPLEO` | Indicadores macroeconómicos **del país de origen del migrante**, por año |
| `AÑO ESTIMACION`, `AÑO`, `CODREGEO`, `REGION`, `CODIGO_PAIS` | Llaves de unión |

> **Nota de trazabilidad — pendiente de confirmar:** este archivo llegó ya
> consolidado; no existe en el repo un script que documente la fuente
> exacta de `INFLACION`/`CRECIMIENTO_PIB`/`DESEMPLEO` (candidatos típicos:
> Banco Mundial — World Development Indicators, o FMI — World Economic
> Outlook, por país y año). **Confirmar con Bastián el origen puntual y la
> fecha de descarga antes de citarlo en el informe.**

### Transformación (`modelo_combinado/`)

```
dataset_combinado.csv
        │  modelo_combinado/limpiar_datos.py
        │  SEXO H/M→1/0 · EDAD (banda)→punto medio · PAIS→código (20 cat.)
        │  numéricas: coerción + NaN→0 · negativos→abs (excepto PIB)
        │
        │  modelo_combinado/features_censo.py
        │  + 5 features censales (fuente 1) agregadas por
        │    región × país × sexo (93,6% match directo; resto = promedio
        │    nacional país-sexo): edad media, escolaridad media,
        │    % urbano, % llegada reciente, % ocupado
        ▼
modelo_combinado/dataset_combinado_enriquecido.csv
```

---

## Mapa de dependencias (qué script usa qué fuente)

| Script | Fuentes 1 (Censo) | 2 (Estim. SERMIG) | 3 (Solicitudes) | 4 (Macro) |
|---|---|---|---|---|
| `src/build_datasets.py` → `dataset_region.csv` | | ✔ | | |
| `src/build_dataset_v2.py` → `dataset_region_v2.csv` | ✔ (agregado) | ✔ | ✔ | ✔ |
| `src/proyeccion_provincial.py` | ✔ (share provincial) | ✔ (vía modelo v2) | ✔ (vía modelo v2) | ✔ (vía modelo v2) |
| `modelo_censo_encoded/`, `modelo_censo_provincia/` | ✔ | | | |
| `modelo_combinado/` | ✔ (enriquecimiento) | | | ✔ |

## Archivos crudos vs. procesados (para el informe: qué es dato oficial y qué es transformación propia)

- **Dato oficial, sin modificar:** `personas_censo2024.csv`, `8. baseregiones.csv`,
  `RD-Resueltas-2o-semestre-2025.xlsx`, `Codigos.xlsx`,
  `diccionario_variables_censo2024 (2).xlsx`, `dataset_combinado.csv`
  (salvo confirmar nota de trazabilidad).
- **Transformación propia (reproducible con los scripts de `src/`):** todo
  lo demás — `CensoData*.csv`, `8. baseregiones_limpio.csv`,
  `data/processed/*.csv`, `data/outputs/**`, `modelo_combinado/*enriquecido*`.
