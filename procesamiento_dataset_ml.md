# Procesamiento del Dataset `dataset_ml_raw.csv`

**Proyecto:** Modelo Predictivo de Distribución Territorial de Inmigrantes en Chile  
**Fecha:** 18 de julio de 2026  
**Nivel de análisis:** Regional (16 regiones de Chile)  
**Nivel de desagregación:** SEXO × EDAD × PAÍS × REGIÓN × AÑO  

---

## 1. Contexto y Objetivo

El trabajo de título requiere predecir la distribución territorial de inmigrantes en Chile a nivel regional. El dataset principal (`dataset_combinado.csv`) proviene del SERMIG y contiene estimaciones de población inmigrante residente, desagregadas por sexo, edad, país de origen, región y año (2018-2023).

Para enriquecer el modelo, se integraron dos fuentes adicionales del SERMIG:

- **RD-Acogidas-2o-semestre-2025.xlsx**: Solicitudes de residencia admitidas a trámite.
- **RD-Resueltas-2o-semestre-2025.xlsx**: Solicitudes de residencia con resolución definida (Otorga, Rechaza, Archiva).

El desafío fue **mantener la granularidad original** (~62.860 filas del dataset combinado) sin colapsar la información a un nivel más agregado, ya que perder la desagregación por sexo y edad reduciría drásticamente la cantidad de datos disponibles para entrenar.

---

## 2. Fuentes de Datos

| Fuente | Formato | Filas originales | Granularidad | Periodo |
|---|---|---|---|---|
| `dataset_combinado.csv` | CSV | 62.860 | SEXO × EDAD × PAÍS × REGIÓN × AÑO | 2018-2023 |
| `RD-Acogidas-2o-semestre-2025.xlsx` | Excel | 44.984 | SEXO × EDAD × ACTIVIDAD × ESTUDIOS × PAÍS × REGIÓN × AÑO | 2000-2025 |
| `RD-Resueltas-2o-semestre-2025.xlsx` | Excel | 54.200 | Idem + TIPO_RESUELTO | 2000-2025 |

**Variable objetivo (target):** `ESTIMACION` — cantidad estimada de extranjeros residentes en Chile, según el SERMIG.

---

## 3. Problema Detectado en el Primer Intento

En un primer análisis exploratorio se realizó un merge por `REGIÓN + AÑO + PAÍS`, agregando previamente el dataset combinado (sumando `ESTIMACION` por grupo). Esto redujo el dataset a **1.824 filas**, perdiendo la desagregación por sexo y edad y dejando muy pocos datos para modelar.

**Decisión:** descartar ese enfoque y rehacer el procesamiento conservando la granularidad original.

---

## 4. Procesamiento Paso a Paso

### 4.1 Carga

Los tres archivos fueron cargados con `pandas`:
- `dataset_combinado.csv` vía `pd.read_csv()`.
- Archivos Excel vía `pd.read_excel()`.

### 4.2 Limpieza de Nombres

Los archivos Excel y el CSV usaban convenciones de nombre diferentes (mayúsculas vs. Title Case, abreviaturas vs. nombres completos, tildes codificadas de distinta forma). Esto impedía cualquier merge directo.

**Estrategia aplicada:**

```python
def norm_text(s):
    if pd.isna(s):
        return s
    return str(s).strip().lower()
```

Todas las columnas de texto (región, país, sexo, edad) fueron normalizadas con:
- `strip()`: eliminar espacios al inicio/final.
- `lower()`: convertir a minúsculas para case-insensitive matching.

**Regiones geocodificables retenidas:** 16 regiones oficiales de Chile.  
**Excluidas:** `"región ignorada"`, `"sin información"`, `"anonimizada"`.

**Países excluidos:** `"país ignorado"`, `"otro país"`, `"otros países"`, `"otros países dentro de los 25 primeros"`. Estas categorías agregadas no tienen variables macroeconómicas asociadas.

### 4.3 Mapeo de Países

Algunos países tenían nombres distintos entre el CSV y los Excel:

| En `dataset_combinado.csv` | En RD-Excel |
|---|---|
| `r. dominicana` | `república dominicana` |
| `estados unidos` | `estados unidos` |

Se aplicó un diccionario de reemplazo en el CSV para homologar los nombres antes del merge.

### 4.4 Conversión de Tipos

La columna `Total` en ambos archivos Excel llegó como tipo `object` (texto) en lugar de numérico. Se aplicó:

```python
pd.to_numeric(df['Total'].astype(str).str.replace(',', ''), errors='coerce')
```

Esto eliminó comas separadoras de miles y forzó la conversión a `float`, asignando `NaN` a valores no convertibles.

### 4.5 Agregación de Solicitudes

Los archivos de solicitudes estaban desagregados por `SEXO × EDAD × ACTIVIDAD × ESTUDIOS`. Para integrarlos al dataset combinado (que ya tiene su propia desagregación sexo-edad), las solicitudes se **agregaron a nivel `PAÍS + REGIÓN + AÑO`**.

**Lógica:** las solicitudes de residencia son un **indicador de flujo migratorio formal** que afecta a toda la población de un país en una región, independientemente del sexo o edad específico de cada subgrupo censal. Por tanto, el mismo total de solicitudes se asigna a todas las filas sexo-edad de ese país-región-año.

**Operaciones realizadas:**

**Acogidas:**
```python
acog_agg = acogidas.groupby(['pais_norm', 'region_norm', 'anio'], as_index=False).agg({'Total': 'sum'})
acog_agg = acog_agg.rename(columns={'Total': 'ACOGIDAS_TOTAL'})
```

**Resueltas:**
```python
res_pivot = resueltas.pivot_table(
    index=['pais_norm', 'region_norm', 'anio'],
    columns='TIPO_RESUELTO',
    values='Total',
    aggfunc='sum',
    fill_value=0
).reset_index()
```

Se calcularon las siguientes columnas derivadas de resoluciones:

| Columna | Fórmula |
|---|---|
| `RESUELTAS_TOTAL` | `Otorga + Rechaza con Rt + Rechaza con abandono + Archiva` |
| `TASA_OTORGA` | `Otorga / RESUELTAS_TOTAL` |
| `TASA_RECHAZO` | `(Rechaza con Rt + Rechaza con abandono) / RESUELTAS_TOTAL` |
| `TASA_ARCHIVO` | `Archiva / RESUELTAS_TOTAL` |

### 4.6 Merge (Left Join)

Se realizó un `merge(..., how='left')` desde `dataset_combinado.csv` hacia las tablas agregadas de solicitudes, usando la clave `PAÍS_norm + REGIÓN_norm + AÑO`.

```python
merged = df.merge(acog_agg, on=['pais_norm_merge', 'region_norm', 'anio'], how='left')
merged = merged.merge(res_pivot, on=['pais_norm_merge', 'region_norm', 'anio'], how='left')
```

**Resultado del merge:**
- Filas del dataset original conservadas: **58.893** (de 62.860 iniciales; se perdieron 3.967 filas de `"región ignorada"` y `"país ignorado"`).
- Columnas añadidas: 15 (totales de acogidas, resueltas, desagregación por tipo, tasas).

### 4.7 Features Derivadas

Se crearon variables adicionales para capturar dinámicas temporales y estructurales:

| Feature | Descripción | Cálculo |
|---|---|---|
| `ACOGIDAS_LAG1` | Acogidas del año anterior (mismo país-región) | `groupby(['pais','region']).shift(1)` |
| `RESUELTAS_LAG1` | Resoluciones totales del año anterior | `groupby(['pais','region']).shift(1)` |
| `TASA_REGULARIZACION` | Proporción de regularizados sobre el stock estimado | `RRAA_TOTAL / (ESTIMACION + 1e-9)` |
| `PROP_ESTIMACION_GRUPO` | Proporción del subgrupo sexo-edad dentro del total país-región-año | `ESTIMACION / sum(ESTIMACION por país-región-año)` |
| `ACOGIDAS_PER_CAPITA` | Solicitudes acogidas por cada inmigrante estimado | `ACOGIDAS_TOTAL / (ESTIMACION + 1e-9)` |
| `RESUELTAS_PER_CAPITA` | Resoluciones por cada inmigrante estimado | `RESUELTAS_TOTAL / (ESTIMACION + 1e-9)` |

**Tratamiento de infinitos:** valores `inf` generados por división por cero fueron reemplazados por `NaN`.

### 4.8 Eliminación de Data Leakage

Durante el EDA se detectó que `ESTIMACION_LAG1` (estimación del año anterior para el mismo subgrupo) tenía una correlación de **0.988** con el target. Esto constituye **data leakage**: es prácticamente el target del período previo.

**Decisión:** eliminar `ESTIMACION_LAG1` y `ESTIMACION_PCT_CHANGE` (que depende del lag) antes de entrenar cualquier modelo. El modelo debe predecir el stock futuro sin conocer el stock pasado del mismo subgrupo.

Las features de rezago de **flujo** (`ACOGIDAS_LAG1`, `RESUELTAS_LAG1`) sí se mantienen, ya que representan solicitudes administrativas del año anterior, no el target mismo.

---

## 5. Dataset Final

**Archivo:** `dataset_ml_raw.csv`  
**Ubicación:** raíz del proyecto (`TT1/dataset_ml_raw.csv`)

| Métrica | Valor |
|---|---|
| Filas | 58.893 |
| Columnas | 37 |
| Periodo | 2018-2023 |
| Regiones | 16 |
| Países | 21 |
| Sexos | 2 (H/M) |
| Rangos etarios | 18 |
| Target | `ESTIMACION` (media=153, mediana=10, skewness=19.67) |

### Columnas principales

**Del dataset combinado original:**
- `SEXO`, `EDAD`, `PAIS`, `REGION`, `ANIO`
- `ESTIMACION` (target)
- `RRAA_REGULAR`, `RRAA_IRREGULAR`, `RRAA_TOTAL`
- `CENSO AJUSTADO`
- `INFLACION`, `CRECIMIENTO_PIB`, `DESEMPLEO` (macro del país de origen)

**Integradas desde RD-Acogidas:**
- `ACOGIDAS_TOTAL`
- `ACOGIDAS_LAG1`
- `ACOGIDAS_PER_CAPITA`

**Integradas desde RD-Resueltas:**
- `RESUELTAS_TOTAL`, `RESUELTAS_OTORGA`, `RESUELTAS_RECHAZA_RT`, `RESUELTAS_RECHAZA_ABANDONO`, `RESUELTAS_ARCHIVA`
- `TASA_OTORGA`, `TASA_RECHAZO`, `TASA_ARCHIVO`
- `RESUELTAS_LAG1`, `RESUELTAS_PER_CAPITA`

**Derivadas:**
- `TASA_REGULARIZACION`
- `PROP_ESTIMACION_GRUPO`

### Missings estratégicos

| Variable | % Missing | Estrategia recomendada |
|---|---|---|
| `ACOGIDAS_TOTAL` | 68.4% | Imputar con **0** (no todas las combinaciones país-región-año reciben solicitudes) |
| `RESUELTAS_TOTAL` | 72.6% | Imputar con **0** |
| `INFLACION` | 24.4% | Imputar con **mediana por país** |
| `CRECIMIENTO_PIB` | 19.6% | Imputar con **mediana por país** |
| `DESEMPLEO` | 14.0% | Imputar con **mediana por país** |
| `RRAA_IRREGULAR` | 65.8% | Considerar excluir o imputar con 0 |
| `CENSO AJUSTADO` | 7.0% | Forward-fill o imputar con mediana por grupo |

---

## 6. Correlaciones Relevantes (Top 10 con el Target)

Sin incluir `ESTIMACION_LAG1` (eliminado por data leakage):

| Feature | Correlación |
|---|---|
| `RRAA_REGULAR` | 0.920 |
| `RRAA_TOTAL` | 0.918 |
| `CENSO AJUSTADO` | 0.846 |
| `RESUELTAS_TOTAL` | 0.584 |
| `ACOGIDAS_TOTAL` | 0.584 |
| `RESUELTAS_OTORGA` | 0.576 |
| `RESUELTAS_LAG1` | 0.562 |
| `ACOGIDAS_LAG1` | 0.528 |
| `RRAA_IRREGULAR` | 0.471 |
| `RESUELTAS_RECHAZA_RT` | 0.409 |

**Observación:** las variables de flujo formal (solicitudes) tienen correlación positiva y moderada con el stock migratorio, validando su utilidad como features. Las variables macro (inflación, PIB, desempleo) son débiles individualmente (r < 0.05) pero pueden aportar en modelos no lineales (Random Forest, XGBoost) al capturar interacciones.

---

## 7. Recomendaciones para el Modelado

1. **Transformación del target:** usar `np.log1p(ESTIMACION)` debido al alto sesgo (skewness=19.67).
2. **Imputación:** 0 para solicitudes sin datos; mediana por país para variables macro.
3. **Codificación categórica:** One-Hot Encoding para `SEXO`, `EDAD`, `PAIS`, `REGION`.
4. **Modelos a entrenar:** Regresión Lineal Múltiple (baseline), Random Forest Regressor, Gradient Boosting Regressor (o XGBoost/LightGBM).
5. **Validación:** Hold-out temporal (último año, 2023, como test) para evitar leakage temporal.
6. **Métricas:** RMSE, MAE, R². Objetivo del TT: R² ≥ 0.70 sobre el conjunto de prueba.
7. **No usar:** `ESTIMACION_LAG1`, `ESTIMACION_PCT_CHANGE`, ni cualquier variable que sea el target rezagado.

---

## 8. Scripts Relacionados

| Script | Propósito |
|---|---|
| `eda_v2_granular.py` | Script del EDA completo que generó `dataset_ml_raw.csv` |
| `src/build_dataset_v2.py` | Pipeline alternativo de construcción del dataset |
| `src/train_v2.py` | Entrenamiento de modelos v2 |
| `src/validacion_v2.py` | Validación de modelos v2 |

---

## 9. Decisiones de Diseño Clave

| Decisión | Justificación |
|---|---|
| Mantener granularidad sexo-edad | El dataset combinado tiene ~62k filas. Agregar a país-región-año lo reduciría a ~1.800 filas, insuficiente para modelos de ML robustos. |
| Asignar solicitudes totales a todos los subgrupos sexo-edad | Las solicitudes son un indicador de flujo a nivel país-región. No hay desagregación sexo-edad en los archivos RD, pero el flujo total afecta a todos los subgrupos. |
| Eliminar ESTIMACION_LAG1 | Data leakage severo (r=0.988). El modelo aprendería a copiar el valor anterior en vez de predecir. |
| Excluir regiones/países ignorados | No tienen ubicación geográfica ni variables macro, por tanto no son predictibles ni geocodificables. |
| Conservar missings en solicitudes como NaN (luego 0) | La ausencia de solicitudes en una combinación país-región-año es informativa: significa flujo formal nulo. |

---

*Documento generado automáticamente tras el procesamiento ETL del 18 de julio de 2026.*
