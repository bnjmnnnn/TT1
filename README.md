# Modelo predictivo de distribución territorial de inmigrantes en Chile

**Trabajo de Título · Universidad Tecnológica Metropolitana**
Bastián Pizarro Pacheco · Benjamín Fernández Toledo

---

## El problema

Chile pasó de ser un país de emigración a uno de recepción en poco más de una
década: hoy viven más de 1,9 millones de personas extranjeras en el territorio.
Ese crecimiento no se repartió de forma pareja. La Región Metropolitana
concentra alrededor del 60 % del total, mientras regiones como Aysén o Los Ríos
no llegan al 1 % cada una.

Esa asimetría es un problema de planificación concreto: los gobiernos
regionales necesitan anticipar cuánta población van a atender para dimensionar
salud, educación y vivienda. Hoy esa anticipación se hace mirando cifras del
pasado, sin un instrumento que proyecte hacia adelante.

## Qué hace este proyecto

Construye un modelo de aprendizaje automático que predice **cómo se reparte la
población inmigrante entre las 16 regiones del país**, usando únicamente
información disponible el año anterior.

La variable que se predice es la *tasa de concentración regional*:

```
rate(r,t) = ESTIMACION(r,t) / Σ_r ESTIMACION(r,t)
```

es decir, qué fracción del total nacional de inmigrantes vive en la región `r`
durante el año `t`. Por construcción las 16 tasas suman 1, así que el resultado
se lee como un reparto porcentual del país. Se modela además una segunda
variable: la cantidad absoluta de personas por región.

El proyecto integra tres fuentes oficiales que habitualmente se analizan por
separado — las estimaciones de población extranjera del SERMIG, las solicitudes
de residencia resueltas y los indicadores macroeconómicos de los países de
origen — y las combina en un único panel región-año.

## Cómo funciona

**Todas las variables predictoras están rezagadas un año.** Para predecir 2023
sólo se usa lo que se sabía en 2022: el share de la región, la composición
demográfica de su población migrante, la proporción en situación irregular, el
peso de la nacionalidad venezolana, las solicitudes de residencia y el
desempleo ponderado de los países de origen. Esto imita el uso real del modelo
y evita que información del futuro se filtre al entrenamiento.

**La partición es temporal, nunca aleatoria:** se entrena con los años
anteriores y se evalúa con el más reciente (2023). Un split aleatorio en datos
de panel dejaría la misma región repartida entre entrenamiento y prueba,
inflando artificialmente el desempeño.

**Se comparan cuatro algoritmos contra un piso mínimo.** Regresión Lineal
Múltiple, Random Forest y Gradient Boosting —los tres comprometidos en el
anteproyecto— más XGBoost como referencia. El piso es la *persistencia*:
predecir que cada región tendrá este año lo mismo que el año pasado, sin
entrenar nada.

## Qué se encontró

| Modelo | R² Test | RMSE | Brecha R² (train − test) |
|---|---|---|---|
| Persistencia (sin modelo) | 0,9997 | 0,00246 | — |
| Regresión Lineal | 0,999 | 0,00153 | 0,0000 |
| Gradient Boosting | 0,995 | 0,00300 | 0,7791 |
| **Random Forest** (seleccionado) | 0,8639 | 0,05199 | 0,0017 |
| XGBoost | 0,1087 | 0,13302 | 0,8910 |

> Cifras de la Tabla 5.1 del informe final de TT I. La fila de Gradient
> Boosting no coincide con `comparacion_completa_v2.csv` (que da R² 0,9996,
> RMSE 0,00271 y brecha 0,0004): el informe la reportó desde una corrida
> anterior. Reconciliar ambas fuentes es tarea pendiente de TT II.

El resultado más importante del semestre no vino de ningún modelo, sino de esa
fila de persistencia: **una regla que no entrena nada obtiene R² = 0,9997**. Sin
ella habríamos leído un R² de 0,999 como un modelo excelente, cuando en
realidad refleja que la distribución territorial de la migración se mueve muy
poco de un año a otro.

El modelo final es **Random Forest**, y no es el de mejor R². Se eligió por su
estabilidad entre entrenamiento y prueba, y por su comportamiento frente al
único cambio real observado en 2023: la Región Metropolitana duplicó su
proporción de población irregular, quedando fuera del rango visto en
entrenamiento. XGBoost colapsó ahí (brecha train-test de 0,89), mientras Random
Forest, que promedia árboles sobre submuestras distintas, amortiguó el golpe.
Esa ventaja es un argumento de diseño del algoritmo, no un resultado medido:
con un solo año de prueba no se puede verificar. Queda declarado como tal.

También se validó que el desempeño no proviene de fuga de información: un test
de permutación de 500 repeticiones, barajando el objetivo, da R² promedio de
−0,41 y p-valor 0,000.

## Cómo reproducir

```bash
pip install -r requirements.txt

python src/modelo_2_regional_v2/build_dataset_v2.py   # panel región-año, 3 fuentes
python src/modelo_2_regional_v2/train_v2.py           # entrena y exporta el modelo final
python src/modelo_2_regional_v2/comparacion_v2.py     # tabla comparativa de algoritmos
python src/modelo_2_regional_v2/validacion_v2.py      # test de permutación
python src/modelo_3_conteo/train_conteo.py            # segunda variable: cantidad absoluta
python src/proyeccion_regional/proyeccion_regional_combinado.py   # proyección 2024-2028
```

Semilla fija en 42: dos ejecuciones consecutivas producen métricas idénticas.

## Estructura

```
TT1/
├── data/raw/       fuentes originales, nunca se modifican
└── src/
    ├── common/     constantes compartidas (SEED, TEST_YEAR)
    ├── etl/        limpieza y exploracion de las fuentes
    └── modelo_*/   un modelo por carpeta, cada uno con su outputs/
```

| Carpeta | Contenido | ¿En el informe? |
|---|---|---|
| `src/etl/` | Limpieza del censo y del SERMIG, EDA | — |
| `src/common/` | Constantes compartidas (SEED, TEST_YEAR) | — |
| `src/modelo_1_deprecado/` | Lineal + XGBoost, 5 variables. Genera `dataset_region.csv`, insumo de todo lo demás | Sí, como referencia |
| `src/modelo_2_regional_v2/` | **Modelo final**: Random Forest, 8 variables, 3 fuentes | Sí, secciones 5 y 6 |
| `src/modelo_3_conteo/` | Mismo protocolo con la cantidad absoluta como objetivo | Sí, sección 5.2 |
| `src/modelo_combinado/` | Clasificador de irregularidad migratoria | **No** — exploratorio |
| `src/proyeccion_regional/` | Tendencia log-lineal por región, 2024-2028 | Sí, sección 6.6 |

Cada carpeta de modelo guarda sus resultados en su propio `outputs/`, de modo
que siempre se sabe qué script produjo qué archivo.

> El nombre `modelo_1_deprecado` induce a error: esa carpeta **no está
> deprecada**. Su salida `dataset_region.csv` es el punto de partida del modelo
> final. Se mantiene el nombre para no romper el historial.

## Datos

Las fuentes originales viven en `data/raw/` y no se modifican nunca: toda
transformación pasa por código.

| Archivo | Fuente | Uso |
|---|---|---|
| `dataset_combinado.csv` | SERMIG + macro | Estimaciones 2018-2023 e indicadores del país de origen |
| `CensoData.csv` | Censo 2024 (INE) | Variables complementarias, sólo modelo exploratorio |
| `RD-Resueltas-2o-semestre-2025.xlsx` | SERMIG | Solicitudes de residencia resueltas |
| `RD-Acogidas-2o-semestre-2025.xlsx` | SERMIG | Solicitudes acogidas a tramitación, usado en el EDA |

`personas_censo2024.csv` (microdatos censales completos) no se versiona por
peso; `src/etl/etl_censo.py` lo transforma en `CensoData.csv`.

**Archivo faltante:** `8. baseregiones_limpio.csv`, fuente cruda de
`src/modelo_1_deprecado/build_datasets.py`, se perdió en una reorganización
previa. Su salida ya calculada (`outputs/dataset_region.csv`) sí está
versionada y es la que consume el resto del pipeline, por lo que los resultados
son reproducibles desde ese punto en adelante.

## Limitaciones declaradas

- **Muestra pequeña:** 32 observaciones de entrenamiento (16 regiones × 2 años)
  y un solo año de prueba, por lo que las métricas cargan bastante varianza.
- **La validación cruzada no es informativa** con dos años de entrenamiento, y
  el esquema k-fold actual mezcla años entre particiones.
- **Alcance regional, no provincial:** las series históricas sólo existen de
  forma consistente a nivel de región. El Censo 2024 llega a provincia, pero es
  una observación puntual y no una serie temporal.
- **Modelo correlacional, no causal:** no anticipa crisis ni cambios normativos.

## Estado

TT I entregado: los objetivos de recopilación de fuentes y de entrenamiento y
comparación de modelos están cumplidos. El objetivo pendiente —una plataforma
web con dashboard y mapa geoespacial interactivo— es el trabajo de Trabajo de
Título II.
