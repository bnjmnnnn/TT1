# Resumen de sesión de trabajo — 18 de julio de 2026

Documento de apoyo para el avance del Trabajo de Título I. Recoge, en orden
cronológico, las decisiones tomadas, los resultados obtenidos, los problemas
que aparecieron y cómo se resolvieron. Sirve como bitácora para escribir el
informe — no reemplaza la documentación técnica de cada pipeline (`docs/*.md`,
`modelo_combinado/README.md`), que queda como fuente de verdad de detalle.

## 1. Punto de partida

El anteproyecto aprobado (`Anteproyecto Trabajo Título I 2026 - Bastián
Pizarro, Benjamín Fernández.md`) compromete:

- Predecir la **tasa de concentración de población inmigrante** a nivel
  **provincial** (56 provincias).
- Fuentes: estimaciones SERMIG + Censo 2024 + solicitudes de residencia.
- Comparar **Regresión Lineal Múltiple, Random Forest y Gradient Boosting**,
  CV k=5.
- Criterio de éxito: **R² ≥ 0,70**.
- Entrega vía dashboard web interactivo con mapa geoespacial (explícitamente
  pospuesto por decisión propia: *"el mapa lo dejaré para futuro"*).

## 2. Decisión de alcance: región en vez de provincia

**Decisión (documentada en `docs/decision_alcance_regional.md`):** el
resultado principal del proyecto se declara a **nivel regional (16
regiones)**, no provincial.

**Por qué:** ninguna fuente con serie histórica (necesaria para entrenar y
validar un modelo predictivo real) llega a nivel de provincia. El Censo 2024
sí tiene esa granularidad, pero es **una sola foto** — sirve para *repartir*
un número regional entre provincias según composición 2024, no para
*proyectar* redistribución futura dentro de una región. Mantener el nivel
provincial como resultado principal habría exagerado la precisión real del
proyecto ante la comisión.

**Consecuencia para el informe:** el objetivo específico 2 se redacta como
*"predecir la tasa de concentración a nivel regional, declarando la
desagregación provincial como línea de trabajo futuro condicionada a
disponibilidad de series históricas SERMIG a ese nivel"*. El pipeline
provincial no se elimina, pero pasa a anexo/trabajo futuro.

## 3. Modelo core (v2): panel regional de 3 fuentes

Pipeline reproducible en `src/`: `censo_limpieza_v2.py` → `build_dataset_v2.py`
→ `train_v2.py`. Documentado en detalle en `docs/decisiones_v2.md`.

- **Panel:** 16 regiones × 3 años (2021-2023) = 48 filas. Split temporal
  (train 2021-2022, N=32; test 2023, N=16) — **no aleatorio**, porque es
  serie temporal y un split aleatorio filtraría información del futuro.
- **8 features**, todas rezagadas a t−1 (anti-leakage): 5 del modelo base
  (`rate_lag1`, `pct_women_lag1`, `mean_age_lag1`, `pct_irregular_lag1`,
  `pct_venezuela_lag1`) + 2 de solicitudes de residencia (`sol_share_lag1`,
  `sol_pct_otorga_lag1`) + 1 macro del país de origen
  (`macro_desempleo_origen_lag1`).
- Las features censales (`censo_*`) quedan **excluidas del set por defecto**:
  el Censo se levantó en 2024, después del año de test (2023) — usarlas
  sería fuga temporal. Quedan marcadas como candidatas legítimas para
  proyecciones 2025+.

**Resultados (test = 2023):**

| Modelo | R² test | ¿Cumple R² ≥ 0,70? |
|---|---|---|
| Lineal v2 | 0,9999 | Sí |
| Gradient Boosting v2 | 0,9995 | Sí |
| Random Forest v2 | 0,8639 | Sí |
| (referencia) XGBoost modelo 1 | 0,2208 | No |

Los 3 modelos comprometidos en el anteproyecto superan el criterio de éxito.
El hallazgo relevante: Gradient Boosting v2 corrige la falla del XGBoost del
modelo 1 (que no extrapolaba bien fuera del rango de entrenamiento) gracias
a las features nuevas, en particular `sol_share_lag1`.

### 3.1 Validación anti-leakage (test de permutación)

Un R² de 0,9999 es sospechoso a primera vista. Se implementó
`src/validacion_v2.py`: reentrenar cada modelo 500 veces con el target de
**entrenamiento barajado al azar** y comparar contra el R² real.

- R² con targets sin sentido: media ≈ −1,09 (muy por debajo de 0).
- R² real: 0,9999 (Lineal) y 0,9995 (Gradient Boosting).
- p-valor < 0,002 en ambos casos.

**Conclusión:** el R² alto no es fuga de información — es la **persistencia
genuina** de `rate_lag1` (el share regional de migrantes cambia muy poco de
un año a otro). Esto valida el modelo, pero también motiva la pregunta de si
predecir el *nivel* es suficientemente informativo (ver §4).

## 4. Intento de reformular el target (Δrate) — construido y luego revertido

**Motivación:** dado que `rate_lag1` domina el modelo por persistencia,
surgió la idea de predecir el **cambio** año a año (`Δrate = rate_t −
rate_{t-1}`) en vez del nivel, para evitar que un R² alto sea "trivial" por
autocorrelación. Se fijó además una regla dura propia: *"nunca aceptaré un
modelo de más de 0,95 de R² ya que eso es improbable e imposible"*.

**Qué se construyó:** un dataset nuevo (`dataset_ml_raw.csv`, 58.893 filas,
2018-2023) con columnas de solicitudes ya fusionadas con macro, y un pipeline
v3 completo (`build_dataset_v3.py`, `train_v3.py`) con un guardrail de código
literal (`R2_MAXIMO_ACEPTABLE = 0.95`, con `raise RuntimeError` si se
superaba).

**Problema de datos encontrado en el proceso:** las columnas de solicitudes
en `dataset_ml_raw.csv` estaban duplicadas ~16,5 veces por cada grupo
(región, país, sexo, año) — de no corregirse, cualquier suma habría inflado
los conteos en ese mismo factor. Se corrigió con deduplicación antes de
agregar.

**Decisión final:** se revirtió por completo esta línea de trabajo —
*"Elimina todo lo relacionado a v3 y recrea el modelo que nos dio 0,61 en
f1score"*. Se eliminaron `build_dataset_v3.py`, `train_v3.py`,
`dataset_region_v3.csv`, `data/outputs/v3/` y `docs/decisiones_v3.md`. El
**modelo v2 (nivel, no delta) queda como resultado oficial**, ya validado por
permutación. La idea de Δrate queda como posible extensión para TT II, no
como trabajo abandonado sin razón — el registro de por qué se descartó vive
en este documento y en el historial de conversación.

## 5. Línea exploratoria: clasificación regular/irregular (`modelo_combinado/`)

Declarada explícitamente como **anexo, no resultado principal** (no responde
la pregunta de investigación del anteproyecto — es un dato individual, no
territorial agregado).

- **Target:** `TIPO_MIGRACION` = 1 si la tasa de irregularidad del grupo
  (`RRAA_IRREGULAR / RRAA_TOTAL`) > 0,5. Grupos región × país × sexo × edad ×
  año (2018-2023), 45.731 filas tras excluir grupos sin RRAA. Desbalance:
  9,3% clase positiva.
- **Progresión de resultados (test, semilla 42):**

| Etapa | Modelo | F1 test | AUC-ROC | AUC-PR |
|---|---|---|---|---|
| Básicos | Random Forest (base) | 0,459 | 0,900 | — |
| Mejorados (one-hot + class_weight + umbral calibrado) | XGBoost base+censo | 0,596 | 0,916 | — |
| **Tuning (final)** | **XGBoost base+censo tuneado** | **0,619** | **0,925** | **0,706** |

- **Decisiones metodológicas y por qué:**
  - `class_weight='balanced'` / `scale_pos_weight` en vez de SMOTE — se
    probó SMOTE explícitamente y quedó 1-3 puntos F1 por debajo en los 4
    casos comparados (`modelo_combinado/modelos_smote.py`). No se volvió a
    intentar sin razón: hay evidencia empírica en contra para este problema.
  - Búsqueda de hiperparámetros optimizada sobre **AUC-PR**
    (`average_precision`), no AUC-ROC: con 9,3% de clase positiva, AUC-ROC no
    penaliza fallar en la clase rara. Azar en AUC-PR = 0,093 (la prevalencia);
    el modelo final llega a 0,706.
  - Umbral de decisión (0,644) calibrado en un set de **validación**, nunca
    en test — evita que la elección del umbral "vea" los datos que luego se
    reportan como resultado.
  - `RandomizedSearchCV` (60 combinaciones, CV 5-fold) + early stopping
    (paró en 122 árboles de 2000 máximos).
- **Features censales agregadas:** 5 variables calculadas desde el
  microdato censal por grupo región × país × sexo (edad media, escolaridad
  media, % urbano, % llegada reciente, % ocupado), con respaldo por promedio
  nacional cuando no hay match directo (6,4% de los casos).

## 6. Otras líneas exploradas (declaradas anexo/trabajo futuro)

Documentadas pero fuera del cuerpo principal del informe
(`docs/decision_alcance_regional.md` §"Qué queda fuera"):

- Clasificación individuo → región desde censo (`modelo_censo_encoded/`),
  ~58% accuracy.
- Clasificación individuo → provincia (`modelo_censo_provincia/`), ~50,5%
  accuracy.
- **Resultado negativo documentado:** intentar predecir el *conteo* de
  migrantes por provincia a partir de la composición censal
  (`modelo_censo_provincia_conteo/`) no funcionó — R² ≈ 0 / levemente
  negativo. Conclusión metodológica: la composición (perfil de quién migra)
  no explica el volumen (cuánta gente migra) — son preguntas distintas. Vale
  la pena mencionarlo en el informe como evidencia de rigor (se probó, no
  funcionó, se documentó por qué), no como fracaso a ocultar.
- Reparto provincial vía censo (`src/proyeccion_provincial_tendencia.py`,
  `docs/fuentes_proyeccion_provincial.md`): usa la composición censal 2024
  para repartir la proyección regional entre provincias — es una foto
  estática, no una proyección temporal real a ese nivel, por eso no es el
  resultado principal.
- Replicación de un enfoque de un compañero de curso
  (`modelo_regional_referencia/`): comparación metodológica split aleatorio
  vs. split temporal, útil como evidencia de por qué el proyecto usa split
  temporal (un split aleatorio en serie temporal infla el R² artificialmente).

## 7. Proyección a futuro 2024-2028 (nivel regional)

Construida esta sesión en `src/proyeccion_regional_combinado.py`
(reconstruido tras perderse en una reorganización de archivos; metodología
ya estaba documentada en `docs/fuentes_proyeccion_regional_combinado.md`,
solo faltaba el script).

**Por qué no se usa el modelo v2 para proyectar a futuro:** el modelo v2
predice un año usando features reales rezagadas a t−1. Para encadenar varios
años (2024→2025→...→2028) habría que también pronosticar las otras 7
variables rezagadas (no solo `rate`), acumulando error de forma no validada.
No es lo que se decidió hacer.

**Método usado:** regresión log-lineal **independiente por región**
(`log(ESTIMACION) ~ AÑO`), ajustada con 2020-2023 (4 puntos por región).
Ventajas: cada región tiene su propio R² (no hay un R² agregado difícil de
defender), no depende del censo, y el total nacional **emerge de sumar las
16 proyecciones regionales** en vez de modelarse aparte — no puede haber
inconsistencia entre el total y las partes.

**Resultados:**

- R² por región: entre 0,835 (Aysén, peor ajuste) y 0,986 (Ñuble, mejor
  ajuste). Tabla completa: `data/outputs/tendencia_r2_por_region_combinado.csv`.
- Proyección de cantidad: total nacional pasa de 1.877.369 (2024) a
  2.437.668 (2028), +29,8%. Lideran Metropolitana, Antofagasta y Valparaíso.
  Detalle: `data/outputs/proyeccion_regional_combinado_2024_2028.csv`.

**Nota importante para el informe:** esta proyección usa `ESTIMACION` (stock
total de población extranjera), que es una variable distinta a
`RRAA_TOTAL` (solicitudes de residencia resueltas, un subconjunto
administrativo) usada en otras proyecciones exploratorias del repo. No son
comparables directamente entre sí.

## 8. Petición de "datos ficticios" — evaluada y no implementada

Se preguntó si convenía agregar datos ficticios/sintéticos para enriquecer
partes con poca información y muy desbalanceadas. Se distinguió entre:

- **Sintético estadístico válido** (SMOTE, ADASYN, bootstrap): interpola
  entre datos reales existentes con base matemática publicada — ya se probó
  para `TIPO_MIGRACION` y perdió contra `class_weight` (ver §5).
- **Dato ficticio inventado sin proceso estadístico:** no defendible en una
  tesis — un jurado que lo detecte puede cuestionar la integridad de todos
  los resultados.

Para el panel regional (N=48, serie temporal real), no existe ningún proceso
válido para "inventar" cómo habría sido la migración en una región en un año
no medido. **Decisión:** no se implementó. Alternativa recomendada si se
retoma el tema: reportar intervalos de confianza más amplios para regiones
con poca población migrante y declarar la limitación explícitamente en el
informe, en vez de rellenar con datos fabricados.

## 9. Problemas técnicos resueltos en la sesión (bitácora)

- **Bug de modelo equivocado en `validacion_v2.py`:** se probó por error
  `XGBRegressor` en vez del `GradientBoostingRegressor` (sklearn) real que
  usa `train_v2.py`, con nombres de hiperparámetros incorrectos. Se detectó
  porque el R² no coincidía con lo documentado (0,108 vs 0,9995 esperado);
  se corrigió cruzando contra `hiperparametros_v2.json`.
- **Nulos en `sol_share_lag1`/`sol_pct_otorga_lag1`** (Aysén 2021, 1 valor
  cada uno): corregido con interpolación por región antes de construir el
  rezago.
- **Pérdida de archivos por reorganización del repositorio:** en más de una
  ocasión (`modelo_combinado/`, `src/proyeccion_regional_combinado.py`) el
  código se perdió al reorganizar carpetas manualmente, aunque la
  documentación de la metodología sobrevivió. Se reconstruyó cada vez
  verificando que los resultados reproducidos coincidieran con los ya
  documentados (mismo R², mismo F1) antes de darlos por válidos. **Lección
  para el resto del proyecto:** hacer commits de git más seguido evitaría
  tener que reconstruir código desde la documentación.
- **Codificación de consola en Windows:** `UnicodeEncodeError` /
  caracteres corruptos (tildes, Δ) al imprimir en la terminal cp1252. No
  afecta los archivos de salida (CSV/JSON quedan en UTF-8 correctos), solo
  la vista en consola — verificado leyendo los CSV directamente.
- **Fuente de datos censales cambiada:** `CensoData_normalizado.csv`
  (versión con etiquetas) dejó de existir tras una reorganización; se
  adaptó `features_censo.py` para leer `CensoData.csv` (códigos INE
  numéricos) con mapeos explícitos, verificando que los resultados fueran
  numéricamente equivalentes a la versión anterior.

## 10. Estado actual / pendientes

- **Resultado principal (para citar en el informe):** modelo v2 regional,
  3 fuentes, R² ≥ 0,70 en los 3 algoritmos comprometidos, validado contra
  leakage por permutación. Fuente de verdad:
  `data/outputs/v2/comparacion_modelos.csv` + `docs/decisiones_v2.md`.
- **Proyección a futuro:** resuelta a nivel regional (§7), lista para usar
  en la sección de resultados o en un futuro mapa (pospuesto).
- **Mapa interactivo:** explícitamente pospuesto para una etapa futura del
  proyecto (Etapa 6 del anteproyecto) — no se ha trabajado en él.
- **Pendiente sin resolver:** decidir si se prioriza (a) documentar más a
  fondo la separación núcleo/anexo en todo el repo, (b) preparar la reunión
  con el profesor guía, o (c) avanzar en la redacción del informe. Se
  preguntó explícitamente y no se ha definido prioridad.
- **Riesgo a vigilar:** el repo ha perdido código más de una vez por
  reorganizaciones manuales de archivos sin commit previo. Recomendación:
  commitear a git antes de mover/eliminar carpetas.
