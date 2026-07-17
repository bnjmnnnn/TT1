# Decisiones de modelamiento — Línea regional (SERMIG)

**Fecha:** 16 de julio de 2026 · **Rama:** `feature/predictive-model`
Complementa `desarrollo_modelo.md`; registra las decisiones tomadas al implementar las Fases 1, 2 y 4 y los resultados obtenidos.

## 1. Variable objetivo (Fase 1 — Opción B)

```
rate_{r,t} = ESTIMACION_{r,t} / Σ_r ESTIMACION_{r,t}
```

- **Numerador:** stock estimado de personas extranjeras de la región `r` en el año `t` (columna `ESTIMACION`, SERMIG).
- **Denominador:** stock nacional *regionalizado* del año `t` (suma de las 16 regiones).
- **Interpretación:** cómo se reparte territorialmente el stock migrante; las tasas suman 1 por año (verificado con tolerancia 0,001).
- **Por qué se descartó la Opción A (población total):** requiere una fuente externa de denominador (proyecciones INE) y mide penetración, no distribución; la Opción B responde directamente la pregunta de investigación y se construye del propio SERMIG. *(Pendiente: validar con el profesor en el spoiler N°1.)*
- **`REGIÓN IGNORADA` (código 17) se excluye** como unidad de modelamiento: no es un territorio predecible y su peso fluctúa entre 3,5 % y 13 % del stock según el año. Se declara como limitación.

## 2. Dataset de modelamiento (Fase 2)

`data/processed/dataset_region.csv` — 48 filas (16 regiones × años 2021–2023), generado por `src/build_datasets.py` desde `8. baseregiones_limpio.csv`.

- Filtro temporal ≥ 2020 (instrucción del profesor). 2020 se usa **solo** como fuente del rezago; se pierden sus 16 filas (contadas y explicadas).
- **Regla anti-leakage:** todas las features van rezagadas a t−1; ninguna variable del año t entra como predictor (la composición contemporánea del stock se deriva del mismo `ESTIMACION` que forma el numerador del target).
- Features candidatas (todas `_lag1`): `rate`, `pct_women`, `mean_age`, `pct_irregular`, `pct_venezuela`, `pct_peru`, `pct_colombia`, `pct_haiti`, `pct_bolivia`.
- Set inicial usado en el modelo (5 de 9, por regla N pequeño → 5–8 features): `rate_lag1`, `pct_women_lag1`, `mean_age_lag1`, `pct_irregular_lag1`, `pct_venezuela_lag1`. La selección con evidencia (correlación/VIF/RF, Fase 3) queda como tarea T5 sobre las 9 candidatas.
- Checks automáticos en `build_datasets.py` (el script falla si alguno no se cumple): suma de tasas ≈ 1, sin duplicados región-año, cero nulos, panel completo, dtypes numéricos, rezago verificado (Valparaíso 2022 ← 2021), sanity check RM lidera.
- Las solicitudes de residencia (XLSX) cubren solo el 2° semestre 2025: **no** son utilizables como feature para 2021–2023; se documenta el descarte.

## 3. Protocolo de modelamiento (Fase 4)

- **Split temporal:** train 2021–2022 (N=32) · test 2023 (N=16). Nunca split aleatorio (panel temporal).
- **Baseline primero:** Regresión Lineal Múltiple (`StandardScaler` + `LinearRegression`).
- **XGBoost:** configuración conservadora (max_depth 3, lr 0.05, min_child_weight 3, reg_lambda 1) + grilla acotada de 108 combinaciones con **LeaveOneOut** (N_train = 32 < 80, decisión documentada según Fase 4.1).
- Semilla fija 42 en todo; métricas RMSE/MAE/R² en train y test → `data/outputs/metricas.csv`.

## 4. Resultados (test = 2023)

| Modelo | RMSE test | MAE test | R² test | Brecha R² train−test |
|---|---|---|---|---|
| Regresión lineal | 0,00145 | 0,00108 | 0,9999 | 0,000 |
| XGBoost (lr 0.03, depth 3, mcw 1, 500 árboles) | 0,12437 | 0,03462 | 0,2208 | 0,779 ← sobreajuste |

**Hallazgos (van al capítulo de resultados):**

1. **El baseline lineal supera ampliamente a XGBoost.** Es el escenario que la guía declaraba defendible con N pequeño: la complejidad adicional no aporta con 32 filas de entrenamiento.
2. **El R² ≈ 1 del baseline no es leakage:** la prueba del target permutado (Fase 5.8) da R² = −0,60 (esperado ≈ 0). La causa real es la **alta persistencia** del share regional: `rate_lag1` explica casi todo — la distribución territorial del stock cambia muy poco año a año.
3. **Diagnóstico del fallo de XGBoost:** el error se concentra en una sola región (RM: real 0,603, predicho 0,106). Causa: `pct_irregular_lag1` de la RM se duplicó en 2023 (0,024 → 0,048), quedando fuera del rango de entrenamiento; los árboles no extrapolan y enrutaron a la RM hacia hojas de regiones pequeñas. Ejemplo de manual de fragilidad de árboles con N pequeño y distribución cambiante.
4. **Implicancia para TT II:** con persistencia tan alta, predecir el *cambio* del share (Δrate) sería un target más informativo y exigente que el nivel; se propone como trabajo futuro.

## 5. Artefactos generados

| Archivo | Contenido |
|---|---|
| `data/processed/dataset_region.csv` | Panel de modelamiento región-año |
| `data/outputs/metricas.csv` | Métricas por modelo × split (fuente única del informe) |
| `data/outputs/predicciones_test_region.csv` | Predicho vs real por región (test 2023) |
| `data/outputs/xgb_region.json` + `_meta.json` | XGBoost exportado (formato nativo) + metadatos y versiones |
| `data/outputs/baseline_region.pkl` + `_meta.json` | Baseline exportado |

Reproducir: `pip install -r requirements.txt`, luego `python src/build_datasets.py && python src/train.py` (dos ejecuciones seguidas dan métricas idénticas; recarga del modelo XGBoost verificada automáticamente).
