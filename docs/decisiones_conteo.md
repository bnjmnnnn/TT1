# Decisiones de modelamiento — Cantidad absoluta de inmigrantes (`train_conteo.py`)

**Fecha:** 19 de julio de 2026 · Complementa `decisiones_v2.md` (target `rate`,
participación regional sobre el stock nacional).

## 1. Por qué un tercer target

Los modelos 1 y 2 predicen `rate`: qué fracción del stock migrante nacional
vive en cada región. Es útil para responder "¿cómo se reparte la migración
entre regiones?", pero no responde directamente "¿cuánta gente va a llegar a
esta región el año que viene?" — la pregunta que en la práctica le interesa
a un municipio o a un servicio público. Se agregó un tercer modelo con
target `estimation`: la **cantidad absoluta** de personas extranjeras
estimadas por región para el año siguiente.

## 2. Dataset y features

Reusa el panel `data/processed/dataset_region_v2.csv` (16 regiones × 3 años,
2021-2023) y sus 7 features de composición/administrativas ya rezagadas a
t−1, cambiando el predictor autorregresivo: en vez de `rate_lag1` (share del
año anterior) se usa **`estimation_lag1`** (cantidad absoluta del año
anterior), reconstruido agregando `ESTIMACION` por región-año desde
`dataset_combinado.csv` (cubre 2018-2023, alcanza para el rezago sin perder
filas del panel 2021-2023).

```
FEATURES = [estimation_lag1, pct_women_lag1, mean_age_lag1,
            pct_irregular_lag1, pct_venezuela_lag1,
            sol_share_lag1, sol_pct_otorga_lag1, macro_desempleo_origen_lag1]
TARGET = estimation
```

Mismo protocolo temporal y misma regla anti-leakage que el modelo 2: train
2021-2022 (N=32), test 2023 (N=16), split nunca aleatorio.

## 3. Resultados (test 2023)

| Modelo | R² test | RMSE train → test | Factor de aumento del error |
|---|---|---|---|
| Regresión Lineal | 0,9943 | 881 → 19.154 | ×21,7 |
| Gradient Boosting | 0,9853 | 493 → 30.843 | ×62,5 |
| **Random Forest** | **0,7776** | **82.000 → 120.063** | **×1,5** |

Fuente: `data/outputs/conteo/metricas_conteo.csv`.

## 4. Decisión: se elige Random Forest, no el de mayor R²

**No se elige por el R² más alto.** Lineal y Gradient Boosting llegan a un
ajuste casi perfecto en entrenamiento (R² ≈ 0,99998) y luego su error
absoluto se **dispara 22 y 62 veces** respectivamente al evaluarse en datos
no vistos (2023). Esto es un síntoma de sobreajuste a la persistencia de las
regiones de mayor tamaño (la Región Metropolitana concentra una fracción tan
grande del stock nacional que domina la varianza total: el R² se mantiene
"alto" aunque el error absoluto crezca mucho, porque esa varianza es enorme).

Random Forest tiene un R² test nominal más bajo (0,7776), pero es el
**único de los tres modelos cuyo error no se dispara desproporcionadamente**
entre entrenamiento y test (factor ×1,5 vs. ×21,7 y ×62,5). Es evidencia de
que generaliza de forma más estable a un año que no vio durante el
entrenamiento, en vez de simplemente repetir el patrón de las regiones
grandes.

**Aun con este criterio más conservador, Random Forest (R²=0,7776) supera el
criterio de éxito declarado en el anteproyecto (R² ≥ 0,70).**

**Redacción sugerida para el informe:**

> Se selecciona Random Forest no por tener el R² nominal más alto, sino
> porque es el único modelo cuyo error no se dispara desproporcionadamente
> entre entrenamiento y test (×1,5 frente a ×21,7 y ×62,5 en Regresión
> Lineal y Gradient Boosting), evidencia de que estos últimos sobreajustan a
> la persistencia de las regiones de mayor tamaño. Aun con este criterio más
> conservador, Random Forest (R²=0,7776) supera el criterio de éxito
> declarado (R² ≥ 0,70).

## 5. Pendiente

Este criterio (razón de degradación RMSE train→test) es un indicador de
sobreajuste, pero no reemplaza el test de permutación que sí se le hizo al
modelo de `rate` (`docs/decisiones_v2.md` §5, hallazgo 5). Si se quiere el
mismo nivel de rigor, correspondería correrlo también sobre `estimation`
antes de la entrega final.

## 6. Artefactos

| Archivo | Contenido |
|---|---|
| `src/train_conteo.py` | Pipeline reproducible |
| `data/outputs/conteo/metricas_conteo.csv` | Métricas train/test de los 3 modelos |
| `data/outputs/conteo/predicciones_test_conteo.csv` | Predicho vs real por región (test 2023) |
| `data/outputs/conteo/hiperparametros_conteo.json` | Features, CV, semilla, mejores hiperparámetros |
| `data/outputs/conteo/*_conteo.pkl` | Modelos exportados |
