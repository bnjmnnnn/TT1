# Comparación: filtro temporal ≥2020 vs historia completa (2018–2023)

**Fecha:** 16 de julio de 2026 · **Rama:** `feature/predictive-model`
**Motivación:** el filtro "solo años ≥ 2020" fue instrucción del profesor. Este experimento cuantifica qué cambia si se alimenta el modelo con toda la historia disponible del SERMIG (2018–2023), como evidencia para validar la decisión en el spoiler.

## 1. Diseño del experimento

Se ejecutó el pipeline completo (`build_datasets.py` → `train.py` → `predict.py`) dos veces, cambiando **una sola línea** (`PANEL_START` en `src/config.py`). Todo lo demás idéntico: mismas 5 features rezagadas a t−1, mismo hold-out (test = 2023), misma grilla de 108 combinaciones con LeaveOneOut, semilla 42.

| Escenario | Años del panel | Train | Test |
|---|---|---|---|
| **A — Con filtro ≥2020** (vigente, instrucción del profesor) | 2021–2023 (2020 solo como lag) | 2021–2022, **N=32** | 2023, N=16 |
| **B — Historia completa 2018+** | 2019–2023 (2018 solo como lag) | 2019–2022, **N=64** | 2023, N=16 |

## 2. Métricas de evaluación (test = 2023)

### Regresión Lineal Múltiple (baseline)

| Escenario | RMSE | MAE | R² | Brecha R² train−test |
|---|---|---|---|---|
| A — Filtro ≥2020 | 0.00145 | 0.00108 | 0.99989 | 0.000 |
| B — Historia completa | 0.00190 | 0.00128 | 0.99982 | 0.000 |

### XGBoost (mejores hiperparámetros: lr=0.03, depth=3, mcw=1, n=500 en ambos)

| Escenario | RMSE | MAE | R² | Brecha R² train−test |
|---|---|---|---|---|
| A — Filtro ≥2020 | 0.12437 | 0.03462 | **0.22081** | 0.779 |
| B — Historia completa | 0.09948 | 0.03182 | **0.50156** | 0.498 |

Prueba anti-leakage (target permutado) aprobada en ambos escenarios (R² = −0.60 y −0.75).

### Lectura

1. **XGBoost mejora sustancialmente con más datos:** R² 0.22 → 0.50 (+0.28), RMSE −20%. Confirma que su limitación es el tamaño muestral, no el algoritmo.
2. **Pero sigue perdiendo contra el baseline** en ambos escenarios (0.50 vs 0.9998) y sigue bajo el criterio de éxito del proyecto (R² ≥ 0.60), con sobreajuste persistente (brecha 0.50 > 0.15).
3. **El baseline lineal es insensible al filtro** (R² ≈ 0.9998 en ambos): el share regional es tan persistente que 32 filas le bastan.
4. **Conclusión de selección de modelo: no cambia.** El modelo óptimo (menor RMSE de test) es la regresión lineal en ambos escenarios.

## 3. Predicción 2024–2026 con el modelo ganador (regresión lineal)

Share del stock migrante nacional por región (%), predicción recursiva (2024 desde datos reales 2023; 2025 y 2026 encadenados). **A** = con filtro ≥2020 · **B** = historia completa.

| Región | 2023 real | 2024 A | 2024 B | 2025 A | 2025 B | 2026 A | 2026 B |
|---|---|---|---|---|---|---|---|
| Metropolitana de Santiago | 60.27 | 58.82 | 59.25 | 57.41 | 58.25 | 56.03 | 57.27 |
| Antofagasta | 7.13 | 7.10 | 7.07 | 7.08 | 7.02 | 7.06 | 6.96 |
| Valparaíso | 6.78 | 6.80 | 6.80 | 6.82 | 6.83 | 6.85 | 6.86 |
| Tarapacá | 4.79 | 4.80 | 4.76 | 4.82 | 4.74 | 4.83 | 4.71 |
| O'Higgins | 3.33 | 3.48 | 3.44 | 3.63 | 3.55 | 3.77 | 3.66 |
| Maule | 2.93 | 3.08 | 3.05 | 3.22 | 3.17 | 3.36 | 3.29 |
| Biobío | 2.79 | 3.05 | 2.99 | 3.29 | 3.17 | 3.53 | 3.36 |
| Coquimbo | 2.70 | 2.92 | 2.85 | 3.12 | 3.00 | 3.33 | 3.14 |
| Los Lagos | 2.09 | 2.29 | 2.24 | 2.49 | 2.40 | 2.68 | 2.55 |
| Arica y Parinacota | 2.04 | 2.09 | 2.06 | 2.14 | 2.08 | 2.19 | 2.09 |
| Atacama | 1.51 | 1.74 | 1.65 | 1.97 | 1.78 | 2.19 | 1.91 |
| La Araucanía | 1.40 | 1.41 | 1.42 | 1.43 | 1.45 | 1.44 | 1.47 |
| Ñuble | 0.79 | 0.88 | 0.88 | 0.97 | 0.97 | 1.06 | 1.05 |
| Magallanes | 0.68 | 0.74 | 0.71 | 0.79 | 0.75 | 0.85 | 0.79 |
| Los Ríos | 0.52 | 0.54 | 0.55 | 0.55 | 0.58 | 0.57 | 0.61 |
| Aysén | 0.25 | 0.26 | 0.26 | 0.26 | 0.27 | 0.27 | 0.28 |

### Lectura

1. **Ambos escenarios cuentan la misma historia:** desconcentración gradual de la RM (60.3% → ~56–57% en 2026) con ganancia repartida en regiones intermedias (Biobío, Coquimbo, Maule, Atacama, Los Lagos). Es la continuación de la tendencia observada 2018–2023.
2. **La diferencia máxima entre escenarios es 1.24 puntos porcentuales** (RM en 2026); en las demás regiones es ≤ 0.4 pp. La predicción es **robusta al filtro temporal**.
3. El escenario B (más historia) proyecta una desconcentración algo más lenta de la RM, porque incorpora años (2019–2020) donde la RM aún ganaba peso.

## 4. Supuestos y limitaciones de la predicción (declarar en el informe)

- Predicción recursiva: el share predicho de un año alimenta el rezago del siguiente → el error se acumula (2026 es menos confiable que 2024).
- Composición demográfica (% mujeres, edad media, % irregular, % Venezuela) congelada en lo observado en 2023.
- Sin métricas para 2024–2026 hasta que el SERMIG publique los datos reales; las métricas de evaluación provienen del hold-out 2023.
- **2027–2028 pendientes** como trabajo de TT II (requiere evaluar el error acumulado de la recursión).

## 5. Recomendación para el spoiler al profesor

Presentar la tabla de la sección 2 y preguntar si autoriza usar la historia completa (2018–2023). Argumentos: (a) XGBoost duplica su R² (0.22→0.50) — evidencia de que la restricción era el N; (b) el modelo óptimo y las predicciones 2026 casi no cambian — la decisión no compromete resultados; (c) más historia da más soporte para proyectar a 2026. Si aprueba, el cambio es una línea en `config.py` + re-ejecutar los 3 scripts.
