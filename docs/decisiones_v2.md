# Decisiones de modelamiento v2 — Integración de 3 fuentes

**Fecha:** 18 de julio de 2026 · Complementa `decisiones_modelo_regional.md` (modelo 1, que queda **intacto** como línea base de comparación).

## 1. Qué motiva la v2

El documento de avance (`Avance_1_Trabajo_Titulo_I_2026.md`) compromete: (a) integración de **3 fuentes** (SERMIG estimaciones + Censo 2024 + solicitudes de residencia), (b) comparación de **Regresión Lineal Múltiple, Random Forest y Gradient Boosting** con CV k=5, y (c) criterio de éxito R² ≥ 0,70. El modelo 1 usaba solo SERMIG y comparaba Lineal vs XGBoost. La v2 cierra esa brecha **sin alterar ningún artefacto del modelo 1** (verificado con git: `data/outputs/metricas.csv`, `dataset_region.csv`, `src/train.py`, etc. sin cambios).

## 2. Corrección de un descarte previo

`decisiones_modelo_regional.md` descartó las solicitudes de residencia por "cubrir solo el 2° semestre 2025". **Eso era incorrecto**: el nombre del archivo (`RD-Resueltas-2o-semestre-2025.xlsx`) alude a la fecha de publicación, pero su columna `AÑO` cubre **2000–2025** con desagregación región × año × tipo de resolución. Son plenamente utilizables como feature rezagada para el panel 2021–2023, y además cumplen el rol de mitigación del Riesgo 1 declarado en el documento (Cap. 4.7).

## 3. Pipeline v2 (reproducible)

```
python src/censo_limpieza_v2.py    # censo_limpio.csv + censo_features_region.csv
python src/build_dataset_v2.py     # dataset_region_v2.csv (48 filas, checks automáticos)
python src/train_v2.py             # 3 modelos + tabla comparativa vs modelo 1
```

| Script | Entrada | Salida | Decisiones clave |
|---|---|---|---|
| `censo_limpieza_v2.py` | `CensoData_normalizado.csv` | `data/processed/censo_limpio.csv`, `censo_features_region.csv` | Missing estructural → categoría `'Sin dato'` (no se imputa: sesgaría composición). Duplicados exactos NO se eliminan (microdato sin ID: perfiles iguales = personas distintas). Porcentajes agregados excluyen `'Sin dato'` del denominador. |
| `homologacion_v2.py` | — | (módulo compartido) | Llave territorial canónica = `CODREGEO` (orden geográfico, el de `dataset_region.csv`; **no** es el código INE). Homologación por nombre normalizado (sin tildes, mayúsculas). Nacionalidad censal → país de `dataset_combinado`; categorías continentales → `OTRO PAIS`. |
| `build_dataset_v2.py` | `dataset_region.csv` (solo lectura), `RD-Resueltas...xlsx`, `dataset_combinado.csv`, `censo_features_region.csv` | `data/processed/dataset_region_v2.csv` | Mismas 48 filas y target del modelo 1. Todas las features nuevas rezagadas a t−1 (misma regla anti-leakage). Checks: 48 filas, cero nulos, columnas del modelo 1 idénticas byte a byte, rezago verificado. |
| `train_v2.py` | `dataset_region_v2.csv` | `data/outputs/v2/` | Mismo split temporal (train 2021–22, N=32; test 2023, N=16). GridSearchCV k=5 (según documento). Semilla 42. |

## 4. Features v2 (8) — todas rezagadas a t−1

- **Base modelo 1 (comparabilidad):** `rate_lag1`, `pct_women_lag1`, `mean_age_lag1`, `pct_irregular_lag1`, `pct_venezuela_lag1`.
- **Solicitudes de residencia:** `sol_share_lag1` (fracción de las solicitudes otorgadas del año que capta la región — misma naturaleza distributiva que el target), `sol_pct_otorga_lag1` (tasa de aprobación regional).
- **Macro país de origen (de `dataset_combinado`):** `macro_desempleo_origen_lag1` (desempleo del país de origen promediado por región, ponderado por stock estimado de cada nacionalidad). `macro_inflacion_origen_lag1` y `macro_crecimiento_pib_origen_lag1` quedan en el dataset como candidatas (regla 5–8 features con N=32).

**Features censales (`censo_*`) excluidas del set por defecto:** el Censo se levantó en 2024, *después* del año de test 2023; usarlas para predecir 2023 sería leakage temporal. Quedan en el dataset como covariables estructurales candidatas para la proyección 2024+ del TT II (donde sí son legítimas: al proyectar 2025–2028 el censo ya es información pasada).

Solicitudes con región "Anonimizada"/"Sin Información" (~1,8% del total) se excluyen y se documentan como limitación.

## 5. Resultados (test = 2023)

| Modelo | RMSE | MAE | R² test | vs criterio R² ≥ 0,70 |
|---|---|---|---|---|
| Lineal modelo 1 (referencia) | 0,00145 | 0,00108 | 0,9999 | ✔ |
| **Lineal v2** | 0,00153 | 0,00115 | 0,9999 | ✔ |
| **Gradient Boosting v2** | 0,00300 | 0,00203 | 0,9995 | ✔ |
| **Random Forest v2** | 0,05199 | 0,02070 | 0,8639 | ✔ |
| XGBoost modelo 1 (referencia) | 0,12437 | 0,03462 | 0,2208 | ✘ |

**Hallazgos:**

1. **Los 3 modelos comprometidos en el documento superan el criterio de éxito (R² ≥ 0,70).** Con el modelo 1 solo lo lograba el lineal.
2. **Gradient Boosting v2 (R² 0,9995) resuelve el fallo del XGBoost del modelo 1 (R² 0,22).** El diagnóstico del modelo 1 fue que los árboles no extrapolan cuando `pct_irregular_lag1` de la RM salió del rango de entrenamiento. Las features nuevas —en particular `sol_share_lag1`, que replica la escala distributiva del target con información de otra fuente— dan a los árboles rutas de decisión estables que ya no dependen de esa variable frágil.
3. **El lineal v2 empata con el modelo 1** (diferencia en la 4ª cifra decimal): la persistencia del share regional (`rate_lag1`) sigue dominando. Las fuentes nuevas no aportan al lineal, pero rescatan a los modelos de árboles — ese contraste es en sí un resultado presentable.
4. La conclusión del modelo 1 se mantiene: el share regional es altamente persistente; para TT II el target Δrate sigue siendo la extensión propuesta.
5. **Test de permutación (2026-07-18, `src/validacion_v2.py`):** el R² tan alto de Lineal v2 y Gradient Boosting v2 nunca se había verificado contra leakage — solo se le había hecho esa prueba al modelo 1 original de 5 features. Se reentrenó cada modelo 500 veces con el target de entrenamiento barajado al azar: el R² con targets sin sentido cae a una media de -1,09 (std ~1,2-1,8), muy por debajo del R² real (0,9999 y 0,9995), con p-valor < 0,002 en ambos casos. **Conclusión: el R² alto no es fuga de información — es la alta persistencia genuina de `rate_lag1` (el share regional cambia muy poco año a año), igual que se concluyó para el modelo 1.** Resultado: `data/outputs/v2/test_permutacion_v2.csv`.

## 6. Artefactos

| Archivo | Contenido |
|---|---|
| `data/processed/censo_limpio.csv` | Microdato censal limpio (540.291 filas) con llaves homologadas |
| `data/processed/censo_features_region.csv` | Composición censal migrante por región (16 filas) |
| `data/processed/dataset_region_v2.csv` | Panel v2: 48 filas × 29 columnas |
| `data/outputs/v2/metricas_v2.csv` | Métricas de los 3 modelos v2 |
| `data/outputs/v2/comparacion_modelos.csv` | Tabla única modelo 1 vs v2 (fuente del informe) |
| `data/outputs/v2/predicciones_test_v2.csv` | Predicho vs real por región, 3 modelos |
| `data/outputs/v2/hiperparametros_v2.json` | Features, CV, semilla y mejores hiperparámetros |
| `data/outputs/v2/*_v2.pkl` | Modelos exportados |

## 7. Pendientes / decisiones para el profesor

- ~~**Granularidad provincial (56 provincias)**~~ — **RESUELTO 2026-07-18:**
  se declara el alcance a nivel regional (16 regiones), justificado por
  disponibilidad de datos. Ver `docs/decision_alcance_regional.md`.
- `pertenece_pueblo_indigena` y `tiene_religion` usan la categoría literal `"No respuesta"` (no vacío); se dejó como categoría válida.
- El pipeline de clasificación de `modelo_censo/` sigue fuera del alcance del documento: decidir si se declara aporte adicional.
