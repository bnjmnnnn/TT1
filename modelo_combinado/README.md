# modelo_combinado — clasificación regular/irregular (TIPO_MIGRACION)

Target: `TIPO_MIGRACION` = 1 si la tasa de irregularidad del grupo
(`RRAA_IRREGULAR / RRAA_TOTAL`) supera 0,5. Grupos = región × país × sexo ×
edad × año (2018-2023) de `dataset_combinado.csv`, enriquecidos con 5
features censales por región × país × sexo desde `CensoData.csv`
(la versión con códigos INE; `CensoData_normalizado.csv` fue eliminado del
repo — los mapeos de códigos están en `features_censo.py`).

## Cadena de ejecución (en orden)

```bash
python features_censo.py          # -> dataset_combinado_enriquecido.csv
python modelos_basicos.py         # -> resultados_modelos.csv (referencia)
python modelos_mejorados.py       # -> resultados_mejorados.csv
python modelos_xgboost_tuning.py  # -> MODELO FINAL: resultados_tuning.csv,
                                  #    hiperparametros_tuning.json, curva_pr.png,
                                  #    xgb_tuneado.json
```

## Resultados (test 20%, semilla 42, prevalencia clase positiva 9,3%)

| Etapa | Modelo | F1 test | AUC-ROC |
|---|---|---|---|
| Básicos | mejor: Random Forest (base) | 0,459 | 0,900 |
| Mejorados (one-hot + class weight + umbral) | XGBoost base+censo | 0,596 | 0,916 |
| **Tuning (final)** | **XGBoost base+censo tuneado** | **0,619** | **0,925** |

AUC-PR test del modelo final: **0,706** (métrica correcta para el
desbalance; azar = 0,093). Umbral calibrado en validación: 0,644.
Early stopping: 122 árboles. Hiperparámetros completos persistidos en
`hiperparametros_tuning.json`.

Decisiones metodológicas que se conservan de las iteraciones previas
(documentadas en el historial del proyecto):
- `scale_pos_weight` en vez de SMOTE (SMOTE quedó 1-3 puntos F1 por debajo
  en todos los casos cuando se comparó).
- Búsqueda de hiperparámetros sobre AUC-PR, no AUC-ROC.
- Umbral de decisión calibrado en validación, nunca en test.
- Filas con `RRAA_TOTAL = 0` excluidas (target indefinido, no clase 0).

## Rol en el proyecto

Línea **exploratoria** (anexo): no responde la pregunta de investigación
del anteproyecto (regresión de tasa de concentración territorial) — ver
`docs/decision_alcance_regional.md`. Se documenta como aporte adicional.
