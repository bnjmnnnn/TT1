# Proyección regional 2024-2028 sin censo (solo dataset_combinado)

Versión alternativa que **no toca el censo en ningún punto** — a diferencia
de `docs/fuentes_proyeccion_provincial.md`, que usa la composición censal
2024 para repartir el número regional entre provincias (Nivel 3).

## Por qué se queda a nivel región y no provincia

`dataset_combinado.csv` solo trae `CODREGEO`/`REGION` — **no tiene columna
de provincia**. El censo es la única de las 4 fuentes con esa granularidad
(ver `docs/fuentes_datos.md` §1). Sin censo, el límite de detalle territorial
que se puede alcanzar es la región (16 unidades), no la provincia (56).
Decisión tomada explícitamente: **no forzar** un reparto provincial
inventado; mostrar solo lo que los datos permiten sostener.

## Método

- **Script:** `src/proyeccion_regional_combinado.py`.
- **Fuente:** `dataset_combinado.csv` completo (sin filtrar por provincia,
  porque no existe esa columna).
- **Limpieza:** `modelo_combinado/limpiar_datos.py` (la misma ya usada y
  documentada para este dataset — codificación SEXO/PAÍS, coerción numérica,
  negativos → valor absoluto excepto `CRECIMIENTO_PIB`).
- **Fusión de niveles:** al no requerir reparto provincial, el stock
  nacional (antiguo Nivel 1) y el share regional (antiguo Nivel 2) se
  calculan en un solo paso: se proyecta directamente `ESTIMACION` de cada
  región con tendencia log-lineal (mismo método que
  `proyeccion_provincial_tendencia.py`, mismo motivo: evitar un modelo
  multivariado con R² agregado difícil de defender), y el total nacional
  **emerge de sumar las 16 proyecciones regionales** — no se modela aparte,
  así que no puede haber inconsistencia entre el total y la suma de partes.
- **Años de ajuste:** 2020-2023 (mismo filtro documentado en el resto del
  proyecto), 4 puntos por región.

## Resultado — ajuste por región (idéntico al de `proyeccion_provincial_tendencia.py`)

Los R² son **exactamente los mismos** que en la versión basada en
`baseregiones_limpio.csv`: la columna `ESTIMACION` de `dataset_combinado.csv`
proviene de la misma fuente SERMIG, solo está empaquetada junto a los
indicadores macro. Esto es evidencia de consistencia (dos archivos con
distinto origen de ensamblaje llegan al mismo número), no un resultado
nuevo — decláralo así si te preguntan, no como "otro modelo que confirma".

| Región | R² tendencia | CAGR anual |
|---|---|---|
| Aysén (peor ajuste) | 0,835 | 4,2% |
| Ñuble (mejor ajuste) | 0,987 | 8,8% |

Tabla completa: `data/outputs/tendencia_r2_por_region_combinado.csv`.

## Proyección de cantidad (top regiones)

| Región | 2024 | 2028 | Crecimiento |
|---|---|---|---|
| Metropolitana de Santiago | 1.122.613 | 1.396.193 | +24,4% |
| Antofagasta | 135.550 | 180.657 | +33,3% |
| Valparaíso | 126.936 | 167.234 | +31,7% |

**Importante — no comparar esta magnitud con `proyeccion_nacional_2024_2028.csv`:**
esa proyección usa `RRAA_TOTAL` (solicitudes de residencia resueltas, un
subconjunto administrativo) como target; esta usa `ESTIMACION` (stock total
de población extranjera estimada, universo mucho mayor). Son variables
distintas — el total nacional aquí (2024: 1.877.371) es más alto que el de
`RRAA_TOTAL` (2024: 1.088.359) porque miden cosas diferentes, no porque uno
sea "mejor" que el otro.

Salida: `data/outputs/proyeccion_regional_combinado_2024_2028.csv`
(80 filas: 16 regiones × 5 años).

## Cuándo usar esta versión vs. la provincial

| | `proyeccion_provincial_tendencia_2024_2028.csv` | `proyeccion_regional_combinado_2024_2028.csv` |
|---|---|---|
| Granularidad | 56 provincias | 16 regiones |
| Usa censo | Sí (solo para repartir dentro de la región) | No, en ningún punto |
| Target | `ESTIMACION` (vía share regional) × share censal | `ESTIMACION` directo |
| Para el mapa interactivo | Necesaria si el mapa es provincial | Solo si el mapa se hace a nivel regional |

Si el mapa final necesita 56 provincias, no hay forma de evitar el censo —
es la única fuente con ese detalle. Esta versión es la alternativa correcta
si se decide (por robustez metodológica, para no depender de una foto
censal fija) bajar la ambición del mapa a nivel regional.
