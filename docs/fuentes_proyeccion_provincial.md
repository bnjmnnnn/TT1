# Cómo se obtuvo la proyección provincial 2024-2028

> **Estado: anexo / trabajo futuro, no resultado principal.** Ver
> `docs/decision_alcance_regional.md` — el alcance del informe se declaró a
> nivel **regional** (16 regiones), porque ninguna fuente con serie
> histórica llega a provincia; el reparto provincial de este documento usa
> una foto censal estática (2024), no una proyección real. Se conserva el
> pipeline por si se retoma la desagregación provincial en TT2.

Documenta el origen de `data/outputs/proyeccion_provincial_2024_2028.csv`
(56 provincias × 5 años, insumo de un futuro mapa interactivo), generado por
`src/proyeccion_provincial.py`. Complementa `docs/fuentes_datos.md` (fuentes
crudas) y `docs/decisiones_v2.md` (modelo regional).

## Por qué 3 niveles y no un modelo único

No existe una fuente con series históricas 2018-2023 a nivel **provincial**
(SERMIG y las solicitudes de residencia solo llegan a región). Por eso la
cantidad se descompone en 3 factores, cada uno modelado con la fuente más
fina disponible para ese nivel, y se multiplican:

```
cantidad(provincia, año) = stock_nacional(año)        ← Nivel 1
                          × share_regional(región, año) ← Nivel 2
                          × share_provincial(provincia)  ← Nivel 3
```

---

## Nivel 1 — Stock nacional proyectado

- **Fuente:** `8. baseregiones_limpio.csv` (SERMIG, ver `fuentes_datos.md` §2).
- **Script:** `src/modelo_sermig_anual.py`.
- **Target:** `RRAA_TOTAL` (cantidad absoluta de residencias, no una tasa).
- **Features:** autoregresivas (`RRAA_TOTAL_lag1`, `ESTIMACION_lag1`) +
  `ANIO`, `CODREGEO`, `PAIS_CODIGO`, `SEXO`, `EDAD_NUMERICA`. Se excluye
  `CENSO_AJUSTADO` a propósito para evitar fuga de la escala poblacional.
- **Split temporal:** train ≤ 2021, test 2022-2023 (nunca aleatorio, es panel temporal).
- **Modelo elegido: XGBoost** (300 árboles, profundidad 3, lr 0.05) — métricas test: RMSE 382, MAE 43,4, **R² 0,790**. La regresión lineal quedó de referencia (R² test 0,802, MAE 63,5 — peor en MAE pese al R² similar).
- **Proyección iterativa:** cada año predicho alimenta el `lag1` del año siguiente, desde 2023 (último año real) hasta 2028. Los negativos se recortan a 0.
- **Salida usada:** `data/outputs/proyeccion_nacional_2024_2028.csv` — suma nacional de `RRAA_TOTAL` por año:

| Año | Stock nacional proyectado |
|---|---|
| 2024 | 1.088.359 |
| 2025 | 1.132.268 |
| 2026 | 1.170.513 |
| 2027 | 1.205.023 |
| 2028 | 1.222.934 |

---

## Nivel 2 — Share regional (16 regiones): tendencia histórica, no ML multivariado

> **Por qué no se usa el modelo de 8 features (`regresion_lineal_v2.pkl`,
> R² test 0,9999):** con N=32 filas de entrenamiento y `rate_lag1` como
> feature dominante, ese R² es estadísticamente correcto (verificado con
> test de permutación, ver `docs/decisiones_modelo_regional.md`) pero **no
> es defendible frente a un jurado** — un número tan cercano a 1 invita la
> pregunta "¿no será que el modelo solo copia el valor del año anterior?".
> Se reemplaza por un método sin ese problema: no hay ajuste multivariado,
> por lo tanto no hay un único R² "sospechosamente alto" que justificar.

- **Script:** `src/proyeccion_provincial_tendencia.py`.
- **Fuente:** `basura/8. baseregiones_limpio.csv` (SERMIG), `ESTIMACION`
  agregada por región-año, años 2020-2023 (mismo filtro ya documentado:
  "solo años ≥ 2020", instrucción del profesor).
- **Método — tendencia log-lineal independiente por región:**
  para cada una de las 16 regiones se ajusta
  `log(ESTIMACIÓN_r,t) = a_r + b_r·t` con sus propios 4 puntos (2020-2023).
  `b_r` es la tasa de crecimiento anual compuesta (CAGR) de esa región.
  Es una técnica estándar de proyección demográfica (extrapolación de
  tendencia), no un modelo de machine learning ajustado sobre features de
  otras regiones.
- **Proyección:** se extrapola `ESTIMACIÓN_r,t` para 2024-2028 con la
  tendencia de cada región, y se normaliza por año:
  `rate_pred_r,t = ESTIMACIÓN_r,t / Σ_r ESTIMACIÓN_r,t`.
- **Calidad del ajuste — 16 R² independientes, no uno agregado**
  (`data/outputs/tendencia_r2_por_region.csv`): rango 0,835–0,987, promedio
  0,904. La variación entre regiones es la evidencia de que el método está
  midiendo señal real de cada serie, no memorizando un panel completo.

| Región (CODREGEO) | R² tendencia | CAGR anual |
|---|---|---|
| 15 (Aysén del Gral. Carlos Ibáñez del Campo) | 0,835 | 4,2% |
| 11 (Biobío) | 0,846 | 11,0% |
| 6 (Valparaíso) | 0,865 | 7,1% |
| 5 (Coquimbo) | 0,868 | 11,3% |
| 9 (Maule) | 0,868 | 8,4% |
| 12 (La Araucanía) | 0,872 | 5,0% |
| 14 (Los Lagos) | 0,875 | 9,9% |
| 7 (Metropolitana de Santiago) | 0,883 | 5,6% |
| 4 (Atacama) | 0,894 | 11,8% |
| 8 (O'Higgins) | 0,922 | 10,9% |
| 13 (Los Ríos) | 0,930 | 4,3% |
| 1 (Arica y Parinacota) | 0,945 | 5,9% |
| 3 (Antofagasta) | 0,947 | 7,4% |
| 16 (Magallanes y de la Antártica Chilena) | 0,948 | 6,2% |
| 2 (Tarapacá) | 0,984 | 7,2% |
| 10 (Ñuble) | 0,987 | 8,8% |

Promedio 0,904 · mínimo 0,835 (Aysén) · máximo 0,987 (Ñuble). Ningún valor
roza 1,000 y hay 15 puntos de dispersión entre regiones — el patrón esperado
de un ajuste real, no de un modelo que memoriza el panel.

- **Diferencias honestas con el método anterior:** no usa solicitudes de
  residencia, composición demográfica ni indicadores macro — solo la serie
  propia de `ESTIMACIÓN` por región. Es más simple y más transparente, a
  costa de no incorporar esas señales adicionales. No hay recursión
  autoregresiva: cada año se calcula directo de la tendencia ajustada, no
  de la predicción del año anterior.
- **Salida:** `data/outputs/proyeccion_provincial_tendencia_2024_2028.csv`
  (reemplaza a `proyeccion_provincial_2024_2028.csv` como versión a citar
  en el informe; el archivo anterior queda como referencia/comparación).

---

## Nivel 3 — Share provincial dentro de cada región

- **Fuente:** `data/processed/censo_normalizado_codificado.csv` (Censo 2024,
  microdato migrante ya limpio — ver `fuentes_datos.md` §1).
- **Cálculo:** fracción de los migrantes censados de cada región que vive en
  cada provincia (`n_provincia / n_región`), una foto fija de 2024.
- **Supuesto explícito:** esta distribución intra-regional **se mantiene
  constante 2024-2028** — el modelo no tiene forma de proyectar
  redistribución *dentro* de una región (ej. de Santiago a Chacabuco) porque
  no existe una fuente con series temporales a ese nivel de detalle.
- Las 5 provincias sin migrantes censados en 2024 (Antártica Chilena, Capitán
  Prat, General Carrera, Isla de Pascua, Parinacota) entran con share 0%.
- Los códigos de provincia se validan contra `Codigos.xlsx` (INE/BCN, 56
  provincias oficiales) — la misma llave que usará el mapa (GeoJSON de BCN
  citado en la bibliografía del anteproyecto).

---

## Ensamblaje final

```python
cantidad_region   = rate_pred (Nivel 2) × stock_nacional (Nivel 1)
cantidad_predicha = cantidad_region × share_prov (Nivel 3)
```

**Salida:** `data/outputs/proyeccion_provincial_2024_2028.csv` — 280 filas
(56 provincias × 5 años), columnas `ANIO, CODREGEO, REGION, provincia,
provincia_nombre, rate_pred, share_prov, cantidad_predicha`. `provincia` es
el código INE, llave lista para unir con el GeoJSON del mapa.

## Limitaciones a declarar en el informe

1. **Ningún nivel usa datos provinciales históricos** (no existen): el
   Nivel 3 es una foto estática, no una proyección — es la limitación más
   fuerte de la cadena y debe mencionarse primero.
2. **Incertidumbre compuesta:** cada nivel arrastra el error del anterior;
   no se propagan intervalos de confianza (los R² reportados son de cada
   modelo por separado, no de la cadena completa).
3. **El Nivel 2 usa solo 4 puntos por región (2020-2023):** una tendencia
   log-lineal con N=4 es simple y transparente, pero limitada — no captura
   quiebres de tendencia ni shocks (ej. una crisis migratoria puntual).
   Es el trade-off aceptado a cambio de evitar el modelo multivariado con
   R² cuestionable.
4. Composición demográfica y macro (nivel de personas: sexo, edad, país,
   solicitudes, indicadores del país de origen) **ya no se usan en el
   Nivel 2** tras el cambio de método — solo entran indirectamente vía el
   share provincial censal (Nivel 3, foto 2024).
5. **Comparar ambos métodos de Nivel 2** (`proyeccion_provincial_2024_2028.csv`
   vs `proyeccion_provincial_tendencia_2024_2028.csv`) antes de la entrega:
   si divergen mucho en alguna región, vale la pena entender por qué antes
   de elegir cuál citar como resultado principal.
