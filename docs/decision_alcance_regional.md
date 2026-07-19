# Decisión de alcance: nivel regional (no provincial)

**Fecha:** 18 de julio de 2026. Ajusta el alcance del anteproyecto
(`Anteproyecto Trabajo Título I 2026`, objetivo específico 2 y §7.1: "nivel
provincial, 56 provincias") a lo que los datos disponibles permiten sostener
con rigor: **nivel regional, 16 regiones**.

## Por qué

Ninguna de las fuentes con **serie histórica** (necesaria para entrenar y
validar un modelo predictivo) llega a nivel provincial:

| Fuente | Nivel territorial máximo |
|---|---|
| SERMIG — estimaciones de extranjeros | Región |
| SERMIG — solicitudes de residencia | Región |
| Indicadores macro (`dataset_combinado`) | Región |
| Censo 2024 | **Provincia** — pero es una sola foto (2024), sin serie temporal |

El Censo es la única fuente provincial, y por eso se usó hasta ahora para
*repartir* el número regional entre provincias con su composición 2024
(`share_prov`, ver `docs/fuentes_proyeccion_provincial.md`) — pero ese
reparto es una foto estática, no una proyección: no captura redistribución
intra-regional futura (ej. de Santiago hacia Chacabuco) porque no existe
ninguna fuente con esa granularidad a través del tiempo. Mantenerlo como
resultado principal habría exagerado la precisión real del proyecto.

## Qué cambia

- **Resultado principal declarado: nivel regional (16 regiones).**
- El pipeline provincial (`src/proyeccion_provincial_tendencia.py`,
  `modelo_censo_provincia/`, `share_prov` censal) **se conserva en el repo**
  pero pasa a trabajo futuro / anexo exploratorio — no es el resultado que
  se reporta como cumplimiento del objetivo específico 2.
- El objetivo específico 2 se redacta en el informe final como: *"predecir
  la tasa de concentración de población inmigrante a nivel regional (16
  regiones), declarando la desagregación provincial como línea de trabajo
  futuro condicionada a la disponibilidad de series históricas SERMIG a ese
  nivel de detalle"* — mismo texto que puede ir directo en §7.2 Limitaciones
  del informe.

## Resultado principal a citar (ya cumple el criterio de éxito del anteproyecto, R² ≥ 0,70)

| Fuente de verdad | Contenido |
|---|---|
| `data/outputs/v2/comparacion_modelos.csv` | Lineal / Random Forest / Gradient Boosting comparados, panel región-año, 3 fuentes integradas — **los 3 superan R² ≥ 0,70** |
| `docs/decisiones_v2.md` | Metodología y justificación del panel de 3 fuentes |
| `data/outputs/proyeccion_regional_combinado_2024_2028.csv` | Proyección 2024-2028 por región (16 filas × 5 años), sin ML multivariado cuestionable — tendencia log-lineal por región |
| `docs/fuentes_proyeccion_regional_combinado.md` | Metodología de la proyección regional |
| `data/outputs/conteo/metricas_conteo.csv` | Cantidad absoluta de inmigrantes (no share) por región, año siguiente — modelo elegido: Random Forest, ver `docs/decisiones_conteo.md` |

## Qué queda fuera del cuerpo principal del informe (anexo, no resultado)

- `src/proyeccion_provincial_tendencia.py` y `docs/fuentes_proyeccion_provincial.md` (reparto provincial vía censo).
- `modelo_censo_encoded/`, `modelo_censo_provincia/` (clasificación individuo→territorio).
- `modelo_combinado/` (clasificación regular/irregular, SMOTE, tuning XGBoost).
- `modelo_regional_referencia/` (comparación metodológica de split aleatorio vs. temporal).
- `modelo_censo_provincia_conteo/` (resultado negativo: composición censal no explica el conteo).

Todo lo anterior es trabajo válido y defendible si se pregunta por él, pero
no responde directamente la pregunta de investigación del anteproyecto —
declararlo así evita que la comisión perciba dispersión de objetivos.
