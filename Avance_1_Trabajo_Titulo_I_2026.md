# Trabajo de Título I

## Modelo Predictivo de Distribución Territorial de Inmigrantes en Chile

**Facultad de Ingeniería — Escuela de Informática y Computación**

---

| **Profesor Guía y Co-Guía** | **Estudiante** |
|---|---|
| **Mauro Castillo Valdés**<br>Profesor de Estado en Matemáticas y Computación / Ingeniero de Ejecución en Informática.<br>Doctor PhD. en Análisis y Procesamiento del Lenguaje<br><br>**Héctor Cifuentes Mella**<br>Ingeniero Civil en Computación | **Benjamín Fernández Toledo**<br>Estudiante de Ingeniería en Informática<br><br>**Bastián Pizarro Pacheco**<br>Estudiante de Ingeniería en Informática |
| mcast@utem.cl<br>hector.cifuentesm@utem.cl | bfernandezt@utem.cl<br>bpizarrop@utem.cl |

---

## Tabla de Contenido

- [Glosario](#glosario)
- [Capítulo 1. Introducción](#capítulo-1-introducción)
  - [1.1 Contexto del Proyecto](#11-contexto-del-proyecto)
  - [1.2 Motivación](#12-motivación)
  - [1.3 Planteamiento del Problema](#13-planteamiento-del-problema)
  - [1.4 Pregunta de Investigación](#14-pregunta-de-investigación)
  - [1.5 Objetivo General](#15-objetivo-general)
  - [1.6 Objetivos Específicos](#16-objetivos-específicos)
  - [1.7 Justificación del Proyecto](#17-justificación-del-proyecto)
  - [1.8 Alcances y Limitaciones](#18-alcances-y-limitaciones)
- [Capítulo 2. Marco Teórico y Estado del Arte](#capítulo-2-marco-teórico-y-estado-del-arte)
  - [2.1 Antecedentes del Problema](#21-antecedentes-del-problema)
  - [2.2 Trabajos Relacionados](#22-trabajos-relacionados)
  - [2.3 Conceptos Técnicos Fundamentales](#23-conceptos-técnicos-fundamentales)
  - [2.4 Tecnologías, Herramientas o Modelos Relevantes](#24-tecnologías-herramientas-o-modelos-relevantes)
  - [2.5 Comparación de Enfoques Existentes](#25-comparación-de-enfoques-existentes)
  - [2.6 Brechas Identificadas](#26-brechas-identificadas)
  - [2.7 Relación del Marco Teórico con la Propuesta](#27-relación-del-marco-teórico-con-la-propuesta)
- [Capítulo 3. Metodología, Planificación y Requerimientos](#capítulo-3-metodología-planificación-y-requerimientos)
  - [3.1 Enfoque Metodológico](#31-enfoque-metodológico)
  - [3.2 Tipo de Proyecto](#32-tipo-de-proyecto)
  - [3.3 Etapas Metodológicas](#33-etapas-metodológicas)
  - [3.4 Actividades y Entregables por Etapa](#34-actividades-y-entregables-por-etapa)
  - [3.5 Requerimientos Funcionales](#35-requerimientos-funcionales)
  - [3.6 Requerimientos No Funcionales](#36-requerimientos-no-funcionales)
  - [3.7 Planificación del Trabajo y Carta Gantt](#37-planificación-del-trabajo-y-carta-gantt)
  - [3.8 Criterios Generales de Éxito](#38-criterios-generales-de-éxito)
- [Capítulo 4. Diseño de la Solución Propuesta](#capítulo-4-diseño-de-la-solución-propuesta)
  - [4.1 Descripción General](#41-descripción-general)
  - [4.2 Arquitectura del Sistema](#42-arquitectura-del-sistema)
  - [4.3 Componentes Principales](#43-componentes-principales)
  - [4.4 Flujo de Funcionamiento](#44-flujo-de-funcionamiento)
  - [4.5 Diseño Técnico](#45-diseño-técnico)
  - [4.6 Justificación de Decisiones de Diseño](#46-justificación-de-decisiones-de-diseño)
  - [4.7 Riesgos Técnicos y Mitigación](#47-riesgos-técnicos-y-mitigación)
- [Capítulo 5. Implementación y Desarrollo](#capítulo-5-implementación-y-desarrollo)
  - [5.1 Entorno de Desarrollo](#51-entorno-de-desarrollo)
  - [5.2 Configuración de Herramientas](#52-configuración-de-herramientas)
  - [5.3 Desarrollo de Módulos](#53-desarrollo-de-módulos)
  - [5.4 Integración de Componentes](#54-integración-de-componentes)
  - [5.5 Descripción del Prototipo](#55-descripción-del-prototipo)
  - [5.6 Problemas Encontrados](#56-problemas-encontrados)
  - [5.7 Soluciones Aplicadas](#57-soluciones-aplicadas)
  - [5.8 Estado Final](#58-estado-final)
- [Capítulo 6. Evaluación y Análisis de Resultados](#capítulo-6-evaluación-y-análisis-de-resultados)
- [Capítulo 7. Conclusiones y Trabajos Futuros](#capítulo-7-conclusiones-y-trabajos-futuros)
- [Bibliografía](#bibliografía)
- [Anexos](#anexos)

---

## Glosario

### (A)

**Aprendizaje automático (Machine Learning):** Rama de la inteligencia artificial que desarrolla algoritmos capaces de aprender patrones a partir de datos y realizar predicciones o clasificaciones sin ser programados explícitamente para cada tarea.

### (C)

**Censo:** Operación estadística que recopila información de todos los individuos o unidades de una población en un momento determinado, con el objetivo de obtener datos completos sobre sus características demográficas, sociales y económicas. En Chile, es realizado periódicamente por el Instituto Nacional de Estadísticas.

### (D)

**Dashboard:** Herramienta visual e interactiva que presenta información clave mediante gráficos, indicadores y tablas, permitiendo monitorear, analizar y comunicar datos de manera clara y en tiempo real para apoyar la toma de decisiones.

**Dataset:** Conjunto estructurado de datos, organizado generalmente en forma de tabla, donde cada fila representa una observación y cada columna una variable o atributo. Se utiliza como base para el análisis, entrenamiento y validación de modelos en ciencia de datos y aprendizaje automático.

### (F)

**Feature (variable predictora):** Atributo o variable de entrada utilizado por el modelo de aprendizaje automático para realizar predicciones. La selección y transformación de features es una etapa crítica del proceso de modelado.

### (M)

**Microdatos:** Conjunto de datos a nivel individual o de unidad de observación (personas, hogares, empresas, etc.), que conserva el detalle original de cada registro sin agregación. Se utilizan para análisis estadísticos y modelamiento, permitiendo estudiar comportamientos y características específicas dentro de una población.

**Modelo predictivo:** Modelo matemático o computacional que utiliza datos históricos para identificar patrones y estimar valores o comportamientos futuros. Se construye mediante técnicas de estadística y aprendizaje automático, y se emplea para apoyar la toma de decisiones en distintos contextos, como predicción de tendencias.

### (R)

**RMSE (Root Mean Squared Error):** Métrica de evaluación de modelos de regresión que mide la raíz cuadrada del promedio de los errores cuadráticos. Indica cuánto se desvían en promedio las predicciones del modelo respecto a los valores reales.

### (S)

**SERMIG (Servicio Nacional de Migraciones):** Organismo público de Chile encargado de gestionar, regular y supervisar los procesos migratorios, incluyendo el control de ingreso, permanencia y regularización de personas extranjeras en el país, así como la implementación de políticas migratorias y la generación de información estadística asociada.

### (V)

**Visualización geoespacial:** Representación gráfica interactiva de datos con componente geográfico, que permite observar la distribución espacial de fenómenos sobre mapas digitales a distintas escalas territoriales.

---

## Capítulo 1. Introducción

### 1.1 Contexto del Proyecto

En la última década Chile ha experimentado un crecimiento acelerado de flujo migratorio, posicionándose como uno de los primeros países en Latinoamérica en términos de recepción de inmigrantes. Este fenómeno ha generado muchas dificultades a nivel gubernamental en materias de planificación territorial, desarrollo de políticas públicas y provisión de servicios públicos, sanitarios y educacionales.

El presente Trabajo de Título propone el desarrollo de un modelo predictivo capaz de estimar la distribución territorial de la población inmigrante en Chile a nivel provincial (56 provincias), utilizando datos provenientes de fuentes oficiales como el SERMIG y el Censo del Instituto Nacional de Estadísticas (INE) para ayudar a solventar el problema planteado anteriormente. Desde la perspectiva de la Ingeniería en Informática y Computación, el proyecto aplica técnicas de aprendizaje automático, procesamiento de datos geoespaciales y desarrollo de sistemas web interactivos para abordar un problema de alto impacto social.

Los resultados del modelo serán presentados mediante una plataforma web interactiva con visualización geoespacial, contribuyendo así a la comprensión de los patrones de asentamiento y movilidad de esta población a lo largo del territorio nacional, siendo una herramienta de apoyo a la toma de decisiones en planificación territorial y políticas públicas.

Este trabajo no contempla eventos externos disruptivos como crisis políticas, nuevas pandemias o cambios drásticos en las leyes fronterizas.

### 1.2 Motivación

En el ramo de Data Science aprendimos a aplicar modelos de machine learning y nos interesó mucho el tema, para trabajo de título pensamos mucho que queríamos realizar pero sabíamos que queríamos realizar un proyecto orientado al Machine Learning, vimos la oportunidad de tomar este tópico que representa un gran desafío pero a su vez es muy interesante y la tomamos.

Principalmente el proyecto representa una oportunidad para aprender mucho en el proceso acerca de procesos ETL, cómo manejar mejor nuestros datos, evaluar de forma comparativa distintos modelos predictivos y el desarrollo de herramientas de visualización geoespacial aplicadas.

### 1.3 Planteamiento del Problema

La creciente complejidad del fenómeno migratorio en Chile demanda herramientas analíticas que permitan proyectar su evolución territorial de forma sistemática y basada en evidencia. Actualmente, no existe una herramienta accesible que integre múltiples fuentes de datos oficiales para generar predicciones a nivel provincial.

El presente proyecto formula la siguiente hipótesis:

> **Hipótesis:** "Si se integran datos oficiales multifuente del SERMIG y del Censo 2024, junto con registros de solicitudes de residencia, entonces es posible construir un modelo predictivo de regresión con capacidad suficiente para estimar la distribución territorial de la población inmigrante a nivel provincial en Chile, entregando resultados útiles para análisis territorial y apoyo a la toma de decisiones de políticas públicas."

### 1.4 Pregunta de Investigación

¿Es posible construir un modelo predictivo de regresión, basado en la integración de datos multifuentes del SERMIG, el censo 2024 y registros de solicitudes de residencia, que estime con precisión la distribución territorial de la población inmigrante a nivel provincial en Chile para el horizonte al 2028?

### 1.5 Objetivo General

Diseñar e implementar una solución informática que integre técnicas de inteligencia artificial, análisis de datos y visualización geoespacial para estimar y presentar la distribución territorial de inmigrantes en Chile mediante un sistema integral.

### 1.6 Objetivos Específicos

- Recopilar las fuentes de datos oficiales relevantes para el análisis migratorio, incluyendo las estimaciones de inmigración del SERMIG, los microdatos del Censo 2024 y los registros de solicitudes de residencia resueltas y no resueltas.
- Construir un dataset consolidado mediante procesos de limpieza, integración y transformación de las fuentes identificadas, seleccionando las variables predictoras más relevantes para el modelo.
- Entrenar y comparar modelos de regresión supervisada (Random Forest Regressor, Gradient Boosting Regressor, Regresión Lineal Múltiple), seleccionando el de mejor desempeño para predecir la tasa de concentración de población inmigrante a nivel provincial en Chile.
- Implementar un sistema integral con dashboard y mapa geoespacial que permita visualizar de forma interactiva los resultados del modelo predictivo, facilitando la interpretación de los patrones territoriales de la inmigración en Chile.

### 1.7 Justificación del Proyecto

El proyecto contempla la integración y análisis de múltiples fuentes de datos complementarias:

- **Estimaciones de Inmigración (SERMIG):** Dataset principal provisto por el Servicio Nacional de Migraciones, que contiene estimaciones oficiales sobre la población inmigrante en Chile.
- **Censo 2024:** Se investigará la adaptación de los microdatos censales para enriquecer el análisis territorial y demográfico del modelo.
- **Solicitudes de Residencia:** Se utilizarán registros de solicitudes de residencia resueltas y no resueltas tramitadas ante el SERMIG, como indicador del flujo migratorio formal.

**Desarrollo del Modelo Predictivo**

Se diseñará e implementará un modelo de aprendizaje automático que integre las fuentes de datos mencionadas para predecir la distribución territorial de la población inmigrante a nivel provincial. El proceso incluirá etapas de preprocesamiento, ingeniería de características, entrenamiento y evaluación del modelo, considerando métricas de desempeño apropiadas para tareas de predicción espacial.

**Visualización y Plataforma Web**

Los resultados del modelo serán expuestos a través de una plataforma web interactiva que funcionará como dashboard analítico. Esta plataforma incorporará un mapa geoespacial que permitirá visualizar de forma dinámica las predicciones de distribución territorial, facilitando la interpretación de los resultados por parte de usuarios técnicos y no técnicos, incluyendo potenciales actores del sector público vinculados a políticas migratorias.

**Relevancia e Impacto**

Este trabajo busca aportar una herramienta de apoyo a la toma de decisiones en materia de política migratoria, planificación territorial y provisión de servicios públicos, en un contexto donde Chile ha experimentado un aumento sostenido de flujos migratorios durante la última década.

### 1.8 Alcances y Limitaciones

#### 1.8.1 Alcances

- Predicción de la distribución de la población inmigrante a nivel provincial (56 provincias) en Chile.
- Horizonte de predicción hasta 2028, acotado a un rango temporal en que la disponibilidad histórica de datos permite mantener fiabilidad estadística en las estimaciones.
- Mapa interactivo de Chile con visualización dinámica de los resultados predictivos por provincia.
- Uso de datos estructurados provenientes de fuentes oficiales.
- Dashboard web accesible para usuarios técnicos y no técnicos.
- La arquitectura del sistema es generalizable a otros fenómenos de distribución territorial, construyendo un aporte de ingeniería reutilizable.

#### 1.8.2 Limitaciones

- No se modela el flujo de inmigrantes entre regiones ni la movilidad interna.
- No se incorpora información sobre flujo de migración no documentada y no oficiales.
- Disponibilidad y oportunidad de datos oficiales y fidedignos.
- Volumen y calidad de los datos disponibles.
- Modelo no causal, sólo correlacional.
- Cambios normativos o eventos externos (crisis, pandemias).

---

## Capítulo 2. Marco Teórico y Estado del Arte

### 2.1 Antecedentes del Problema

Chile ha experimentado una transformación profunda de su perfil migratorio en las últimas dos décadas. Según estimaciones del SERMIG (2025), la población inmigrante en Chile supera el millón de habitantes, concentrándose principalmente en la Región Metropolitana, aunque con creciente dispersión hacia regiones del norte y sur del país.

Esta distribución territorial genera desafíos concretos para la gestión pública, por la asignación de recursos en salud, educación y vivienda. La ausencia de herramientas predictivas accesibles para proyectar esta distribución a nivel provincial constituye una brecha relevante para la gestión del fenómeno migratorio.

El Censo entrega datos actualizados sobre la composición demográfica y territorial de Chile, incluida la población de origen extranjero. Esto, combinado con los registros administrativos del SERMIG, ofrece una oportunidad para construir modelos predictivos de alta resolución territorial.

### 2.2 Trabajos, Soluciones o Enfoques Relacionados

- La OCDE cuenta con un dashboard internacional de migración con comparación entre países.
- UNHCR muestra una plataforma país con información sobre refugiados y migrantes en Chile, pero está orientada a contexto humanitario y no a un modelo predictivo propio.

### 2.3 Conceptos Técnicos Fundamentales

- **Aprendizaje automático**
- **Regresión Lineal Múltiple**
- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- **Métricas de evaluación de regresión**

Sus definiciones están en el glosario.

### 2.4 Tecnologías, Herramientas o Modelos Relevantes

| **Concepto** | **Tecnología / Herramienta** |
|---|---|
| **Recursos humanos** | Para el presente trabajo los únicos recursos humanos requeridos son los 2 estudiantes desarrolladores del trabajo de título. |
| **Hardware** | - Computadores personales.<br>- GPU (opcional, según requerimientos del modelo seleccionado). |
| **Software** | **Entorno y control de versiones:**<br>- Python<br>- Visual Studio Code<br>- Github<br><br>**Procesamiento de datos:**<br>- Polars (Arrow / Rust)<br>- Pandas / Numpy<br>- Openpyxl<br><br>**Modelado y evaluación:**<br>- Scikit-learn<br>- XGBoost o LightGBM<br>- SHAP<br><br>**Datos geoespaciales:**<br>- GeoPandas (manejo de datos geográficos)<br>- Shapely (geometría)<br><br>**Visualización e interfaz:**<br>- Plotly<br>- Folium o Leaflet.js (mapa interactivo)<br>- Mapbox (opcional)<br>- FastAPI o Flask |
| **Apoyo** | - Documentos de Google<br>- Claude<br>- Perplexity<br>- OpenCode |
| **Fuentes de datos estáticas** | - SERMIG<br>- Censo |

### 2.5 Comparación de Enfoques Existentes

Existen algunos enfoques similares que sirven como antecedente para el proyecto, aunque ninguno coincide exactamente con la propuesta planteada. Por un lado, hay tableros internacionales de migración, como los de la OCDE, que permiten visualizar indicadores migratorios comparables entre países, pero con un enfoque principalmente estadístico y descriptivo. Por otro lado, existen plataformas geoespaciales y estudios académicos sobre la distribución territorial de inmigrantes en Chile, que analizan patrones espaciales a partir de datos censales o territoriales, aunque sin incorporar un modelo predictivo integrado en una plataforma web interactiva. En conjunto, estos referentes muestran que el tema ha sido abordado desde la visualización y el análisis espacial, pero dejan un espacio claro para una solución que combine procesamiento de datos, modelado de aprendizaje automático y despliegue web orientado a la predicción territorial.

### 2.6 Brechas Identificadas en la Literatura o en Soluciones Actuales

Como se menciona en el subcapítulo anterior, estas soluciones actuales no están enfocadas principalmente en modelos de machine learning, lo que sería la principal brecha existente.

### 2.7 Relación del Marco Teórico con la Propuesta del Proyecto

El capítulo 2 se relaciona directamente con la propuesta del proyecto, ya que permite fundamentar la necesidad de la solución, sustentar los conceptos técnicos utilizados, delimitar las tecnologías de implementación y evidenciar la brecha existente en las implementaciones actuales. En conjunto, estos elementos justifican el desarrollo de una sistema integral con enfoque geoespacial para la distribución territorial de inmigrantes en Chile.

---

## Capítulo 3. Metodología, Planificación y Requerimientos

### 3.1 Enfoque Metodológico del Proyecto

### 3.2 Tipo de Proyecto: Desarrollo Tecnológico, Prototipo, Prueba de Concepto o Investigación Aplicada

El presente proyecto corresponde a uno de desarrollo tecnológico con componente de investigación aplicada. Se desarrolla un prototipo funcional (modelo predictivo + sistema integral) que integra técnicas de ciencias de datos y visualización geoespacial para abordar un problema de relevancia social. El componente de investigación aplicada se manifiesta en la selección, comparación y evaluación de algoritmos de aprendizaje automático sobre datos reales.

### 3.3 Etapas Metodológicas

La metodología del proyecto se estructura en seis etapas secuenciales, con posibilidad de retroalimentación entre ellas según resultados obtenidos en cada fase. Cada etapa agrupa un conjunto de actividades orientadas a lograr un entregable concreto que contribuye al objetivo general del trabajo.

#### Etapa 1: Investigación de Temas Relevantes

Esta etapa corresponde a la fase inicial del proyecto y tiene por objetivo establecer las bases conceptuales y definir el alcance del trabajo de título.

- **Fase 1: Selección del tema para trabajo de título.** Evaluación de diversos temas competentes y relevantes para realizar el trabajo, considerando relevancia social y viabilidad técnica.
- **Fase 2: Definición del problema.** Análisis del fenómeno migratorio en Chile, revisión de literatura relevante, e identificación de la problemática a abordar.

#### Etapa 2: Recopilación y Comprensión de los Datos

Esta etapa abarca la obtención, exploración y comprensión de las fuentes de datos que alimentarán el modelo predictivo.

- **Fase 1: Descarga de datasets.** Obtención de los datos proporcionados por el SERMIG y el Censo 2024, verificando formatos, integridad y nivel de desagregación territorial.
- **Fase 2: Investigación de los features.** Análisis de los datasets en profundidad para determinar features que serán usados para la creación del modelo.

#### Etapa 3: Preparación y Procesamiento de los Datos

Esta etapa transforma los datos brutos en un dataset estructurado, limpio y listo para el entrenamiento del modelo.

- **Fase 1: Limpieza de archivos.** Tratamiento de valores nulos, duplicados, outliers que puedan afectar el entrenamiento del modelo.
- **Fase 2: Reasignación de variables.** Variables categóricas estandarizadas para su uso en los modelos.
- **Fase 3: Creación de dataset.** Unificación de features elegidos para el modelo de predicción en un solo dataset.

#### Etapa 4: Modelado Predictivo

Esta etapa cubre el diseño, entrenamiento y comparación de los modelos de regresión propuestos.

- **Fase 1: Implementación de modelos base.** Entrenamiento de un modelo de Regresión Lineal Múltiple como línea base para establecer un punto de comparación mínimo de referencia.
- **Fase 2: Implementación de modelos avanzados.** Entrenamiento de Modelo de Random Forest Regressor y Gradient Boosting Regressor, con ajuste de hiperparámetros mediante validación cruzada.
- **Fase 3: Comparación de modelos.** En base a los resultados obtenidos del entrenamiento se selecciona al más óptimo.

#### Etapa 5: Evaluación de los Datos

Esta etapa valida el desempeño del modelo seleccionado y analiza el rol de cada variable en las predicciones.

- **Fase 1: Análisis de los datos.** Identificación de las variables con mayor poder predictivo mediante técnicas de importancia de features, con el fin de comprender cuáles factores explican mejor la distribución territorial.
- **Fase 2: Diagnóstico de resultados.** Análisis detallado de las métricas de evaluación del modelo (RMSE, MAE y R²), revisión de residuales y evaluación de la capacidad de generalización sobre el conjunto de pruebas.

#### Etapa 6: Comunicación de Resultados

Esta etapa final consolida los resultados del proyecto en una plataforma accesible y en el informe técnico correspondiente.

- **Fase 1: Visualización.** Creación de dashboard y mapa para un consumo de datos satisfactorio.
- **Fase 2: Preparación del informe final.** Elaboración y entrega del informe técnico completo del Trabajo de Título II.
- **Fase 3: Preparación de la presentación para la defensa.** Elaboración de la presentación y preparación de la exposición ante la comisión evaluadora.

### 3.4 Actividades y Entregables por Etapa

| **Etapa** | **Nombre** | **Entregable** | **Fecha Hito** | **Estado** |
|---|---|---|---|---|
| **1** | Investigación de temas relevantes | - Formulación del problema<br>- Anteproyecto aprobado | **10 may 26** | **Completado** |
| **2** | Recopilación y comprensión de los datos | - Datasets descargados | **01 jun 26** | **Completado** |
| **3** | Preparación y procesamiento de los datos | - Dataset limpio<br>- Proceso ETL completo | **05 jul 26** | **En progreso** |
| **4** | Modelado predictivo | | **04 oct 26** | **Pendiente** |
| **5** | Evaluación de los datos | | **02 nov 26** | **Pendiente** |
| **6** | Comunicación de resultados | - Dashboard funcional<br>- Sistema integral<br>- Informe final | **06 dic 26** | **Pendiente** |

### 3.5 Requerimientos Funcionales

- El sistema debe permitir cargar o integrar datos provenientes de fuentes oficiales migratorias y censales.
- El sistema debe realizar procesos de limpieza, validación y transformación de los datos antes del modelado.
- El sistema debe mostrar las predicciones en una interfaz web interactiva.
- El sistema debe visualizar los resultados en un mapa geoespacial por provincia.
- El sistema debe permitir consultar métricas de evaluación del modelo.
- El sistema debe permitir filtrar o explorar los resultados por variables territoriales o temporales.

### 3.6 Requerimientos No Funcionales

- El sistema debe presentar una interfaz intuitiva y fácil de usar para usuarios técnicos y no técnicos.
- El sistema debe responder en un tiempo razonable al cargar mapas, gráficos y predicciones.
- El sistema debe mantener integridad y consistencia en el tratamiento de los datos.
- El sistema debe ser modular para facilitar futuras mejoras o incorporación de nuevas fuentes.
- El sistema debe ser compatible con navegadores web modernos.
- El sistema debe proteger el acceso a funciones administrativas o de carga de datos, si corresponde.
- El sistema debe permitir mantenimiento y actualización sin afectar el funcionamiento general.

### 3.7 Planificación del Trabajo y Carta Gantt

El desarrollo del proyecto abarca dos semestres académicos (Trabajo de Título I y II), con inicio en abril de 2026 y cierre en diciembre. Comenzará con la fase inicial de definición (Fase 0), seguidas por sus etapas de investigación y desarrollo, y cierre.

#### 3.7.1 Plan de Hitos

| **N°** | **Hitos** | **Fecha** |
|---|---|---|
| 0 | Entrega Anteproyecto | 10 may 26 |
| 2 | Entrega informe de avance 1 | 14 jun 26 |
| 4 | Entrega final - Trabajo de Título I | 07 jul 26 |
| 6 | Entrega avance 2 - Trabajo de Título II | 11 oct 26 (Estimada) |
| 8 | Entrega avance 3 - Trabajo de Título II | 08 nov 26 (Estimada) |
| 10 | Entrega final - Trabajo de Título II | 06 dic 26 (Estimada) |

*Fuente: Elaboración propia.*

#### 3.7.2 Carta Gantt

*(Ver imágenes adjuntas en el documento original: Actividades Carta Gantt y Planificación Carta Gantt)*

### 3.8 Criterios Generales de Éxito del Proyecto

El proyecto se considerará exitoso si cumple los siguientes criterios técnicos y funcionales al momento de la entrega final:

- **Criterio 1 - Calidad del modelo predictivo:** El modelo seleccionado debe alcanzar un coeficiente de determinación R² mayor o igual a 0,70 sobre el conjunto de prueba, indicando capacidad explicativa suficiente para su uso como herramienta de apoyo a decisiones.

---

## Capítulo 4. Diseño de la Solución Propuesta

### 4.1 Descripción General de la Solución

El proyecto de predicción territorial de inmigración será una solución tecnológica de tres capas que integra un pipeline de procesamiento de datos de alto volumen, un módulo de modelado de aprendizaje automático supervisado y una plataforma web de visualización geoespacial. Su propósito es proveer a organismos institucionales (MINREL, INE, SERMIG, OEA) una herramienta cuantitativa para la toma de decisiones de política pública en materia migratoria.

### 4.2 Arquitectura del Sistema o Prototipo

La arquitectura del sistema se organiza en tres capas con responsabilidades claramente delimitadas:

*(Ver diagrama de arquitectura en el documento original)*

**Capa 1 — Fuentes de datos oficiales**

Comprende las tres fuentes de datos de entrada del sistema, todas de carácter oficial y público:

- **SERMIG - Estimaciones de extranjeros residentes:** Dataset histórico de estimaciones de población inmigrante por provincia y año. Formato CSV, granularidad anual, cobertura nacional (56 provincias).
- **Censo 2024 (INE) - Microdatos de personas:** Archivo de 18.480.432 registros en 63 variables (2.400 MB). Contiene información demográfica, territorial y migratoria a nivel de individuo.
- **SERMIG - Solicitudes de residencia:** Registros administrativos de solicitudes resueltas y no resueltas. Granularidad mensual; se agrega a nivel anual para compatibilidad con las estimaciones SERMIG.

**Capa 2 — Procesamiento ETL y modelado predictivo**

Núcleo técnico del sistema. Comprende cuatro módulos secuenciales:

- **Pipeline ETL:** Procesa el Censo 2024 en streaming (chunks de 100.000 filas), filtra, limpia y genera variables derivadas. Produce CensoData.csv (540.291 registros).
- **Módulo de integración:** Unifica las tres fuentes mediante join por código de provincia INE. Selecciona los features predictores mediante análisis de correlación y VIF. Produce dataset_final.csv.
- **Módulo de modelado:** Entrena tres algoritmos (Regresión Lineal, Random Forest, Gradient Boosting) con validación cruzada k=5. Evalúa con métricas RMSE, MAE y R². Exporta el modelo óptimo como modelo_optimo.pkl.
- **Módulo de predicción:** Carga el modelo exportado y genera predicciones para las 56 provincias en el horizonte 2025–2028. Exporta predicciones.csv.

**Capa 3 — Plataforma web de visualización**

Interfaz de usuario que expone los resultados del sistema a usuarios institucionales:

- **Backend API (Flask/FastAPI):** Carga predicciones.csv y expone endpoints REST que alimentan el frontend.
- **Mapa geoespacial (Folium/Leaflet.js):** Mapa choropleth interactivo de Chile con cobertura provincial (GeoJSON INE). Permite visualizar la distribución predicha de inmigrantes por provincia y año.
- **Dashboard:** Filtros por año y provincia, gráficos de tendencia temporal y tabla comparativa de predicciones.

### 4.3 Componentes Principales

- **Módulo ETL:** Responsable de la descarga, validación, limpieza e integración de las tres fuentes de datos en un dataset consolidado.
- **Módulo de selección de features:** Análisis de correlaciones y selección de las variables predictoras más relevantes mediante técnicas estadísticas.
- **Módulo de entrenamiento y comparación de modelos:** Implementación de los tres algoritmos de regresión con validación cruzada k-fold y registro de métricas.
- **Módulo de predicción:** Generación de estimaciones para el horizonte 2025–2028 usando el modelo óptimo seleccionado.
- **Módulo de visualización web:** Dashboard con mapa interactivo de Chile a nivel provincial y filtros por año.

### 4.4 Flujo de Funcionamiento

*(Ver diagrama de flujo en el documento original)*

### 4.5 Diseño Lógico, Físico o Técnico de la Solución

### 4.6 Justificación de las Decisiones de Diseño

### 4.7 Riesgos Técnicos y Medidas de Mitigación

Se identifican los siguientes riesgos técnicos y sus medidas de mitigación.

- **Riesgo 1 - Insuficiencia de datos históricos:** El dataset SERMIG puede tener cobertura temporal limitada, reduciendo el número de observaciones para entrenamiento. Mitigación: incorporación de solicitudes de residencia como variable proxy para aumentar la riqueza informacional del dataset consolidado.

---

## Capítulo 5. Implementación y Desarrollo

### 5.1 Entorno de Desarrollo

### 5.2 Configuración de Herramientas, Plataformas o Dispositivos

### 5.3 Desarrollo de Módulos o Componentes Principales

Se avanzó en las Etapas 1 y 2, que ya están completadas, y la 3 en progreso:

**Etapa 1 completada - Investigación y definición**

Se completó la revisión de literatura relevante, formulación de problema y la entrega del anteproyecto (10 de mayo 2026). El anteproyecto fue aprobado.

**Etapa 2 Completada - Recopilación y comprensión de datos e implementación del pipeline ETL**

Se descargaron y documentaron las tres fuentes de datos del proyecto. En el caso del Censo 2024, se completó un proceso ETL de alto volumen cuyo resultado constituye el principal insumo del preprocesamiento.

- **Dataset SERMIG - Estimaciones de personas extranjeras residentes en Chile:** Datos históricos de estimaciones de población inmigrante por provincia y año. Se verificó la integridad del archivo, se documentó la estructura de columnas y se identificaron los features candidatos para el modelo.
- **Registros de solicitudes de residencia (SERMIG):** Se descargaron los registros de solicitudes resueltas y no resueltas. Se evaluó su granularidad territorial y temporal para determinar su uso como feature predictivo.
- **Microdatos Censo 2024 (INE) - Pipeline ETL completado y validado:** El dataset fuente contenía 18.480.432 registros en 63 columnas (2.400 MB). Se implementó un pipeline ETL en Python con lectura en streaming por chunks de 100.000 filas. El resultado es un archivo CensoData.csv de 540.291 observaciones en 20 variables, reducción del 97,08%. El proceso fue validado mediante checksum MD5 y muestreo aleatorio, confirmando integridad total.

**Detalle técnico: Pipeline ETL Censo 2024**

El procesamiento del Censo 2024 constituye la actividad técnica de mayor complejidad de la Etapa 2 y materializa la identidad de ingeniería del proyecto. Se documenta en detalle:

- **Extracción:** Lectura del archivo personas_censo2024.csv (delimitador `;`, UTF-8) en chunks de 100.000 filas mediante el módulo csv de Python estándar, resolviendo la imposibilidad de carga completa del archivo de 2.400 MB en memoria.
- **Filtrado:** Se conservaron únicamente registros de personas con información válida sobre período de llegada al país. La reducción de 18.480.432 a 540.291 registros (97,08%) es coherente con el universo analítico definido.
- **Selección de columnas:** De 63 variables originales se conservaron 20 por pertinencia analítica. Se descartaron 43 columnas correspondientes a identificadores internos, dimensiones fuera del alcance o campos redundantes.
- **Limpieza:** Valores faltantes codificados como -99 y -66 normalizados a cadenas vacías para evitar su interpretación como observaciones válidas.
- **Variables derivadas creadas:** `rango_edad` (agrupación etaria discreta) y `llegada_reciente` (indicador binario de llegada en período reciente).
- **Exportación:** Formato CSV con delimitador `,` y codificación UTF-8, compatible con pandas y scikit-learn.
- **Validación:** Checksum MD5, conteo de filas y muestra aleatoria confirman integridad total del proceso.

**Etapa 3 en progreso — Preprocesamiento**

Al momento de la entrega del presente informe, el proceso de limpieza de datos se encuentra iniciado. Las actividades realizadas incluyen:

- Identificación de valores nulos y duplicados en el dataset SERMIG.
- Análisis de la distribución de variables numéricas y detección preliminar de outliers.
- Evaluación de la compatibilidad de codificación territorial entre las tres fuentes (códigos de provincia INE).

Las actividades pendientes de la Etapa 3 — codificación de variables categóricas, integración de fuentes y construcción del dataset consolidado — se completarán conforme al plan de hitos.

### 5.4 Integración de Componentes

La integración de los tres datasets aún no ha sido realizada al momento de este avance. Está planificada como parte de la fase 3 de la Etapa 3 (Creación de dataset), con fecha objetivo de 05 de julio de 2026. La estrategia de integración se basará en la clave provincial estándar del INE como identificador común entre las tres fuentes.

### 5.5 Descripción del Prototipo o Sistema Implementado

Al cierre del Avance 1, el estado de implementación del sistema es el siguiente:

**Componente implementado y validado - Pipeline ETL Censo 2024:** Completamente implementado. El script `etl_censo.py` procesa el archivo fuente de 18.480.432 registros en streaming y produce CensoData.csv (540.291 observaciones, 20 variables). La validación mediante checksum MD5, conteo de filas y muestreo aleatorio confirma integridad total.

**Componente en progreso - Preprocesamiento SERMIG:** Los datasets de estimaciones y solicitudes de residencia han sido descargados y documentados. Se ha realizado exploración inicial, identificación de valores nulos y análisis de distribución de variables. La codificación de variables categóricas e integración de fuentes están planificadas para completarse el 5 de julio de 2026.

**Componentes pendientes:** *(Ver documento original)*

### 5.6 Problemas Encontrados Durante el Desarrollo

Durante el desarrollo de las etapas completadas se identificaron los siguientes problemas técnicos:

### 5.7 Soluciones Aplicadas y Ajustes Realizados

### 5.8 Estado Final de la Implementación

---

## Capítulo 6. Evaluación y Análisis de Resultados

### 6.1 Diseño de las Pruebas o Escenarios de Evaluación

### 6.2 Métricas de Evaluación

### 6.3 Procedimiento de Prueba

### 6.4 Resultados Obtenidos

### 6.5 Análisis de Resultados

### 6.6 Comparación con los Objetivos del Proyecto

### 6.7 Limitaciones Detectadas en la Evaluación

---

## Capítulo 7. Conclusiones y Trabajos Futuros

### 7.1 Conclusiones del Trabajo Realizado

### 7.2 Cumplimiento del Objetivo General y Objetivos Específicos

### 7.3 Principales Aportes del Proyecto

### 7.4 Trabajos Futuros

---

## Bibliografía

1. Biblioteca del Congreso Nacional de Chile. (s.f.). *Mapoteca: Provincias*. Sistema Integrado de Información Territorial. https://www.bcn.cl/siit/mapoteca/provincias

2. Servicio Nacional de Migraciones. (2025, julio). *Estadísticas SERMIG: Reporte N° 4* [Archivo PDF]. https://serviciomigraciones.cl/wp-content/uploads/2025/07/Reporte-4-Estadisticas-SERMIG-010725.pdf

3. Servicio Nacional de Migraciones. (s.f.). *Estimaciones de personas extranjeras residentes en Chile*. https://serviciomigraciones.cl/estudios-migratorios/estimaciones-de-extranjeros/

4. Subsecretaría de Previsión Social. (2025, diciembre). *Estudio: Análisis de la migración en Chile* [Archivo PDF]. https://previsionsocial.gob.cl/wp-content/uploads/2025/12/Estudio-Analisis-de-la-migracion-en-Chile.pdf

5. Instituto Nacional de Estadísticas. (2024). *Resultados — Censo 2024*. INE. https://censo2024.ine.gob.cl/resultados/

---

## Anexos

### Anexo A. Diagramas Técnicos

### Anexo B. Código Fuente Relevante

### Anexo C. Manual de Instalación o Uso, si Corresponde
