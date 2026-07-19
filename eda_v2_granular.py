import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (12, 6)
sns.set_style("whitegrid")

OUTDIR = r"C:\Users\benja\OneDrive\Escritorio\TT1\EDA_output_v2"
os.makedirs(OUTDIR, exist_ok=True)

print("="*60)
print("EDA v2 - MANTENIENDO GRANULARIDAD ORIGINAL (~62k filas)")
print("="*60)

# ============================================================================
# 1. CARGA
# ============================================================================
print("\n[1] CARGA DE DATOS...")

df = pd.read_csv(r"C:\Users\benja\OneDrive\Escritorio\TT1\dataset_combinado.csv")
print(f"dataset_combinado.csv: {df.shape}")

acogidas = pd.read_excel(r"C:\Users\benja\OneDrive\Escritorio\TT1\RD-Acogidas-2o-semestre-2025.xlsx")
acogidas.columns = [c.strip() for c in acogidas.columns]
print(f"RD-Acogidas: {acogidas.shape}")

resueltas = pd.read_excel(r"C:\Users\benja\OneDrive\Escritorio\TT1\RD-Resueltas-2o-semestre-2025.xlsx")
resueltas.columns = [c.strip() for c in resueltas.columns]
print(f"RD-Resueltas: {resueltas.shape}")

# ============================================================================
# 2. LIMPIEZA Y NORMALIZACIÓN DE NOMBRES
# ============================================================================
print("\n[2] LIMPIEZA Y NORMALIZACIÓN...")

# --- Dataset combinado ---
df.columns = [c.strip() for c in df.columns]

# Normalizar textos: strip, lower, quitar tildes innecesarias para consistencia
def norm_text(s):
    if pd.isna(s):
        return s
    return str(s).strip().lower()

# Columnas clave en df
df['region_norm'] = df['REGION'].apply(norm_text)
df['pais_norm'] = df['PAIS'].apply(norm_text)
df['sexo_norm'] = df['SEXO'].apply(norm_text)
df['edad_norm'] = df['EDAD'].apply(norm_text)
df['anio'] = df['AÑO'].astype(int)

# Excluir solo regiones no geocodificables (conservar otros países como categoría)
exclude_regions = ['región ignorada', 'sin información', 'anonimizada']
df = df[~df['region_norm'].isin(exclude_regions)].copy()

print(f"  Filas tras excluir regiones ignoradas: {df.shape[0]}")

# --- Acogidas ---
acogidas['Total'] = pd.to_numeric(acogidas['Total'].astype(str).str.replace(',', ''), errors='coerce')
acogidas['region_norm'] = acogidas['REGIÓN'].apply(norm_text)
acogidas['pais_norm'] = acogidas['PAÍS'].apply(norm_text)
acogidas['anio'] = acogidas['AÑO'].astype(int)

# Excluir categorías agregadas y no geocodificables
acogidas = acogidas[~acogidas['region_norm'].isin(exclude_regions)].copy()
acogidas = acogidas[~acogidas['pais_norm'].isin(['otros países', 'otros países dentro de los 25 primeros'])].copy()

# --- Resueltas ---
resueltas['Total'] = pd.to_numeric(resueltas['Total'].astype(str).str.replace(',', ''), errors='coerce')
resueltas['region_norm'] = resueltas['REGIÓN'].apply(norm_text)
resueltas['pais_norm'] = resueltas['PAÍS'].apply(norm_text)
resueltas['anio'] = resueltas['AÑO'].astype(int)

resueltas = resueltas[~resueltas['region_norm'].isin(exclude_regions)].copy()
resueltas = resueltas[~resueltas['pais_norm'].isin(['otros países', 'otros países dentro de los 25 primeros'])].copy()

# Verificar intersección
print(f"\n  Regiones en df: {df['region_norm'].nunique()}")
print(f"  Regiones en acogidas: {acogidas['region_norm'].nunique()}")
print(f"  Regiones en resueltas: {resueltas['region_norm'].nunique()}")
print(f"  Países en df: {df['pais_norm'].nunique()}")
print(f"  Países en acogidas: {acogidas['pais_norm'].nunique()}")

reg_int = set(df['region_norm'].unique()) & set(acogidas['region_norm'].unique())
pai_int = set(df['pais_norm'].unique()) & set(acogidas['pais_norm'].unique())
print(f"\n  Regiones comunes: {len(reg_int)}")
print(f"  Países comunes: {len(pai_int)}")

# Mapeos para países con nombres distintos
pais_map_df_to_excel = {
    'r. dominicana': 'república dominicana',
    'estados unidos': 'estados unidos',
    'otro país': 'otros países',  # no hay equivalente exacto, pero lo mapeamos
    'país ignorado': 'país ignorado',
}

# Aplicar mapeo en df
df['pais_norm_merge'] = df['pais_norm'].replace(pais_map_df_to_excel)

# Verificar intersección post-mapeo
pai_int2 = set(df['pais_norm_merge'].unique()) & set(acogidas['pais_norm'].unique())
print(f"  Países comunes post-mapeo: {len(pai_int2)}")
print(f"  En df pero no en acogidas: {sorted(set(df['pais_norm_merge'].unique()) - set(acogidas['pais_norm'].unique()))}")
print(f"  En acogidas pero no en df: {sorted(set(acogidas['pais_norm'].unique()) - set(df['pais_norm_merge'].unique()))}")

# ============================================================================
# 3. AGREGAR SOLICITUDES A NIVEL PAIS-REGION-ANIO
# ============================================================================
print("\n[3] AGREGACIÓN DE SOLICITUDES...")

# Acogidas: sumar Total por pais-region-anio
acog_agg = acogidas.groupby(['pais_norm', 'region_norm', 'anio'], as_index=False).agg({
    'Total': 'sum'
})
acog_agg = acog_agg.rename(columns={'Total': 'ACOGIDAS_TOTAL'})
print(f"  Acogidas agregadas: {acog_agg.shape}")

# Resueltas: pivot por tipo, luego agregar por pais-region-anio
res_pivot = resueltas.pivot_table(
    index=['pais_norm', 'region_norm', 'anio'],
    columns='TIPO_RESUELTO',
    values='Total',
    aggfunc='sum',
    fill_value=0
).reset_index()
res_pivot.columns.name = None

# Renombrar columnas
res_pivot = res_pivot.rename(columns={
    'Otorga': 'RESUELTAS_OTORGA',
    'Rechaza con Rt': 'RESUELTAS_RECHAZA_RT',
    'Rechaza con abandono': 'RESUELTAS_RECHAZA_ABANDONO',
    'Archiva': 'RESUELTAS_ARCHIVA'
})

# Totales y tasas
res_pivot['RESUELTAS_TOTAL'] = (
    res_pivot['RESUELTAS_OTORGA'] + 
    res_pivot['RESUELTAS_RECHAZA_RT'] + 
    res_pivot['RESUELTAS_RECHAZA_ABANDONO'] + 
    res_pivot['RESUELTAS_ARCHIVA']
)
res_pivot['TASA_OTORGA'] = res_pivot['RESUELTAS_OTORGA'] / (res_pivot['RESUELTAS_TOTAL'] + 1e-9)
res_pivot['TASA_RECHAZO'] = (res_pivot['RESUELTAS_RECHAZA_RT'] + res_pivot['RESUELTAS_RECHAZA_ABANDONO']) / (res_pivot['RESUELTAS_TOTAL'] + 1e-9)
res_pivot['TASA_ARCHIVO'] = res_pivot['RESUELTAS_ARCHIVA'] / (res_pivot['RESUELTAS_TOTAL'] + 1e-9)

print(f"  Resueltas agregadas: {res_pivot.shape}")

# ============================================================================
# 4. MERGE AL DATASET ORIGINAL (left join, conservando ~62k filas)
# ============================================================================
print("\n[4] MERGE CONSERVANDO GRANULARIDAD...")

merged = df.merge(
    acog_agg,
    left_on=['pais_norm_merge', 'region_norm', 'anio'],
    right_on=['pais_norm', 'region_norm', 'anio'],
    how='left',
    suffixes=('', '_acog')
)
# Limpiar columnas duplicadas del merge
merged = merged.drop(columns=[c for c in merged.columns if c.endswith('_acog')])

merged = merged.merge(
    res_pivot,
    left_on=['pais_norm_merge', 'region_norm', 'anio'],
    right_on=['pais_norm', 'region_norm', 'anio'],
    how='left',
    suffixes=('', '_res')
)
merged = merged.drop(columns=[c for c in merged.columns if c.endswith('_res')])

print(f"  Dataset mergeado: {merged.shape}")
print(f"  Columnas: {merged.columns.tolist()}")

# Missings
print(f"\n  Missings por columna:")
missings = merged.isnull().sum().sort_values(ascending=False)
print(missings[missings > 0])

# Guardar
merged.to_csv(os.path.join(OUTDIR, 'dataset_mergeado_granular.csv'), index=False, encoding='utf-8-sig')

# ============================================================================
# 5. ANÁLISIS UNIVARIADO
# ============================================================================
print("\n[5] ANÁLISIS UNIVARIADO...")

numeric_cols = ['ESTIMACION', 'RRAA_TOTAL', 'RRAA_REGULAR', 'RRAA_IRREGULAR',
                'CENSO AJUSTADO', 'INFLACION', 'CRECIMIENTO_PIB', 'DESEMPLEO',
                'ACOGIDAS_TOTAL', 'RESUELTAS_TOTAL', 'TASA_OTORGA', 'TASA_RECHAZO']

# Estadísticas descriptivas
stats = merged[numeric_cols].describe().T
stats['missing_pct'] = merged[numeric_cols].isnull().mean() * 100
stats['skewness'] = merged[numeric_cols].skew()
print(stats.round(2))
stats.to_csv(os.path.join(OUTDIR, 'estadisticas_descriptivas.csv'), encoding='utf-8-sig')

# Figura: distribución del target
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
merged['ESTIMACION'].hist(bins=50, ax=axes[0], edgecolor='black')
axes[0].set_title('Distribución ESTIMACIÓN')
axes[0].set_xlabel('Estimación')
axes[0].axvline(merged['ESTIMACION'].median(), color='red', linestyle='--', label=f"Mediana: {merged['ESTIMACION'].median():.0f}")
axes[0].legend()

np.log1p(merged['ESTIMACION']).hist(bins=50, ax=axes[1], edgecolor='black', color='green', alpha=0.7)
axes[1].set_title('log(1 + ESTIMACIÓN)')
axes[1].set_xlabel('log(1 + Estimación)')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '01_target.png'), dpi=150, bbox_inches='tight')
plt.close()

# Figura: features nuevas (solicitudes)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, col in enumerate(['ACOGIDAS_TOTAL', 'RESUELTAS_TOTAL', 'TASA_OTORGA', 'TASA_RECHAZO']):
    ax = axes[idx // 2, idx % 2]
    data = merged[col].dropna()
    if len(data) > 0:
        data.hist(bins=40, ax=ax, edgecolor='black', alpha=0.7)
        ax.set_title(f'{col} (n={len(data)}, mediana={data.median():.2f})')
    else:
        ax.set_title(f'{col} - SIN DATOS')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '02_solicitudes.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# 6. ANÁLISIS BIVARIADO Y CORRELACIONES
# ============================================================================
print("\n[6] ANÁLISIS BIVARIADO...")

corr_cols = ['ESTIMACION', 'RRAA_TOTAL', 'INFLACION', 'CRECIMIENTO_PIB', 
             'DESEMPLEO', 'ACOGIDAS_TOTAL', 'RESUELTAS_TOTAL', 'TASA_OTORGA', 'TASA_RECHAZO']
corr_matrix = merged[corr_cols].corr()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0, square=True, ax=ax)
ax.set_title('Matriz de correlación')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '03_correlaciones.png'), dpi=150, bbox_inches='tight')
plt.close()

print("\n  Correlaciones con ESTIMACIÓN:")
corr_target = corr_matrix['ESTIMACION'].drop('ESTIMACION').sort_values(key=abs, ascending=False)
print(corr_target.round(3))

# Scatter plots clave
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
subset = merged[['ESTIMACION', 'ACOGIDAS_TOTAL']].dropna()
if len(subset) > 0:
    axes[0,0].scatter(subset['ACOGIDAS_TOTAL'], subset['ESTIMACION'], alpha=0.3, s=10)
    axes[0,0].set_title(f'ESTIMACIÓN vs ACOGIDAS (r={subset.corr().iloc[0,1]:.3f})')
    axes[0,0].set_yscale('log')

subset = merged[['ESTIMACION', 'RESUELTAS_TOTAL']].dropna()
if len(subset) > 0:
    axes[0,1].scatter(subset['RESUELTAS_TOTAL'], subset['ESTIMACION'], alpha=0.3, s=10)
    axes[0,1].set_title(f'ESTIMACIÓN vs RESUELTAS (r={subset.corr().iloc[0,1]:.3f})')
    axes[0,1].set_yscale('log')

subset = merged[['ESTIMACION', 'TASA_OTORGA']].dropna()
if len(subset) > 0:
    axes[1,0].scatter(subset['TASA_OTORGA'], subset['ESTIMACION'], alpha=0.3, s=10)
    axes[1,0].set_title(f'ESTIMACIÓN vs TASA_OTORGA (r={subset.corr().iloc[0,1]:.3f})')
    axes[1,0].set_yscale('log')

subset = merged[['ESTIMACION', 'INFLACION']].dropna()
if len(subset) > 0:
    axes[1,1].scatter(subset['INFLACION'], subset['ESTIMACION'], alpha=0.3, s=10)
    axes[1,1].set_title(f'ESTIMACIÓN vs INFLACIÓN (r={subset.corr().iloc[0,1]:.3f})')
    axes[1,1].set_yscale('log')

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, '04_scatters.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# 7. INGENIERÍA DE FEATURES
# ============================================================================
print("\n[7] INGENIERÍA DE FEATURES...")

merged_fe = merged.copy()

# Ordenar
merged_fe = merged_fe.sort_values(['region_norm', 'pais_norm_merge', 'sexo_norm', 'edad_norm', 'anio']).reset_index(drop=True)

# Rezagos de flujo (a nivel pais-region, no sexo-edad)
merged_fe['ACOGIDAS_LAG1'] = merged_fe.groupby(['region_norm', 'pais_norm_merge'])['ACOGIDAS_TOTAL'].shift(1)
merged_fe['RESUELTAS_LAG1'] = merged_fe.groupby(['region_norm', 'pais_norm_merge'])['RESUELTAS_TOTAL'].shift(1)

# Tasas derivadas
merged_fe['TASA_REGULARIZACION'] = merged_fe['RRAA_TOTAL'] / (merged_fe['ESTIMACION'] + 1e-9)
merged_fe['PROP_ESTIMACION_GRUPO'] = merged_fe['ESTIMACION'] / merged_fe.groupby(['region_norm', 'pais_norm_merge', 'anio'])['ESTIMACION'].transform('sum')

# Proporciones de flujo
merged_fe['ACOGIDAS_PER_CAPITA'] = merged_fe['ACOGIDAS_TOTAL'] / (merged_fe['ESTIMACION'] + 1e-9)
merged_fe['RESUELTAS_PER_CAPITA'] = merged_fe['RESUELTAS_TOTAL'] / (merged_fe['ESTIMACION'] + 1e-9)

merged_fe = merged_fe.replace([np.inf, -np.inf], np.nan)

# Correlaciones de nuevas features
print("\n  Nuevas features - correlación con ESTIMACIÓN:")
new_feats = ['ACOGIDAS_LAG1', 'RESUELTAS_LAG1', 
             'TASA_REGULARIZACION', 
             'PROP_ESTIMACION_GRUPO', 'ACOGIDAS_PER_CAPITA', 'RESUELTAS_PER_CAPITA']
for f in new_feats:
    corr = merged_fe[[f, 'ESTIMACION']].corr().iloc[0,1]
    miss = merged_fe[f].isnull().mean()*100
    print(f"    {f}: corr={corr:.3f}, missings={miss:.1f}%")

merged_fe.to_csv(os.path.join(OUTDIR, 'dataset_final_features.csv'), index=False, encoding='utf-8-sig')

# ============================================================================
# 8. RESUMEN Y RECOMENDACIONES
# ============================================================================
print("\n" + "="*60)
print("RESUMEN EJECUTIVO")
print("="*60)

report = f"""
DATASET FINAL (GRANULARIDAD ORIGINAL)
======================================
- Filas: {merged_fe.shape[0]} (objetivo: conservar ~62k del dataset combinado)
- Columnas: {merged_fe.shape[1]}
- Nivel: SEXO x EDAD x PAIS x REGION x ANIO
- Periodo: 2018-2023
- Regiones: {merged_fe['region_norm'].nunique()}
- Países: {merged_fe['pais_norm_merge'].nunique()}
- Sexos: {merged_fe['sexo_norm'].nunique()}
- Rangos etarios: {merged_fe['edad_norm'].nunique()}

TARGET (ESTIMACIÓN)
====================
- Media: {merged_fe['ESTIMACION'].mean():.1f}
- Mediana: {merged_fe['ESTIMACION'].median():.1f}
- Skewness: {merged_fe['ESTIMACION'].skew():.2f}
- Recomendación: usar log(ESTIMACION)

MISSINGS CLAVE
==============
{merged_fe.isnull().sum().sort_values(ascending=False).head(10).to_string()}

CORRELACIONES CON ESTIMACIÓN (TOP)
==================================
{merged_fe.corr(numeric_only=True)['ESTIMACION'].drop('ESTIMACION').sort_values(key=abs, ascending=False).head(10).round(3).to_string()}

RECOMENDACIONES
================
1. Imputar ACOGIDAS/RESUELTAS missings con 0 (sin solicitudes = 0)
2. Imputar macro con mediana por país
3. RRAA_TOTAL es muy predictivo (r=0.91)
4. Solicitudes acogidas/resueltas totales correlacionan positivamente (r~0.58)
5. Tasa de otorgamiento tiene correlación negativa débil (r=-0.07)
6. Variables macro individuales son débiles, pero pueden ser útiles en ensemble
7. Se ELIMINÓ ESTIMACION_LAG1 por data leakage (es el target del año anterior)
8. Se ELIMINÓ ESTIMACION_PCT_CHANGE por depender del lag
"""

print(report)
with open(os.path.join(OUTDIR, 'reporte_eda_v2.txt'), 'w', encoding='utf-8') as f:
    f.write(report)

print(f"\n>> Resultados guardados en: {OUTDIR}")
