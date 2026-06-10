import csv
import sys
import os
from datetime import datetime

# Configuración
SOURCE_FILE = r'C:\Users\benja\OneDrive\Escritorio\TT1\personas_censo2024.csv'
TARGET_FILE = r'C:\Users\benja\OneDrive\Escritorio\TT1\CensoData.csv'
# Alternativamente, puedes usar paths relativos:
# SOURCE_FILE = 'personas_censo2024.csv'
# TARGET_FILE = 'CensoData.csv'

# Columnas que se mantienen directamente del source (mapeo 1:1)
DIRECT_COLUMNS = [
    'region',
    'provincia', 
    'comuna',
    'area',
    'sexo',
    'edad',
    'edad_quinquenal',
    'p26_llegada_periodo',
    'p27_nacionalidad',
    'p27_nacionalidad_rec',
    'p27_nacionalidad_esp',
    'p25_lug_nacimiento',
    'p25_lug_nacimiento_rec',
    'escolaridad',
    'cine11',
    'sit_fuerza_trabajo',
    'p40_cise_rec',
    'p45_medio_transporte',
    'p28_autoid_pueblo',
    'p31_religion_rec'
]

# Columnas del target (incluyendo las calculadas)
TARGET_COLUMNS = DIRECT_COLUMNS + ['llegada_reciente', 'rango_edad']

# Valores que representan datos faltantes en el source
MISSING_VALUES = {'-99', '-66', ''}


def clean_value(value):
    """Limpia valores: convierte -99 y -66 a '' (vacío)"""
    if value is None:
        return ''
    value = str(value).strip()
    if value in MISSING_VALUES:
        return ''
    return value


def calcular_rango_edad(edad_str):
    """
    Calcula el rango de edad basado en el valor numérico de edad.
    
    Reglas identificadas:
    - 0-17: edad de 0 a 17 años
    - 18-30: edad de 18 a 30 años  
    - 31-45: edad de 31 a 45 años
    - 46-60: edad de 46 a 60 años
    - 60+: edad de 60 años o más
    """
    try:
        edad = int(edad_str)
    except (ValueError, TypeError):
        return ''
    
    if edad < 0:
        return ''
    elif edad <= 17:
        return '0-17'
    elif edad <= 30:
        return '18-30'
    elif edad <= 45:
        return '31-45'
    elif edad <= 60:
        return '46-60'
    else:
        return '60+'


def calcular_llegada_reciente(p26_str):
    """
    Calcula llegada_reciente basado en p26_llegada_periodo.
    
    Regla identificada:
    - p26 = '1' -> llegada_reciente = '1'
    - p26 = '2' -> llegada_reciente = '0'
    - Otros valores -> '' (no debería ocurrir tras el filtro)
    """
    p26_clean = str(p26_str).strip()
    if p26_clean == '1':
        return '1'
    elif p26_clean == '2':
        return '0'
    else:
        return ''


def aplicar_filtro(row):
    """
    Aplica los criterios de filtrado identificados.
    
    Filtro:
    - p27_nacionalidad == '3'
    - p27_nacionalidad_rec == '2'
    - p26_llegada_periodo en ['1', '2']
    """
    p27 = str(row.get('p27_nacionalidad', '')).strip()
    p27_rec = str(row.get('p27_nacionalidad_rec', '')).strip()
    p26 = str(row.get('p26_llegada_periodo', '')).strip()
    
    return (
        p27 == '3' and
        p27_rec == '2' and
        p26 in ('1', '2')
    )


def procesar_fila(row):
    """
    Procesa una fila del source y devuelve la fila transformada para el target.
    """
    # Aplicar filtro
    if not aplicar_filtro(row):
        return None
    
    # Crear nueva fila con columnas directas
    new_row = {}
    for col in DIRECT_COLUMNS:
        new_row[col] = clean_value(row.get(col, ''))
    
    # Calcular columnas derivadas
    new_row['llegada_reciente'] = calcular_llegada_reciente(row.get('p26_llegada_periodo', ''))
    new_row['rango_edad'] = calcular_rango_edad(row.get('edad', ''))
    
    return new_row


def ejecutar_etl():
    """
    Ejecuta el proceso ETL completo.
    """
    print("=" * 60)
    print("ETL: personas_censo2024.csv -> CensoData.csv")
    print("=" * 60)
    print(f"Source: {SOURCE_FILE}")
    print(f"Target: {TARGET_FILE}")
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Verificar que existe el archivo source
    if not os.path.exists(SOURCE_FILE):
        print(f"ERROR: No se encontró el archivo source: {SOURCE_FILE}")
        sys.exit(1)
    
    total_rows = 0
    filtered_rows = 0
    processed_rows = 0
    
    try:
        with open(SOURCE_FILE, 'r', encoding='utf-8', newline='') as infile, \
             open(TARGET_FILE, 'w', encoding='utf-8', newline='') as outfile:
            
            # Leer source con delimitador ';'
            reader = csv.DictReader(infile, delimiter=';')
            
            # Escribir target con delimitador ','
            writer = csv.DictWriter(outfile, fieldnames=TARGET_COLUMNS)
            writer.writeheader()
            
            # Procesar fila por fila (streaming para manejar archivos grandes)
            for row in reader:
                total_rows += 1
                
                # Aplicar filtro
                if not aplicar_filtro(row):
                    continue
                
                filtered_rows += 1
                
                # Procesar fila
                new_row = procesar_fila(row)
                if new_row:
                    writer.writerow(new_row)
                    processed_rows += 1
                
                # Log de progreso cada 1M filas
                if total_rows % 1000000 == 0:
                    print(f"  Procesadas: {total_rows:,} | Filtradas: {filtered_rows:,} | Escritas: {processed_rows:,}")
        
        print()
        print("=" * 60)
        print("ETL Completado")
        print("=" * 60)
        print(f"Total filas leídas: {total_rows:,}")
        print(f"Filas tras filtro: {filtered_rows:,}")
        print(f"Filas escritas: {processed_rows:,}")
        print(f"Ratio de filtrado: {filtered_rows/total_rows*100:.2f}%")
        print(f"Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\nERROR durante el proceso ETL: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    ejecutar_etl()
