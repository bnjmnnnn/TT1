import pandas as pd

def limpiar_datos(df):
    if df.empty:
        print("El DataFrame esta vacio.")
        return df
    else:
        # Primero transformar SEXO y EDAD (ANTES de convertir a numérico)
        # Transformar H y M a 1 y 0 respectivamente en la columna "SEXO"
        df['SEXO'] = df['SEXO'].map({'H': 1, 'M': 0})
        print(f"Columna SEXO codificada: H->1, M->0")

        # Pasar EDAD de rangos a promedio numerico
        edad_map = {
            "00 A 04": 2,
            "05 A 09": 7,
            "10 A 14": 12,
            "15 A 19": 17,
            "20 A 24": 22,
            "25 A 29": 27,
            "30 A 34": 32,
            "35 A 39": 37,
            "40 A 44": 42,
            "45 A 49": 47,
            "50 A 54": 52,
            "55 A 59": 57,
            "60 A 64": 62,
            "65 A 69": 67,
            "70 A 74": 72,
            "75 A 79": 77,
            "80 O MÁS": 85,
            "IGNORADA": -1
        }
        
        # Crear columna EDAD_NUMERICA
        df['EDAD_NUMERICA'] = df['EDAD'].map(edad_map)
        
        # Verificar si hay valores no mapeados
        valores_no_mapeados = df[df['EDAD_NUMERICA'].isna()]['EDAD'].unique()
        if len(valores_no_mapeados) > 0:
            print(f"\n Valores de EDAD no reconocidos: {valores_no_mapeados}")
            df['EDAD_NUMERICA'] = df['EDAD_NUMERICA'].fillna(-1)
        
        print(f"Columna EDAD_NUMERICA creada")
        
        # Codificar PAIS a números
        pais_map = {
            "ARGENTINA": 1,
            "BOLIVIA": 2,
            "BRASIL": 3,
            "COLOMBIA": 4,
            "ECUADOR": 5,
            "PERÚ": 6,
            "VENEZUELA": 7,
            "PARAGUAY": 8,
            "URUGUAY": 9,
            "ESTADOS UNIDOS": 10,
            "ESPAÑA": 11,
            "MÉXICO": 12,
            "CUBA": 13,
            "R. DOMINICANA": 14,
            "HAITÍ": 15,
            "CHINA": 16,
            "ALEMANIA": 17,
            "FRANCIA": 18,
            "ITALIA": 19,
            "OTRO PAÍS": 20,
            "PAÍS IGNORADO": 0
        }

        # Crear columna PAIS_CODIGO
        df['PAIS_CODIGO'] = df['PAIS'].map(pais_map)
        
        # Verificar si hay países no mapeados
        valores_pais_no_mapeados = df[df['PAIS_CODIGO'].isna()]['PAIS'].unique()
        if len(valores_pais_no_mapeados) > 0:
            print(f"\nPaíses no reconocidos: {valores_pais_no_mapeados}")
            df['PAIS_CODIGO'] = df['PAIS_CODIGO'].fillna(0)
        
        print(f"Columna PAIS_CODIGO creada")
        
        # Arreglo para convertir columnas numéricas que realmente existan en el dataset
        columnas_numericas_posibles = ["CENSO AJUSTADO", "RRAA_REGULAR", "RRAA_IRREGULAR", "RRAA_TOTAL", "ESTIMACION", "INFLACION", "CRECIMIENTO_PIB", "DESEMPLEO"]
        columnas_numericas = [col for col in columnas_numericas_posibles if col in df.columns]
        
        # Convertir a numérico (por si hay strings mezclados)
        for col in columnas_numericas:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        print(f"Columnas numéricas convertidas y NaN rellenados con 0")

        # Verificar valores negativos (solo en columnas donde no deberían existir)
        # Excluir CRECIMIENTO_PIB porque puede ser negativo (recesión económica)
        columnas_sin_negativos_posibles = ["CENSO AJUSTADO", "RRAA_REGULAR", "RRAA_IRREGULAR", "RRAA_TOTAL", "ESTIMACION", "INFLACION", "DESEMPLEO"]
        columnas_sin_negativos = [col for col in columnas_sin_negativos_posibles if col in df.columns]
        if columnas_sin_negativos:
            negativos = (df[columnas_sin_negativos] < 0).any(axis=1)
        else:
            negativos = pd.Series([False] * len(df))
        
        if negativos.any():
            print(f"\nExisten {negativos.sum()} filas con valores negativos:")
            for col in columnas_sin_negativos: 
                num_negativos = (df[col] < 0).sum()
                if num_negativos > 0:
                    print(f"  - {col}: {num_negativos} valores negativos")
            
            # Convertir todos los valores negativos a su valor absoluto (excepto CRECIMIENTO_PIB)
            df[columnas_sin_negativos] = df[columnas_sin_negativos].abs()
            print(f"\nValores negativos convertidos a positivos (valor absoluto)")
            print(f"Nota: CRECIMIENTO_PIB puede contener valores negativos (recesión económica)")
        
        # ============================================================
        # 1. MANEJO COMPLETO DE NULOS
        # ============================================================
        
        # Contar nulos antes del tratamiento
        nulos_antes = df.isnull().sum().sum()
        print(f"\nTotal valores nulos antes del tratamiento: {nulos_antes}")
        
        # Nulos en columnas categóricas
        cols_categoricas = ['SEXO', 'EDAD', 'PAIS', 'REGION']
        for col in cols_categoricas:
            if col in df.columns:
                nulos_col = df[col].isnull().sum()
                if nulos_col > 0:
                    print(f"  - {col}: {nulos_col} nulos rellenados con 'IGNORADO'")
                    df[col] = df[col].fillna('IGNORADO')
        
        # Nulos en columnas numéricas adicionales (por si acaso)
        for col in columnas_numericas:
            if col in df.columns:
                nulos_col = df[col].isnull().sum()
                if nulos_col > 0:
                    print(f"  - {col}: {nulos_col} nulos rellenados con 0")
                    df[col] = df[col].fillna(0)
        
        # ============================================================
        # 2. AGREGAR CÓDIGO ISO DE PAÍS (estándar internacional)
        # ============================================================
        
        iso_map = {
            "ARGENTINA": "AR",
            "BOLIVIA": "BO",
            "BRASIL": "BR",
            "COLOMBIA": "CO",
            "ECUADOR": "EC",
            "PERÚ": "PE",
            "VENEZUELA": "VE",
            "PARAGUAY": "PY",
            "URUGUAY": "UY",
            "ESTADOS UNIDOS": "US",
            "ESPAÑA": "ES",
            "MÉXICO": "MX",
            "CUBA": "CU",
            "R. DOMINICANA": "DO",
            "HAITÍ": "HT",
            "CHINA": "CN",
            "ALEMANIA": "DE",
            "FRANCIA": "FR",
            "ITALIA": "IT",
            "OTRO PAÍS": "OT",
            "PAÍS IGNORADO": "XX"
        }
        
        df['PAIS_ISO'] = df['PAIS'].map(iso_map)
        # Si hay países no mapeados, asignar 'XX'
        df['PAIS_ISO'] = df['PAIS_ISO'].fillna('XX')
        print(f"\nColumna PAIS_ISO creada (códigos ISO de 2 letras)")
        
        # ============================================================
        # 3. CODIFICACIÓN CATEGÓRICA (sin one-hot encoding)
        # ============================================================
        
        # Se mantiene PAIS_CODIGO (numérico) y PAIS_ISO (texto)
        # No se generan columnas dummy adicionales
        
        # ============================================================
        # 4. LIMPIEZA ADICIONAL DEL DATASET
        # ============================================================
        
        # Eliminar filas completamente duplicadas
        filas_antes = len(df)
        df = df.drop_duplicates()
        filas_despues = len(df)
        if filas_antes != filas_despues:
            print(f"\nFilas duplicadas eliminadas: {filas_antes - filas_despues}")
        
        # Estandarizar nombres de columnas (quitar espacios, tildes, mayúsculas)
        df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('Á','A').str.replace('É','E').str.replace('Í','I').str.replace('Ó','O').str.replace('Ú','U').str.replace('Ñ','N').str.replace('ñ','n').str.replace('á','a').str.replace('é','e').str.replace('í','i').str.replace('ó','o').str.replace('ú','u')
        
        # Renombrar columna de año si existe con tilde o espacio
        for col in df.columns:
            if 'ANO' in col.upper() or 'AÑO' in col.upper():
                df.rename(columns={col: 'ANIO'}, inplace=True)
                print(f"\nColumna renombrada: {col} -> ANIO")
                break
        
        # Ordenar columnas lógicamente
        cols_primero = []
        for c in ['SEXO','EDAD','EDAD_NUMERICA','PAIS','PAIS_CODIGO','PAIS_ISO','ANIO','CODREGEO','REGION']:
            if c in df.columns:
                cols_primero.append(c)
        
        otras_cols = [c for c in df.columns if c not in cols_primero]
        df = df[cols_primero + otras_cols]
        
        # ============================================================
        # 5. INFORME FINAL
        # ============================================================
        
        nulos_despues = df.isnull().sum().sum()
        print(f"\n{'='*60}")
        print("RESUMEN DE LIMPIEZA")
        print(f"{'='*60}")
        print(f"Filas procesadas: {len(df)}")
        print(f"Columnas finales: {len(df.columns)}")
        print(f"Valores nulos restantes: {nulos_despues}")
        print(f"Columnas generadas:")
        print(f"  - EDAD_NUMERICA (edad como promedio numérico)")
        print(f"  - PAIS_CODIGO (código numérico de país)")
        print(f"  - PAIS_ISO (código ISO de 2 letras)")
        print(f"  - REGION (mantenida como columna categórica original)")
        print(f"  - PAIS_CODIGO y PAIS_ISO (códigos de país)")
        print(f"{'='*60}\n")
        
        return df


def main():
    """
    Ejecuta la limpieza sobre el archivo 8. baseregiones.csv
    """
    import os
    
    archivo_entrada = "8. baseregiones.csv"
    archivo_salida = "8. baseregiones_limpio.csv"
    
    if not os.path.exists(archivo_entrada):
        print(f"ERROR: No se encontró el archivo {archivo_entrada}")
        print("Buscando archivos CSV similares...")
        csv_files = [f for f in os.listdir('.') if f.lower().endswith('.csv') and 'base' in f.lower()]
        if csv_files:
            print(f"Archivos encontrados: {csv_files}")
        return
    
    print(f"Cargando archivo: {archivo_entrada}")
    
    # Cargar CSV manejando comillas y saltos de línea
    df = pd.read_csv(archivo_entrada, quotechar='"', skipinitialspace=True)
    
    print(f"Dataset cargado: {df.shape[0]} filas x {df.shape[1]} columnas\n")
    
    # Mostrar info inicial
    print("INFO INICIAL:")
    print(df.head())
    print(f"\nColumnas: {list(df.columns)}")
    
    # Limpiar
    df_limpio = limpiar_datos(df)
    
    # Guardar
    df_limpio.to_csv(archivo_salida, index=False)
    print(f"Dataset limpio guardado en: {archivo_salida}")
    print(f"Dimensiones finales: {df_limpio.shape[0]} filas x {df_limpio.shape[1]} columnas")


if __name__ == "__main__":
    main()
