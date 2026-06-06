import polars as pl

def limpieza(df):
    # Elimina filas completamente nulas y edades inválidas
    df = df.drop_nulls()

    # Elimina columnas innecesarias de forma segura
    columnas_a_eliminar = ['sit_fuerza_trabajo', 'p40_cise_rec']
    df = df.drop(*columnas_a_eliminar, strict=False)

    return df

if __name__ == "__main__":
    df = pl.read_csv('CensoData.csv')
    df_limpio = limpieza(df)
    print(df_limpio)
