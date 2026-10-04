"""Modelos basicos sobre dataset_combinado_enriquecido.

  Target: TIPO_MIGRACION = 1 si la tasa de irregularidad del grupo > 0.5
  Split temporal (train < 2023, test = 2023), NUNCA aleatorio -- ver
  split_temporal() para el motivo (evitar fuga entre anios de la misma
  celda demografica). Se excluyen filas con RRAA_TOTAL == 0 (sin registros
  no hay tasa que clasificar).

Uso:  python modelos_basicos.py  ->  resultados_modelos.csv
"""
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from limpiar_datos import limpiar_datos

AQUI = Path(__file__).resolve().parent
OUT_DIR = AQUI / "outputs"
SEED = 42
TEST_YEAR = 2023  # holdout temporal, mismo protocolo que src/train_v2.py

FEATURES_BASE = ["SEXO", "EDAD_NUMERICA", "PAIS_CODIGO", "AÑO", "CODREGEO",
                 "CENSO AJUSTADO", "INFLACION", "CRECIMIENTO_PIB", "DESEMPLEO"]
FEATURES_CENSO = ["CENSO_MEAN_EDAD", "CENSO_MEAN_ESCOLARIDAD",
                  "CENSO_PCT_URBANO", "CENSO_PCT_LLEGADA_RECIENTE",
                  "CENSO_PCT_OCUPADO"]

MODELOS = {
    "regresion_logistica": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=SEED)),
    ]),
    "random_forest": RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=SEED),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=3, random_state=SEED),
}


def preparar():
    df = pd.read_csv(OUT_DIR / "dataset_combinado_enriquecido.csv")
    limpiar_datos(df)

    n0 = len(df)
    df = df[df["RRAA_TOTAL"] > 0].copy()
    print(f"\nFilas sin RRAA (target indefinido) excluidas: {n0 - len(df):,} "
          f"-> quedan {len(df):,}")

    df["TASA_IRREGULARIDAD"] = df["RRAA_IRREGULAR"] / (df["RRAA_TOTAL"] + 1e-10)
    df["TIPO_MIGRACION"] = (df["TASA_IRREGULARIDAD"] > 0.5).astype(int)
    print(f"Distribucion del target: {df['TIPO_MIGRACION'].value_counts().to_dict()}")
    return df


def split_temporal(df, cols):
    """Split temporal (train < TEST_YEAR, test = TEST_YEAR), NUNCA aleatorio.

    Un split aleatorio por fila deja la misma celda demografica (sexo x edad
    x pais x region) repartida entre train y test en anios distintos -- esas
    celdas son casi identicas en features y target (la tasa de irregularidad
    de un grupo persiste de un anio a otro, igual que el share regional en
    v2/conteo), asi que el modelo podia acertar por memorizar la celda en
    vez de generalizar. Con split temporal, 2023 completo queda fuera del
    entrenamiento.
    """
    train = df[df["AÑO"] < TEST_YEAR]
    test = df[df["AÑO"] == TEST_YEAR]
    return (train[cols], test[cols],
            train["TIPO_MIGRACION"], test["TIPO_MIGRACION"])


def main():
    df = preparar()

    resultados = []
    for set_nombre, cols in (("base", FEATURES_BASE),
                             ("base + censo", FEATURES_BASE + FEATURES_CENSO)):
        X_tr, X_te, y_tr, y_te = split_temporal(df, cols)
        for nombre, modelo in MODELOS.items():
            modelo.fit(X_tr, y_tr)
            pred = modelo.predict(X_te)
            proba = modelo.predict_proba(X_te)[:, 1]
            resultados.append({
                "modelo": nombre, "features": set_nombre,
                "n_features": len(cols),
                "accuracy": accuracy_score(y_te, pred),
                "f1": f1_score(y_te, pred),
                "auc": roc_auc_score(y_te, proba),
            })
            print(f"{nombre:22s} [{set_nombre:12s}] "
                  f"acc={resultados[-1]['accuracy']:.4f} "
                  f"f1={resultados[-1]['f1']:.4f} auc={resultados[-1]['auc']:.4f}")

    res = pd.DataFrame(resultados)
    res.to_csv(OUT_DIR / "resultados_modelos.csv", index=False)
    print(f"\n-> {OUT_DIR / 'resultados_modelos.csv'}")


if __name__ == "__main__":
    main()
