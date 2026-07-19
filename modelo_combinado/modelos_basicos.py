"""Modelos basicos sobre dataset_combinado_enriquecido.

  Target: TIPO_MIGRACION = 1 si la tasa de irregularidad del grupo > 0.5
  Se evalua sobre test separado y se excluyen filas con RRAA_TOTAL == 0
  (sin registros no hay tasa que clasificar).

Uso:  python modelos_basicos.py  ->  resultados_modelos.csv
"""
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from limpiar_datos import limpiar_datos

AQUI = Path(__file__).resolve().parent
SEED = 42

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
    df = pd.read_csv(AQUI / "dataset_combinado_enriquecido.csv")
    limpiar_datos(df)

    n0 = len(df)
    df = df[df["RRAA_TOTAL"] > 0].copy()
    print(f"\nFilas sin RRAA (target indefinido) excluidas: {n0 - len(df):,} "
          f"-> quedan {len(df):,}")

    df["TASA_IRREGULARIDAD"] = df["RRAA_IRREGULAR"] / (df["RRAA_TOTAL"] + 1e-10)
    df["TIPO_MIGRACION"] = (df["TASA_IRREGULARIDAD"] > 0.5).astype(int)
    print(f"Distribucion del target: {df['TIPO_MIGRACION'].value_counts().to_dict()}")
    return df


def main():
    df = preparar()
    y = df["TIPO_MIGRACION"]

    resultados = []
    for set_nombre, cols in (("base", FEATURES_BASE),
                             ("base + censo", FEATURES_BASE + FEATURES_CENSO)):
        X = df[cols]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=SEED, stratify=y)
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
    res.to_csv(AQUI / "resultados_modelos.csv", index=False)
    print(f"\n-> {AQUI / 'resultados_modelos.csv'}")


if __name__ == "__main__":
    main()
