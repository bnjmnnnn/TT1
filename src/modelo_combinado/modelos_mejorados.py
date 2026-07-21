"""Version mejorada: XGBoost + Regresion Logistica bien especificada.

  1. PAIS_CODIGO y CODREGEO -> one-hot para la logistica (nominales).
  2. Desbalance 91/9 -> class_weight='balanced' / scale_pos_weight.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from modelos_basicos import FEATURES_BASE, FEATURES_CENSO, preparar, split_temporal

AQUI = Path(__file__).resolve().parent
SEED = 42
CATEGORICAS = ["PAIS_CODIGO", "CODREGEO"]


def hacer_logistica(cols):
    numericas = [c for c in cols if c not in CATEGORICAS]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
        ("num", StandardScaler(), numericas),
    ])
    return Pipeline([
        ("pre", pre),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced",
                                     random_state=SEED)),
    ])


def hacer_xgboost(y_train):
    peso = (y_train == 0).sum() / (y_train == 1).sum()
    return XGBClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=5,
        min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=peso, eval_metric="logloss", random_state=SEED)


def umbral_optimo(modelo, X_val, y_val):
    """Umbral que maximiza F1 en validacion (nunca en test)."""
    proba = modelo.predict_proba(X_val)[:, 1]
    prec, rec, thr = precision_recall_curve(y_val, proba)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return float(thr[np.argmax(f1[:-1])])


def evaluar_cv(nombre, construir, X, y):
    f1s, aucs = [], []
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
        modelo = construir(y.iloc[tr]) if nombre == "xgboost" else construir(list(X.columns))
        modelo.fit(X.iloc[tr], y.iloc[tr])
        proba = modelo.predict_proba(X.iloc[va])[:, 1]
        f1s.append(f1_score(y.iloc[va], proba > 0.5))
        aucs.append(roc_auc_score(y.iloc[va], proba))
    return np.mean(f1s), np.std(f1s), np.mean(aucs), np.std(aucs)


def main():
    df = preparar()

    resultados = []
    for set_nombre, cols in (("base", FEATURES_BASE),
                             ("base + censo", FEATURES_BASE + FEATURES_CENSO)):
        X_tr, X_te, y_tr, y_te = split_temporal(df, cols)
        # fit/val: split aleatorio, pero SOLO dentro de los anios de train
        # (< 2023) -- no toca el holdout, solo calibra el umbral.
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_tr, y_tr, test_size=0.25, random_state=SEED, stratify=y_tr)

        for nombre, construir in (("regresion_logistica", hacer_logistica),
                                  ("xgboost", hacer_xgboost)):
            f1_cv, f1_sd, auc_cv, auc_sd = evaluar_cv(nombre, construir, X_tr, y_tr)

            modelo = (construir(y_fit) if nombre == "xgboost"
                      else construir(cols))
            modelo.fit(X_fit, y_fit)
            thr = umbral_optimo(modelo, X_val, y_val)
            proba_te = modelo.predict_proba(X_te)[:, 1]
            pred_te = proba_te > thr

            resultados.append({
                "modelo": nombre, "features": set_nombre, "umbral": round(thr, 3),
                "f1_cv_mean": f1_cv, "f1_cv_std": f1_sd,
                "auc_cv_mean": auc_cv, "auc_cv_std": auc_sd,
                "f1_test": f1_score(y_te, pred_te),
                "auc_test": roc_auc_score(y_te, proba_te),
            })
            r = resultados[-1]
            print(f"{nombre:20s} [{set_nombre:12s}] "
                  f"CV f1={f1_cv:.3f}+-{f1_sd:.3f} auc={auc_cv:.3f}+-{auc_sd:.3f} | "
                  f"test f1={r['f1_test']:.3f} auc={r['auc_test']:.3f} thr={thr:.2f}")

    res = pd.DataFrame(resultados)
    res.to_csv(AQUI / "resultados_mejorados.csv", index=False)
    print(f"\n-> {AQUI / 'resultados_mejorados.csv'}")


if __name__ == "__main__":
    main()
