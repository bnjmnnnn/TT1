"""Tuning sistematico de XGBoost — el modelo final (F1 ~0.616 en test).

  - RandomizedSearchCV (60 combinaciones, CV 5-fold) sobre AUC-PR
    (average_precision), no AUC-ROC: con 9,3% de clase positiva el AUC-ROC
    no penaliza fallar en la clase rara.
  - n_estimators por early stopping contra validacion (max 2000).
  - scale_pos_weight para el desbalance; umbral calibrado en validacion.

Uso:  python modelos_xgboost_tuning.py
      -> resultados_tuning.csv + hiperparametros_tuning.json
       + curva_pr.png + xgb_tuneado.json
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_curve, roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     train_test_split)
from xgboost import XGBClassifier

from modelos_basicos import FEATURES_BASE, FEATURES_CENSO, preparar
from modelos_mejorados import umbral_optimo

AQUI = Path(__file__).resolve().parent
SEED = 42
N_ESTIMATORS_MAX = 2000
EARLY_STOPPING_ROUNDS = 30

ESPACIO_BUSQUEDA = {
    "max_depth": randint(3, 9),
    "min_child_weight": randint(1, 8),
    "gamma": uniform(0, 1),
    "reg_lambda": uniform(0.5, 4.5),
    "reg_alpha": uniform(0, 1),
    "learning_rate": loguniform(0.01, 0.15),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
}


def buscar_hiperparametros(X_tr, y_tr, scale_pos_weight):
    base = XGBClassifier(n_estimators=300, scale_pos_weight=scale_pos_weight,
                         eval_metric="aucpr", random_state=SEED, n_jobs=-1)
    busqueda = RandomizedSearchCV(
        base, ESPACIO_BUSQUEDA, n_iter=60, scoring="average_precision",
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
        random_state=SEED, n_jobs=-1, verbose=1)
    busqueda.fit(X_tr, y_tr)
    print(f"\nMejor AUC-PR (CV): {busqueda.best_score_:.4f}")
    print(f"Mejores hiperparametros: {busqueda.best_params_}")
    return busqueda.best_params_


def fit_con_early_stopping(params, X_fit, y_fit, X_val, y_val, scale_pos_weight):
    modelo = XGBClassifier(
        **params, n_estimators=N_ESTIMATORS_MAX,
        scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        random_state=SEED, n_jobs=-1)
    modelo.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    print(f"Early stopping: {modelo.best_iteration} arboles "
          f"(de {N_ESTIMATORS_MAX} maximos)")
    return modelo


def main():
    df = preparar()
    y = df["TIPO_MIGRACION"]
    cols = FEATURES_BASE + FEATURES_CENSO
    X = df[cols]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_tr, y_tr, test_size=0.25, random_state=SEED, stratify=y_tr)
    scale_pos_weight = (y_fit == 0).sum() / (y_fit == 1).sum()

    print("=== 1. Busqueda de hiperparametros (RandomizedSearchCV, AUC-PR) ===")
    mejores_params = buscar_hiperparametros(X_tr, y_tr, scale_pos_weight)

    print("\n=== 2. Fit final con early stopping ===")
    modelo = fit_con_early_stopping(mejores_params, X_fit, y_fit, X_val, y_val,
                                    scale_pos_weight)

    print("\n=== 3. Umbral calibrado en validacion ===")
    thr = umbral_optimo(modelo, X_val, y_val)
    print(f"Umbral optimo: {thr:.3f}")

    proba_te = modelo.predict_proba(X_te)[:, 1]
    pred_te = proba_te > thr
    resultado = {
        "modelo": "xgboost_tuneado", "features": "base + censo",
        "n_arboles": int(modelo.best_iteration), "umbral": round(thr, 3),
        "f1_test": f1_score(y_te, pred_te),
        "auc_roc_test": roc_auc_score(y_te, proba_te),
        "auc_pr_test": average_precision_score(y_te, proba_te),
    }
    print(f"\nTest: F1={resultado['f1_test']:.4f}  "
          f"AUC-ROC={resultado['auc_roc_test']:.4f}  "
          f"AUC-PR={resultado['auc_pr_test']:.4f}")

    previo = pd.read_csv(AQUI / "resultados_mejorados.csv").query(
        "modelo == 'xgboost' and features == 'base + censo'").iloc[0]
    comp = pd.DataFrame([
        {"version": "xgboost_mejorado (hiperparam. fijos)",
         "f1_test": previo["f1_test"], "auc_test": previo["auc_test"]},
        {"version": "xgboost_tuneado (RandomizedSearchCV + early stopping)",
         "f1_test": resultado["f1_test"], "auc_test": resultado["auc_roc_test"]},
    ])
    comp.to_csv(AQUI / "resultados_tuning.csv", index=False)
    print("\n== Comparacion ==")
    print(comp.to_string(index=False))

    # Persistir hiperparametros y umbral (antes solo quedaban en consola)
    (AQUI / "hiperparametros_tuning.json").write_text(json.dumps(
        {"mejores_params": {k: float(v) if not isinstance(v, int) else v
                            for k, v in mejores_params.items()},
         "n_arboles_early_stopping": int(modelo.best_iteration),
         "umbral": round(thr, 3), "scale_pos_weight": round(scale_pos_weight, 3),
         "seed": SEED, **{k: float(v) for k, v in resultado.items()
                          if isinstance(v, float)}}, indent=2))

    prec, rec, _ = precision_recall_curve(y_te, proba_te)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, label=f"XGBoost tuneado (AUC-PR={resultado['auc_pr_test']:.3f})")
    ax.axhline(y_te.mean(), color="gray", linestyle="--",
              label=f"Azar (prevalencia={y_te.mean():.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curva Precision-Recall (test)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(AQUI / "curva_pr.png", dpi=120)

    modelo.save_model(AQUI / "xgb_tuneado.json")
    print(f"\n-> {AQUI / 'resultados_tuning.csv'}")
    print(f"-> {AQUI / 'hiperparametros_tuning.json'}")
    print(f"-> {AQUI / 'curva_pr.png'}")
    print(f"-> {AQUI / 'xgb_tuneado.json'}")


if __name__ == "__main__":
    main()
