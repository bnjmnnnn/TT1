"""Entrenamiento y evaluacion de modelos (Fases 4-6 de desarrollo_modelo.md).

Protocolo, identico para cualquier dataset del proyecto:
  1. Hold-out temporal: test = ultimo anio disponible (TEST_YEAR).
  2. Baseline (Regresion Lineal Multiple) SIEMPRE se entrena y mide primero.
  3. XGBoost con configuracion conservadora + grilla acotada (LOOCV, N<80).
  4. Metricas RMSE/MAE/R2 en train y test -> data/outputs/metricas.csv.
  5. Prueba anti-leakage del target permutado (Fase 5.8).
  6. Exportacion de modelos + metadatos y verificacion de recarga.

Uso:  python src/train.py
"""
import json
from datetime import date

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import config

DATASET_NAME = "region"

# Grilla maxima permitida (108 combinaciones, regla dura de la Fase 4.3)
XGB_GRID = {
    "max_depth": [2, 3, 4],
    "learning_rate": [0.03, 0.05, 0.1],
    "n_estimators": [200, 300, 500],
    "min_child_weight": [1, 3, 5],
}


def make_baseline():
    # El escalado solo es necesario para la regresion lineal
    return make_pipeline(StandardScaler(), LinearRegression())


def make_xgboost():
    # Configuracion conservadora para N pequeno: arboles poco profundos,
    # regularizacion activa (Fase 4.2)
    return XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_lambda=1.0,
        random_state=config.SEED,
        n_jobs=-1,
    )


def evaluate(model, X, y):
    """Las tres metricas que exige el informe (RMSE via sqrt: compatible
    con cualquier version de scikit-learn)."""
    pred = model.predict(X)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "MAE": float(mean_absolute_error(y, pred)),
        "R2": float(r2_score(y, pred)),
    }


def temporal_split(df):
    """Hold-out temporal: nunca un split aleatorio en datos de panel."""
    train = df[df["ANIO"] < config.TEST_YEAR]
    test = df[df["ANIO"] == config.TEST_YEAR]
    assert len(test) > 0, f"No hay filas para TEST_YEAR={config.TEST_YEAR}"
    return train, test


def collect_metrics(rows, model_name, model, X_tr, y_tr, X_te, y_te):
    for split, X, y in [("train", X_tr, y_tr), ("test", X_te, y_te)]:
        rows.append({
            "dataset": DATASET_NAME,
            "model": model_name,
            "split": split,
            **evaluate(model, X, y),
            "n_rows": len(X),
            "seed": config.SEED,
            "date": date.today().isoformat(),
        })


def permuted_target_check(X_tr, y_tr, X_te, y_te):
    """Fase 5.8: con el target barajado, el R2 de test debe ser ~0 o negativo.
    Un valor alto delataria leakage en las features."""
    rng = np.random.default_rng(config.SEED)
    model = make_xgboost()
    model.fit(X_tr, rng.permutation(y_tr.values))
    r2 = r2_score(y_te, model.predict(X_te))
    assert r2 < 0.3, f"Posible leakage: R2 con target permutado = {r2:.3f}"
    return r2


def export_model(model, name, features, hyperparams, test_metrics):
    """Exporta modelo + metadatos (_meta.json) para reproducibilidad."""
    import sklearn
    import xgboost

    if isinstance(model, XGBRegressor):
        path = config.OUTPUTS_DIR / f"{name}.json"
        model.save_model(path)
    else:
        path = config.OUTPUTS_DIR / f"{name}.pkl"
        joblib.dump(model, path)

    meta = {
        "date": date.today().isoformat(),
        "dataset": DATASET_NAME,
        "features": features,
        "target": config.TARGET,
        "test_year": config.TEST_YEAR,
        "hyperparams": hyperparams,
        "test_metrics": test_metrics,
        "seed": config.SEED,
        "versions": {"xgboost": xgboost.__version__,
                     "scikit-learn": sklearn.__version__},
    }
    meta_path = config.OUTPUTS_DIR / f"{name}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def verify_reload(path, X_te, expected_pred):
    """El modelo exportado debe reproducir exactamente las predicciones."""
    reloaded = XGBRegressor()
    reloaded.load_model(path)
    assert np.allclose(reloaded.predict(X_te), expected_pred), \
        "El modelo recargado no reproduce las predicciones de test"


def main():
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(config.DATASET_REGION)

    train, test = temporal_split(df)
    X_tr, y_tr = train[config.FEATURES], train[config.TARGET]
    X_te, y_te = test[config.FEATURES], test[config.TARGET]
    print(f"Split temporal: train {sorted(train['ANIO'].unique())} "
          f"(N={len(train)}) | test {config.TEST_YEAR} (N={len(test)})")
    print(f"Features ({len(config.FEATURES)}): {config.FEATURES}\n")

    metrics_rows = []

    # --- 1) Baseline lineal: SIEMPRE primero, es la vara ---
    baseline = make_baseline()
    baseline.fit(X_tr, y_tr)
    collect_metrics(metrics_rows, "linear_regression", baseline,
                    X_tr, y_tr, X_te, y_te)
    print("Baseline lineal entrenado")

    # --- 2) XGBoost con grilla acotada y LOOCV (N_train < 80) ---
    cv = LeaveOneOut()
    search = GridSearchCV(make_xgboost(), XGB_GRID, cv=cv,
                          scoring="neg_root_mean_squared_error", n_jobs=-1)
    search.fit(X_tr, y_tr)
    xgb_best = search.best_estimator_
    print(f"XGBoost: mejores hiperparametros {search.best_params_}")
    collect_metrics(metrics_rows, "xgboost", xgb_best, X_tr, y_tr, X_te, y_te)

    # --- 3) Metricas -> CSV (fuente unica para el informe) ---
    metrics = pd.DataFrame(metrics_rows)
    metrics_path = config.OUTPUTS_DIR / "metricas.csv"
    metrics.to_csv(metrics_path, index=False)
    print(f"\n{metrics.to_string(index=False)}")

    # Diagnostico rapido de sobreajuste (brecha R2 train vs test, Fase 5.2)
    for model_name in metrics["model"].unique():
        m = metrics[metrics["model"] == model_name].set_index("split")["R2"]
        gap = m["train"] - m["test"]
        flag = "  <-- brecha > 0.15: senal de sobreajuste" if gap > 0.15 else ""
        print(f"Brecha R2 train-test {model_name}: {gap:.3f}{flag}")

    # --- 4) Prueba anti-leakage (target permutado) ---
    r2_perm = permuted_target_check(X_tr, y_tr, X_te, y_te)
    print(f"\nPrueba target permutado: R2 test = {r2_perm:.3f} (esperado ~0) OK")

    # --- 5) Predicciones de test por region (insumo grafico Fase 5.3) ---
    preds = test[["CODREGEO", "REGION", "ANIO", config.TARGET]].copy()
    preds["pred_linear"] = baseline.predict(X_te)
    preds["pred_xgboost"] = xgb_best.predict(X_te)
    preds_path = config.OUTPUTS_DIR / f"predicciones_test_{DATASET_NAME}.csv"
    preds.to_csv(preds_path, index=False)

    # --- 6) Exportacion y verificacion de recarga ---
    xgb_test_metrics = evaluate(xgb_best, X_te, y_te)
    xgb_path = export_model(xgb_best, f"xgb_{DATASET_NAME}", config.FEATURES,
                            search.best_params_, xgb_test_metrics)
    verify_reload(xgb_path, X_te, xgb_best.predict(X_te))
    export_model(baseline, f"baseline_{DATASET_NAME}", config.FEATURES,
                 {"model": "LinearRegression + StandardScaler"},
                 evaluate(baseline, X_te, y_te))
    print(f"\nModelos exportados en {config.OUTPUTS_DIR} "
          "(recarga verificada para XGBoost)")

    return metrics


if __name__ == "__main__":
    main()
