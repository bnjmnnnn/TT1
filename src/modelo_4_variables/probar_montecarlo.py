"""Modelo 4, ronda 3: tests de Monte Carlo.

Para que sirven (y para que NO). No suben el R2: el R2 puntual no se mueve. Lo que
hacen es responder tres preguntas que hoy no tienen respuesta:

  1. TEST DE PERMUTACION (Monte Carlo sobre la etiqueta)
     Se baraja la redistribucion ENTRE REGIONES DENTRO DE CADA ANIO (asi se conserva
     la estructura temporal y solo se destruye la asociacion region <-> cambio) y se
     reentrena el modelo. Si el modelo aprendio algo real, el R2 real debe quedar muy
     por encima de la nube de R2 barajados. De ahi sale un p-valor.
     Es el mismo tipo de test que ya tiene el modelo v2 (test_permutacion_v2.csv).

  2. BOOTSTRAP SOBRE LAS REGIONES DE PRUEBA (5.000 replicas)
     Se remuestrean las 16 regiones con reemplazo y se recalcula el skill del modelo
     y el de la persistencia en la MISMA muestra (pareado). Da el intervalo de
     confianza del skill y la probabilidad de que el modelo le gane a no hacer nada.
     Ojo: con 16 regiones el intervalo es ancho; eso es informacion, no un defecto.

  3. MONTE CARLO SOBRE LAS REGIONES DE ENTRENAMIENTO
     Se entrena muchas veces dejando fuera 4 regiones al azar (y evaluando justo en
     esas 4, en 2023). Mide cuanto depende el resultado de QUE regiones cayeron en el
     entrenamiento: es la sensibilidad al muestreo, que con 16 regiones es la
     incertidumbre dominante.

Uso:  python src/modelo_4_variables/probar_montecarlo.py
      -> outputs/montecarlo_observaciones.md
      -> outputs/montecarlo_permutacion.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parents[1]
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(ROOT / "src" / "common"))
from config import SEED  # noqa: E402
from explorar_variables import cargar_panel  # noqa: E402
from probar_ronda2 import agregar_derivadas  # noqa: E402

OUT_DIR = AQUI / "outputs"
FEATURES = ["sol_otorgadas_lag1", "sol_share_lag1", "sol_total_lag1",
            "pct_peru_lag1", "mean_age_lag1", "rate_lag2"]
N_PERM, N_BOOT, N_SUBSET = 500, 5000, 300


def gb():
    return GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=2,
                                     min_samples_leaf=2, random_state=SEED)


def skill(y, pred, ref) -> float:
    mse, mse_ref = ((y - pred) ** 2).mean(), ((y - ref) ** 2).mean()
    return float(1 - mse / mse_ref)


def test_permutacion(p, anio):
    """Baraja el target entre regiones dentro de cada anio de entrenamiento."""
    rng = np.random.default_rng(SEED)
    tr, te = p[p["ANIO"] < anio], p[p["ANIO"] == anio]
    X_tr, X_te = tr[FEATURES].to_numpy(), te[FEATURES].to_numpy()
    y_te = te["redistribucion"].to_numpy()

    m = gb().fit(X_tr, tr["redistribucion"].to_numpy())
    r2_real = float(r2_score(y_te, m.predict(X_te)))

    r2s = []
    for _ in range(N_PERM):
        y_perm = tr.copy()
        # permutar DENTRO de cada anio: se conserva el patron temporal y solo se
        # rompe la asociacion entre las features y el cambio de peso
        y_perm["redistribucion"] = (y_perm.groupby("ANIO")["redistribucion"]
                                    .transform(lambda s: rng.permutation(s.to_numpy())))
        mp = gb().fit(X_tr, y_perm["redistribucion"].to_numpy())
        r2s.append(float(r2_score(y_te, mp.predict(X_te))))
    r2s = np.array(r2s)
    p_valor = float((r2s >= r2_real).mean())
    return {"anio_test": anio, "r2_real": r2_real, "r2_perm_media": float(r2s.mean()),
            "r2_perm_desv": float(r2s.std()), "r2_perm_max": float(r2s.max()),
            "p_valor": p_valor, "n_permutaciones": N_PERM}


def bootstrap_regiones(p, anio):
    """Remuestreo pareado de las 16 regiones de prueba."""
    rng = np.random.default_rng(SEED)
    tr, te = p[p["ANIO"] < anio], p[p["ANIO"] == anio]
    m = gb().fit(tr[FEATURES].to_numpy(), tr["redistribucion"].to_numpy())
    pred = m.predict(te[FEATURES].to_numpy())
    y = te["redistribucion"].to_numpy()
    ref = np.zeros_like(y)
    n = len(y)

    skills, r2s = [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        skills.append(skill(y[idx], pred[idx], ref[idx]))
        r2s.append(float(r2_score(y[idx], pred[idx])))
    skills, r2s = np.array(skills), np.array(r2s)
    return {
        "anio_test": anio,
        "skill_ic95": (float(np.percentile(skills, 2.5)), float(np.percentile(skills, 97.5))),
        "r2_ic95": (float(np.percentile(r2s, 2.5)), float(np.percentile(r2s, 97.5))),
        "p_skill_positivo": float((skills > 0).mean()),
        "skill_mediana": float(np.median(skills)),
    }


def montecarlo_regiones_entrenamiento(p, anio):
    """Deja 4 regiones fuera del entrenamiento y evalua en ellas (2023)."""
    rng = np.random.default_rng(SEED)
    tr, te = p[p["ANIO"] < anio], p[p["ANIO"] == anio]
    regiones = sorted(p["CODREGEO"].unique())
    r2s, skills = [], []
    for _ in range(N_SUBSET):
        fuera = rng.choice(regiones, 4, replace=False)
        tr_s = tr[~tr["CODREGEO"].isin(fuera)]
        te_s = te[te["CODREGEO"].isin(fuera)]
        if len(te_s) == 0:
            continue
        m = gb().fit(tr_s[FEATURES].to_numpy(), tr_s["redistribucion"].to_numpy())
        pred = m.predict(te_s[FEATURES].to_numpy())
        y = te_s["redistribucion"].to_numpy()
        skills.append(skill(y, pred, np.zeros_like(y)))
    skills = np.array(skills)
    return {"anio_test": anio, "skill_mediana": float(np.median(skills)),
            "skill_ic95": (float(np.percentile(skills, 2.5)), float(np.percentile(skills, 97.5))),
            "p_skill_positivo": float((skills > 0).mean()), "n": len(skills)}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = agregar_derivadas(cargar_panel())
    p["redistribucion"] = p["rate"] - p["rate_lag1"]

    print("1) test de permutacion...")
    perms = [test_permutacion(p, a) for a in (2023, 2022)]
    pd.DataFrame(perms).to_csv(OUT_DIR / "montecarlo_permutacion.csv", index=False)
    for r in perms:
        print(f"   {r['anio_test']}: R2 real {r['r2_real']:+.3f} | "
              f"barajado {r['r2_perm_media']:+.3f} +- {r['r2_perm_desv']:.3f} "
              f"(max {r['r2_perm_max']:+.3f}) | p = {r['p_valor']:.4f}")

    print("2) bootstrap de regiones de prueba...")
    boots = [bootstrap_regiones(p, a) for a in (2023, 2022)]
    for b in boots:
        print(f"   {b['anio_test']}: skill mediana {b['skill_mediana']:+.3f} "
              f"IC95 [{b['skill_ic95'][0]:+.3f}; {b['skill_ic95'][1]:+.3f}] | "
              f"P(skill>0) = {b['p_skill_positivo']*100:.1f}%")

    print("3) montecarlo sobre regiones de entrenamiento...")
    subs = [montecarlo_regiones_entrenamiento(p, a) for a in (2023, 2022)]
    for s in subs:
        print(f"   {s['anio_test']}: skill mediana {s['skill_mediana']:+.3f} "
              f"IC95 [{s['skill_ic95'][0]:+.3f}; {s['skill_ic95'][1]:+.3f}] | "
              f"P(skill>0) = {s['p_skill_positivo']*100:.1f}%")

    L = ["# Modelo 4: tests de Monte Carlo", "",
         "Generado por `src/modelo_4_variables/probar_montecarlo.py`.", "",
         "**Para qué sirven:** no suben el R² puntual (eso no lo hace ningún test). "
         "Responden si el resultado es distinguible del azar y cuánta incertidumbre "
         "tiene. Es el paso que faltaba.", "",
         "## 1. Test de permutación", "",
         f"Se barajó la redistribución entre regiones **dentro de cada año** "
         f"({N_PERM} permutaciones), conservando la estructura temporal, y se reentrenó "
         "el modelo con la etiqueta barajada. Si el modelo aprendió algo real, el R² "
         "verdadero debe quedar muy por encima de la nube de R² barajados.", "",
         "| test | R² real | R² barajado (media ± desv.) | R² barajado (máximo) | p-valor |",
         "|---|---|---|---|---|"]
    for r in perms:
        L.append(f"| {r['anio_test']} | **{r['r2_real']:+.3f}** | "
                 f"{r['r2_perm_media']:+.3f} ± {r['r2_perm_desv']:.3f} | "
                 f"{r['r2_perm_max']:+.3f} | **{r['p_valor']:.4f}** |")
    L += ["", "## 2. Bootstrap sobre las regiones de prueba", "",
          f"{N_BOOT} remuestreos de las 16 regiones con reemplazo, pareado (el modelo y "
          "la persistencia se miden en la misma muestra).", "",
          "| test | skill (mediana) | intervalo de confianza 95% | P(skill > 0) |", "|---|---|---|---|"]
    for b in boots:
        L.append(f"| {b['anio_test']} | {b['skill_mediana']:+.3f} | "
                 f"[{b['skill_ic95'][0]:+.3f}; {b['skill_ic95'][1]:+.3f}] | "
                 f"{b['p_skill_positivo']*100:.1f}% |")
    L += ["", "## 3. Monte Carlo sobre las regiones de entrenamiento", "",
          f"{N_SUBSET} corridas dejando **4 regiones fuera** del entrenamiento (y "
          "evaluando en ellas, en el año de prueba). Mide cuánto depende el resultado "
          "de **qué regiones** cayeron en el entrenamiento.", "",
          "| test | skill (mediana) | intervalo de confianza 95% | P(skill > 0) | corridas |",
          "|---|---|---|---|---|"]
    for s in subs:
        L.append(f"| {s['anio_test']} | {s['skill_mediana']:+.3f} | "
                 f"[{s['skill_ic95'][0]:+.3f}; {s['skill_ic95'][1]:+.3f}] | "
                 f"{s['p_skill_positivo']*100:.1f}% | {s['n']} |")
    L += ["", "## Lectura", "",
          "1. **El resultado no es azar** si el p-valor de la permutación es muy bajo: "
          "significa que ninguna de las etiquetas barajadas alcanza el R² real.",
          "2. **El intervalo del bootstrap es la incertidumbre honesta**: si el extremo "
          "inferior sigue sobre 0, el modelo le gana a no hacer nada incluso en el peor "
          "caso razonable.",
          "3. **La incertidumbre dominante es qué regiones se usaron**: si el skill "
          "cambia mucho al dejar 4 regiones fuera, el número puntual depende del "
          "muestreo y hay que reportarlo como un rango, no como un valor único.",
          "4. **Ningún test sube el R².** El camino para mejorarlo de verdad sigue "
          "siendo más años de datos (extender el panel), no más ajuste.", ""]
    (OUT_DIR / "montecarlo_observaciones.md").write_text("\n".join(L))
    print(f"\n-> {OUT_DIR / 'montecarlo_observaciones.md'}")


if __name__ == "__main__":
    main()
