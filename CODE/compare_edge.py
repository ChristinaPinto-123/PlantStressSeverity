"""Why SNV + LDA on the edge?  Same data, same group-aware nested CV, same preprocessing budget
(smooth -> [SNV] -> SelectKBest(k=8) + climate); only the preprocessing flag / classifier changes.

    python compare_edge.py            # synthetic stand-in data (same as run_all.py default)
    python compare_edge.py --csv d.csv --n_bands 52

Writes results/edge_comparison.json, which the dashboard shows as the table at the top.
NOTE: scikit-learn's GradientBoostingClassifier stands in for XGBoost here (same model family,
no extra dependency). If you pip-install xgboost you can add XGBClassifier to CANDIDATES below.
"""
import argparse
import json
import os

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from config import EDGE_K, SEED
from data import load_csv, make_synthetic
from evaluate import cv_group, latency_ms, model_size_kb, perturb
from models import N_CLIMATE, _pre, _spectral_pipe


def make(n_bands, nc, clf, use_snv=True, scale=True):
    steps = [("pre", _pre(n_bands, _spectral_pipe(min(EDGE_K, n_bands), use_snv), True, nc))]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("clf", clf))
    return Pipeline(steps)


CANDIDATES = {  # name -> (classifier factory, use_snv, scale, note)
    "SNV + LDA (proposed edge)": (lambda: LinearDiscriminantAnalysis(solver="svd", priors=np.full(4, 0.25)), True, True),
    "LDA, no SNV": (lambda: LinearDiscriminantAnalysis(solver="svd", priors=np.full(4, 0.25)), False, True),
    "SNV + Logistic regression": (lambda: LogisticRegression(max_iter=2000, class_weight="balanced"), True, True),
    "SNV + SVM (RBF)": (lambda: SVC(probability=True, class_weight="balanced", random_state=SEED), True, True),
    "SNV + k-NN (k=5)": (lambda: KNeighborsClassifier(5), True, True),
    "SNV + Decision tree (d6)": (lambda: DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=SEED), True, False),
    "SNV + Random forest": (lambda: RandomForestClassifier(200, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED), True, False),
    "SNV + Gradient boosting (XGBoost-style)": (lambda: GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=SEED), True, False),
    "Gradient boosting, no SNV": (lambda: GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=SEED), False, False),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--n_bands", type=int, default=52)
    a = ap.parse_args()
    if a.csv:
        X, y, meta = load_csv(a.csv, a.n_bands)
    else:
        X, y, meta = make_synthetic()
    nb = a.n_bands
    nc = X.shape[1] - nb
    plants = meta["plant_id"].to_numpy()

    out = {"protocol": "group-aware (plant) stratified 5-fold CV, band selection inside each fold",
           "data": "csv" if a.csv else "synthetic stand-in (make_synthetic)", "rows": {}}
    for name, (fac, snv_on, sc) in CANDIDATES.items():
        m = make(nb, nc, fac(), snv_on, sc)
        r = cv_group(m, X, y, plants)
        row = {"macro_f1": r["macro_f1"], "mcc": r["mcc"], "ece": r["ece"]}
        for tag, kind, s in (("gain_0.10", "gain", 0.10), ("contact_0.08", "contact", 0.08), ("noise_0.005", "noise", 0.005)):
            row[tag] = cv_group(m, X, y, plants, test_transform=lambda Xt, k=kind, s=s: perturb(Xt, nb, k, s, np.random.default_rng(SEED)))["macro_f1"]
        fitted = make(nb, nc, fac(), snv_on, sc).fit(X, y)
        row["size_kb"] = model_size_kb(fitted)
        row["latency_ms_host"] = latency_ms(fitted, X[0], repeats=100)
        out["rows"][name] = row
        print(f"{name:42s} F1={row['macro_f1']:.3f} MCC={row['mcc']:.3f} gain={row['gain_0.10']:.3f} "
              f"contact={row['contact_0.08']:.3f} noise={row['noise_0.005']:.3f} {row['size_kb']:.1f}kB {row['latency_ms_host']:.2f}ms")
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/edge_comparison.json", "w"), indent=2, default=float)
