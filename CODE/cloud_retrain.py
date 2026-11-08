"""Cloud loop: retrain with newly labelled samples, validate on a held-out benchmark,
version the model, and only publish it if it does not regress.

    python cloud_retrain.py train.csv new_labelled.csv benchmark.csv 18
"""
import json
import sys
import time

import numpy as np
from sklearn.metrics import f1_score

from config import WL_AS7265X
from data import load_csv
from export_edge import export_header
from models import build_edge, build_gateway


def retrain(train_csv, new_csv, bench_csv, n_bands, prev_f1=0.0, tol=0.01):
    X1, y1, _ = load_csv(train_csv, n_bands)
    X2, y2, _ = load_csv(new_csv, n_bands)
    Xb, yb, _ = load_csv(bench_csv, n_bands)
    X, y = np.vstack([X1, X2]), np.r_[y1, y2]
    edge, gw = build_edge(n_bands).fit(X, y), build_gateway(n_bands).fit(X, y)
    f1 = f1_score(yb, edge.predict(Xb), average="macro")
    if f1 + tol < prev_f1:
        print(f"REJECTED: benchmark macro-F1 {f1:.3f} < previous {prev_f1:.3f}")
        return None
    version = time.strftime("v%Y%m%d_%H%M%S")
    info = export_header(edge, n_bands, WL_AS7265X, f"edge_model_{version}.h")
    import joblib
    joblib.dump(gw, f"gateway_model_{version}.joblib")
    json.dump({"version": version, "benchmark_macro_f1": f1, **info}, open(f"manifest_{version}.json", "w"), indent=2)
    print("published", version, "macro-F1", round(f1, 3))
    return version


if __name__ == "__main__":
    retrain(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
