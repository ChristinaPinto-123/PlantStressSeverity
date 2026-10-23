"""Supplementary experiment: edge -> gateway ladder with confidence gate on a TABULAR feature set
(chlorophyll-fluorescence PSII colour + GLCM features of wheat canopy, control vs drought).

    python run_tabular.py --x XyWheatCW.npy --y YWheatCW.npy

IMPORTANT LIMITS (state these in the paper):
  * Not spectral data: this tests the edge-gateway-cloud design, not the VIS-NIR method.
  * Rows are shuffled and carry no plant/day/image ID, so only a RANDOM stratified split is
    possible. Augmented copies of an image may sit in both train and test folds.
  * The gate threshold is swept on out-of-fold predictions; pick it on a separate validation
    split before reporting a single operating point.
"""
import argparse
import json
import os

import numpy as np
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from config import SEED
from evaluate import expected_calibration_error, latency_ms, model_size_kb

ap = argparse.ArgumentParser()
ap.add_argument("--x", required=True)
ap.add_argument("--y", required=True)
ap.add_argument("--folds", type=int, default=5)
args = ap.parse_args()
os.makedirs("results", exist_ok=True)

X, y = np.load(args.x), np.load(args.y)
keep = [j for j in range(X.shape[1]) if not any(np.allclose(X[:, j], X[:, i]) for i in range(j))]
if len(keep) < X.shape[1]:
    print(f"dropped duplicate columns: {sorted(set(range(X.shape[1])) - set(keep))}")
X = X[:, keep]
print(f"data {X.shape}, class counts {np.bincount(y).tolist()}")

edge_candidates = {                                    # MCU-sized models
    "edge_LDA": LinearDiscriminantAnalysis(),
    "edge_tree_d6": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=SEED),
    "edge_GB_small": GradientBoostingClassifier(n_estimators=40, max_depth=3, random_state=SEED),
}
gateway = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                 n_jobs=-1, random_state=SEED)


def metrics(y, p):
    pred = p.argmax(1)
    return {"macro_f1": f1_score(y, pred, average="macro"), "mcc": matthews_corrcoef(y, pred),
            "ece": expected_calibration_error(p, y)}


# out-of-fold probabilities for every model on the SAME folds -> gate sweep needs no refitting
cv = StratifiedKFold(args.folds, shuffle=True, random_state=SEED)
oof = {k: np.zeros((len(y), 2)) for k in [*edge_candidates, "gateway_RF"]}
for tr, te in cv.split(X, y):
    for k, m in {**edge_candidates, "gateway_RF": gateway}.items():
        oof[k][te] = clone(m).fit(X[tr], y[tr]).predict_proba(X[te])

R = {"validation": f"random stratified {args.folds}-fold (no group IDs available)", "tiers": {}, "gate": {}}
for k, p in oof.items():
    full = clone(edge_candidates.get(k, gateway)).fit(X, y)
    R["tiers"][k] = {**metrics(y, p), "size_kb_sklearn": model_size_kb(full),
                     "latency_ms_host": latency_ms(full, X[0], repeats=100)}
    print(f"{k:14s} macro-F1 {R['tiers'][k]['macro_f1']:.3f}  MCC {R['tiers'][k]['mcc']:.3f}  "
          f"ECE {R['tiers'][k]['ece']:.3f}  size {R['tiers'][k]['size_kb_sklearn']:.0f} KB")

# confidence gate: edge answers if max-prob >= thr, else gateway answers
thrs = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01]
for k in edge_candidates:
    R["gate"][k] = []
    for t in thrs:
        unc = oof[k].max(1) < t
        p = np.where(unc[:, None], oof["gateway_RF"], oof[k])
        R["gate"][k].append({"threshold": t, "offload_rate": float(unc.mean()), **metrics(y, p)})
    print(f"\ngate with {k}:  thr  offload  macro-F1")
    for r in R["gate"][k]:
        print(f"   {r['threshold']:5.2f}  {r['offload_rate']:6.1%}  {r['macro_f1']:.3f}")

# int8 export sanity check for the linear edge model (binary LDA -> one weight vector)
lda = LinearDiscriminantAnalysis().fit(X, y)
w, b = lda.coef_[0], lda.intercept_[0]
s = np.abs(w).max() / 127
wq = np.clip(np.round(w / s), -127, 127)
agree = float(((X @ w + b > 0) == ((X @ wq) * s + b > 0)).mean())
R["int8_LDA_label_agreement"] = agree
R["int8_LDA_flash_bytes"] = int(len(wq) + 8)
print(f"\nint8 LDA agrees with float LDA on {agree:.4f} of rows ({R['int8_LDA_flash_bytes']} bytes)")

json.dump(R, open("results/tabular_results.json", "w"), indent=2, default=float)

try:  # trade-off plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for k, c in zip(edge_candidates, ["#2a6fbb", "#d9822b", "#3a9a5b"]):
        ax.plot([100 * r["offload_rate"] for r in R["gate"][k]], [r["macro_f1"] for r in R["gate"][k]],
                "o-", color=c, label=f"{k} + RF gateway")
    ax.axhline(R["tiers"]["gateway_RF"]["macro_f1"], color="gray", ls="--", lw=1, label="gateway RF only")
    ax.set_xlabel("Samples sent to gateway (%)"); ax.set_ylabel("Macro-F1")
    ax.set_title("Confidence gate trade-off (random-split CV)")
    ax.grid(alpha=.3); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig("results/gate_tradeoff.png", dpi=160)
    print("saved results/gate_tradeoff.png")
except ImportError:
    print("matplotlib not installed; skipped plot")
