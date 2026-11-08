"""Runs the full evaluation plan from Section III-E and writes results/*.json.

    python run_all.py                 # synthetic stand-in data
    python run_all.py --csv data.csv  # real data (see data.load_csv)
"""
import argparse
import json
import os

import numpy as np

from config import EDGE_K, SEED, WL_AS7265X, WL_SRC
from data import load_csv, make_synthetic, resample_to_sensor
from evaluate import (cv_group, latency_ms, leaky_cv, leave_one_group_out, model_size_kb,
                      perturb, score)
from export_edge import export_header, predict_int8, extract_linear, quantize_int8
from gate import HierarchicalClassifier
from models import build_baseline, build_edge, build_gateway
from preprocess import CalibrationTransfer

ap = argparse.ArgumentParser()
ap.add_argument("--csv")
ap.add_argument("--n_bands", type=int, default=52)
args = ap.parse_args()
os.makedirs("results", exist_ok=True)
rng = np.random.default_rng(SEED)
R = {}

if args.csv:
    X, y, meta = load_csv(args.csv, args.n_bands)
    wl = WL_SRC if args.n_bands == 52 else WL_AS7265X
else:
    X, y, meta = make_synthetic()
    wl = WL_SRC
nb = len(wl)
nc = X.shape[1] - nb          # number of climate columns (0, 1 or 2)
plants, sites = meta["plant_id"].to_numpy(), meta["site"].to_numpy()
print(f"data: {X.shape}, class counts {np.bincount(y).tolist()}, plants {len(set(plants))}, sites {len(set(sites))}")

# ---- 1. primary: group-aware nested CV, model ladder -------------------------------------
ladder = {"baseline_VI_RF": build_baseline(nb, wl, n_clim=nc), "edge_SNV_LDA": build_edge(nb, n_clim=nc),
          "gateway_RF": build_gateway(nb, n_clim=nc)}
R["group_cv"] = {k: cv_group(m, X, y, plants) for k, m in ladder.items()}

# ---- 2. cross-domain: leave-one-site-out ------------------------------------------------
if len(set(sites)) > 1:
    R["leave_one_site_out"] = {k: leave_one_group_out(m, X, y, sites) for k, m in ladder.items()}

# ---- 3. perturbation tests (test folds only) ----------------------------------------------
R["perturbation_edge"] = {}
for kind, strengths in {"gain": [0.05, 0.10], "offset": [0.01, 0.03],
                        "noise": [0.005, 0.02], "contact": [0.03, 0.08]}.items():
    for s in strengths:
        R["perturbation_edge"][f"{kind}_{s}"] = cv_group(
            build_edge(nb, n_clim=nc), X, y, plants,
            test_transform=lambda Xt, k=kind, s=s: perturb(Xt, nb, k, s, np.random.default_rng(SEED)))["macro_f1"]

# ---- 4. ablations ---------------------------------------------------------------------------
R["ablation"] = {
    "leaky_selection": leaky_cv(X, y, plants, nb, EDGE_K)["macro_f1"],
    "nested_selection": R["group_cv"]["edge_SNV_LDA"]["macro_f1"],
    "snv_off": cv_group(build_edge(nb, use_snv=False, n_clim=nc), X, y, plants)["macro_f1"],
    "raw_bands_vs_VI": {"raw_edge": R["group_cv"]["edge_SNV_LDA"]["macro_f1"],
                        "VI_RF": R["group_cv"]["baseline_VI_RF"]["macro_f1"]},
}
if nc > 0:
    R["ablation"]["no_climate_fusion"] = cv_group(build_edge(nb, use_climate=False, n_clim=nc), X, y, plants)["macro_f1"]
chan = {}
for c in (6, 12, 24, 52):
    if c <= nb:
        idx = np.linspace(0, nb - 1, c).round().astype(int)
        Xc = np.hstack([X[:, idx], X[:, nb:]])
        chan[c] = cv_group(build_edge(c, k=min(EDGE_K, c), n_clim=nc), Xc, y, plants)["macro_f1"]
R["ablation"]["channel_count_macro_f1"] = chan

# ---- 5. calibration transfer: public/virtual sensor -> real 18-ch device ------------------------
Xv = np.hstack([resample_to_sensor(X[:, :nb], wl, WL_AS7265X), X[:, nb:]])   # virtual sensor space
nv = len(WL_AS7265X)
dev_gain, dev_off = rng.normal(1.0, 0.08, nv), rng.normal(0.0, 0.01, nv)      # simulated device
Xdev = Xv.copy()
Xdev[:, :nv] = Xv[:, :nv] * dev_gain + dev_off + rng.normal(0, 0.003, Xv[:, :nv].shape)
tiles = rng.uniform(0.05, 0.9, (25, nv))                                      # paired reference tiles
tiles_dev = tiles * dev_gain + dev_off + rng.normal(0, 0.003, tiles.shape)
ct = CalibrationTransfer().fit(tiles_dev, tiles)
def _ct(Xt):
    Xt = Xt.copy(); Xt[:, :nv] = ct.transform(Xt[:, :nv]); return Xt
virt_model = build_edge(nv, k=EDGE_K, n_clim=nc).fit(Xv, y)
R["calibration_transfer"] = {
    "virtual_sensor_in_domain": score(y, virt_model.predict(Xv))["macro_f1"],
    "device_without_transfer": score(y, virt_model.predict(Xdev))["macro_f1"],
    "device_with_transfer": score(y, virt_model.predict(_ct(Xdev)))["macro_f1"],
}

# ---- 6. hierarchical edge->gateway with confidence gate ------------------------------------------
from sklearn.model_selection import StratifiedGroupKFold
tr, te = next(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(Xv, y, plants))
edge18, gw18 = build_edge(nv, n_clim=nc).fit(Xv[tr], y[tr]), build_gateway(nv, n_clim=nc).fit(Xv[tr], y[tr])
R["gate"] = {}
for thr in (0.6, 0.7, 0.8, 0.9, 0.95):
    h = HierarchicalClassifier(edge18, gw18, thr)
    pred, conf, tier = h.predict(Xv[te])
    R["gate"][thr] = {"macro_f1": score(y[te], pred)["macro_f1"], "offload_rate": float(tier.mean())}

# ---- 7. cost metrics on the deployed 18-channel edge model ------------------------------------------
R["cost_edge18"] = {"latency_ms_python_host": latency_ms(edge18, Xv[0]),
                    "model_size_kb_sklearn": model_size_kb(edge18)}

# ---- 8. int8 export + agreement check ------------------------------------------------------------------
sel, W, b, mean, inv_std = extract_linear(edge18)
Wq, scale = quantize_int8(W)
p_f = edge18.predict_proba(Xv[te]); p_q = predict_int8(Xv[te], nv, sel, Wq, scale, b, mean, inv_std)
R["export"] = export_header(edge18, nv, WL_AS7265X, "results/edge_model.h")
R["export"]["float_vs_int8_label_agreement"] = float((p_f.argmax(1) == p_q.argmax(1)).mean())
R["export"]["max_abs_prob_diff"] = float(np.abs(p_f - p_q).max())

json.dump(R, open("results/results.json", "w"), indent=2, default=float)
print(json.dumps({k: v for k, v in R.items() if k != "group_cv"}, indent=2, default=float))
print("\nGROUP-CV macro-F1/MCC:", {k: (round(v["macro_f1"], 3), round(v["mcc"], 3)) for k, v in R["group_cv"].items()})
