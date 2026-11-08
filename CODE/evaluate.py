"""Evaluation: group-aware nested CV, leave-one-site-out, perturbations, leakage ablation,
latency and model size."""
import io
import time

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import SEED
from preprocess import SNV, Smooth


# ------------------------------------------------------------------ metrics
def expected_calibration_error(proba, y, n_bins=10):
    conf, pred = proba.max(1), proba.argmax(1)
    acc = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def score(y, pred, proba=None):
    out = {
        "macro_f1": f1_score(y, pred, average="macro"),
        "mcc": matthews_corrcoef(y, pred),
        "per_class_f1": f1_score(y, pred, average=None, labels=range(4)).round(3).tolist(),
        "confusion": confusion_matrix(y, pred, labels=range(4)).tolist(),
    }
    if proba is not None:
        out["ece"] = expected_calibration_error(proba, y)
    return out


# ------------------------------------------------------------------ perturbations
def perturb(X, n_bands, kind, strength, rng):
    """Apply gain / offset / noise / contact perturbation to the spectral part of X only."""
    X = X.copy()
    S = X[:, :n_bands]
    n = len(S)
    if kind == "gain":
        S *= rng.normal(1.0, strength, (n, 1))
    elif kind == "offset":
        S += rng.normal(0.0, strength, (n, 1))
    elif kind == "noise":
        S += rng.normal(0.0, strength, S.shape)
    elif kind == "contact":  # smooth wavelength-dependent tilt (leaf-clip pressure / angle)
        t = np.linspace(-1, 1, n_bands)
        S *= 1.0 + rng.normal(0, strength, (n, 1)) * t + rng.normal(0, strength, (n, 1))
    else:
        raise ValueError(kind)
    X[:, :n_bands] = S
    return X


# ------------------------------------------------------------------ validation
def cv_group(model, X, y, groups, n_splits=5, test_transform=None, seed=SEED):
    """Group-aware stratified CV (group = plant). Feature selection is inside `model`, so it is
    refitted on each training fold (nested). `test_transform` perturbs only the test folds."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    proba = np.zeros((len(y), 4))
    for tr, te in cv.split(X, y, groups):
        m = clone(model).fit(X[tr], y[tr])
        Xte = X[te] if test_transform is None else test_transform(X[te])
        proba[te] = m.predict_proba(Xte)
    return score(y, proba.argmax(1), proba)


def leave_one_group_out(model, X, y, groups):
    """Leave-one-site (or cultivar / crop) -out cross-domain validation."""
    proba = np.zeros((len(y), 4))
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        m = clone(model).fit(X[tr], y[tr])
        p = m.predict_proba(X[te])
        proba[np.ix_(te, m.classes_)] = p
    return score(y, proba.argmax(1), proba)


def leaky_cv(X, y, groups, n_bands, k, n_splits=5, seed=SEED):
    """DELIBERATELY LEAKY baseline: bands are selected on ALL data before CV.
    Compare with cv_group(build_edge) to quantify leakage inflation."""
    sel = Pipeline([("smooth", Smooth()), ("snv", SNV()), ("select", SelectKBest(f_classif, k=k))])
    Z = sel.fit_transform(X[:, :n_bands], y)
    Z = np.hstack([Z, X[:, n_bands:]])
    model = Pipeline([("scale", StandardScaler()),
                      ("clf", LinearDiscriminantAnalysis(solver="svd", priors=np.full(4, 0.25)))])
    return cv_group(model, Z, y, groups, n_splits=n_splits, seed=seed)


# ------------------------------------------------------------------ cost metrics
def latency_ms(model, x_one, repeats=300):
    """Median single-sample inference time on THIS machine. For the paper, run on the target
    edge hardware (ESP32 timing is measured by the firmware, see edge_node.ino)."""
    x_one = x_one.reshape(1, -1)
    ts = []
    for _ in range(repeats):
        t = time.perf_counter()
        model.predict_proba(x_one)
        ts.append((time.perf_counter() - t) * 1e3)
    return float(np.median(ts))


def model_size_kb(model):
    buf = io.BytesIO()
    joblib.dump(model, buf)
    return buf.tell() / 1024.0
