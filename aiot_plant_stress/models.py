"""Model ladder: edge (SNV-LDA), gateway (Random Forest), baseline (vegetation indices + RF).

Input convention everywhere: X = [spectral bands (n_bands) | temp, hum]. Band selection
(SelectKBest) lives INSIDE the pipeline, so it is re-fitted on each training fold only
(nested / leakage-free).
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from config import EDGE_K, SEED
from preprocess import SNV, Smooth

N_CLIMATE = 2


def _spectral_pipe(k, use_snv=True, use_smooth=True):
    steps = []
    if use_smooth:
        steps.append(("smooth", Smooth()))
    if use_snv:
        steps.append(("snv", SNV()))
    steps.append(("select", SelectKBest(f_classif, k=k)))
    return Pipeline(steps)


def _pre(n_bands, spec_transformer, use_climate=True, n_clim=N_CLIMATE):
    spec_idx = list(range(n_bands))
    clim_idx = list(range(n_bands, n_bands + n_clim))
    transformers = [("spec", spec_transformer, spec_idx)]
    if use_climate and n_clim > 0:
        transformers.append(("clim", "passthrough", clim_idx))
    return ColumnTransformer(transformers, remainder="drop")


def build_edge(n_bands, k=EDGE_K, use_snv=True, use_climate=True, n_clim=N_CLIMATE):
    """Tier 1 - runs on the MCU:  smooth -> SNV -> band select -> standardise -> LDA."""
    return Pipeline([
        ("pre", _pre(n_bands, _spectral_pipe(min(k, n_bands), use_snv), use_climate, n_clim)),
        ("scale", StandardScaler()),
        ("clf", LinearDiscriminantAnalysis(solver="svd", priors=np.full(4, 0.25))),  # class-balanced
    ])


def build_gateway(n_bands, k=24, use_climate=True, n_clim=N_CLIMATE):
    """Tier 2 - second opinion on the gateway (Raspberry Pi class)."""
    return Pipeline([
        ("pre", _pre(n_bands, _spectral_pipe(min(k, n_bands)), use_climate, n_clim)),
        ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                       n_jobs=-1, random_state=SEED)),
    ])


class VegIndices(BaseEstimator, TransformerMixin):
    """NDVI, NDRE, PRI, GNDVI, red-edge slope from raw reflectance (needs wavelength grid)."""

    def __init__(self, wl=None):
        self.wl = wl

    def fit(self, X, y=None):
        return self

    def _b(self, X, nm):
        return X[:, int(np.argmin(np.abs(np.asarray(self.wl) - nm)))]

    def transform(self, X):
        nd = lambda a, b: (a - b) / (a + b + 1e-9)
        r = lambda nm: self._b(X, nm)
        return np.c_[nd(r(810), r(680)), nd(r(760), r(705)), nd(r(535), r(560)),
                     nd(r(810), r(560)), (r(760) - r(705)) / 55.0]


def build_baseline(n_bands, wl, use_climate=True, n_clim=N_CLIMATE):
    """Comparison tier: fixed vegetation indices + classifier (what low-cost devices usually do)."""
    return Pipeline([
        ("pre", _pre(n_bands, VegIndices(wl), use_climate, n_clim)),
        ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                       n_jobs=-1, random_state=SEED)),
    ])
