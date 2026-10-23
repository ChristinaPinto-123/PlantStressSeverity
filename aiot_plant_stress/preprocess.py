"""Spectral preprocessing: reference normalisation, QC, smoothing, SNV, calibration transfer.

Every transformer here is a scikit-learn transformer so it can sit inside a Pipeline and be
fit ONLY on training folds (leakage-free).
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from config import SMOOTH_WINDOW


def reference_normalize(raw, dark, white, eps=1e-9):
    """R = (raw - dark) / (white - dark). Done on device for every session."""
    return (raw - dark) / np.maximum(white - dark, eps)


def qc_mask(raw, dark, white, sat_level=0.98, min_signal=0.02, max_dark_ratio=0.15):
    """Return True for scans that pass QC: not saturated, enough signal, no light leak."""
    sig = white - dark
    ok_sat = (raw < sat_level * np.max(white, axis=-1, keepdims=True)).all(axis=-1)
    ok_sig = (sig.mean(axis=-1) > min_signal * np.max(white, axis=-1))
    ok_dark = (dark.mean(axis=-1) < max_dark_ratio * white.mean(axis=-1))
    return ok_sat & ok_sig & ok_dark


def moving_average(X, window=SMOOTH_WINDOW):
    """Moving average along the spectral axis with edge replication (same as firmware)."""
    pad = window // 2
    Xp = np.pad(X, ((0, 0), (pad, pad)), mode="edge")
    out = np.zeros_like(X, dtype=float)
    for i in range(window):
        out += Xp[:, i:i + X.shape[1]]
    return out / window


def snv(X, eps=1e-9):
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)  # population std (ddof=0), same as firmware
    return (X - mu) / (sd + eps)


class Smooth(BaseEstimator, TransformerMixin):
    def __init__(self, window=SMOOTH_WINDOW):
        self.window = window

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return moving_average(np.asarray(X, float), self.window)


class SNV(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return snv(np.asarray(X, float))


class CalibrationTransfer(BaseEstimator, TransformerMixin):
    """Per-band linear map  device -> virtual sensor, fitted from paired measurements of the
    same reference tiles on both instruments (simple, MCU-friendly calibration transfer)."""

    def fit(self, X_device, X_virtual):
        n_b = X_device.shape[1]
        self.gain_, self.offset_ = np.ones(n_b), np.zeros(n_b)
        for j in range(n_b):
            A = np.c_[X_device[:, j], np.ones(len(X_device))]
            (self.gain_[j], self.offset_[j]), *_ = np.linalg.lstsq(A, X_virtual[:, j], rcond=None)
        return self

    def transform(self, X):
        return np.asarray(X) * self.gain_ + self.offset_
