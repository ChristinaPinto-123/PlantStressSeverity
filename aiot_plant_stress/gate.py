"""Confidence gate (edge -> gateway) and the decision layer (smoothing + context mapping)."""
import numpy as np

from config import CONF_THRESHOLD, EWMA_ALPHA


class HierarchicalClassifier:
    """Edge model answers if max-probability >= threshold, otherwise the gateway model does."""

    def __init__(self, edge, gateway, threshold=CONF_THRESHOLD):
        self.edge, self.gateway, self.threshold = edge, gateway, threshold

    def predict_proba(self, X):
        p = self.edge.predict_proba(X)
        unc = p.max(axis=1) < self.threshold
        tier = np.where(unc, 1, 0)
        if unc.any():
            p = p.copy()
            p[unc] = self.gateway.predict_proba(X[unc])
        return p, tier

    def predict(self, X):
        p, tier = self.predict_proba(X)
        return p.argmax(axis=1), p.max(axis=1), tier


class EWMASmoother:
    """Exponentially weighted moving average of class-probability vectors, per plant."""

    def __init__(self, alpha=EWMA_ALPHA):
        self.alpha, self.state = alpha, {}

    def update(self, plant_id, proba):
        prev = self.state.get(plant_id)
        cur = np.asarray(proba, float) if prev is None else self.alpha * np.asarray(proba) + (1 - self.alpha) * prev
        self.state[plant_id] = cur
        return cur


def majority_vote(labels):
    return int(np.bincount(labels).argmax())


def decision_map(severity, temp, hum):
    """Map smoothed severity + climate context to an action."""
    if severity == 0:
        return "no_action"
    ctx = ""
    if hum > 80 and 22 <= temp <= 32:
        ctx = " (humid+warm: check for fungal/bacterial disease)"
    elif hum < 40 and temp > 35:
        ctx = " (hot+dry: likely heat/drought stress)"
    return {1: "monitor_closely", 2: "alert_and_inspect", 3: "alert_and_irrigate_or_treat"}[severity] + ctx
