"""Streamlit dashboard for the AIoT plant-stress pipeline.

    streamlit run dashboard.py

Tabs
  1. Live monitor   - replays a stream of leaf scans through edge -> confidence gate -> gateway ->
                      EWMA smoothing -> decision layer, exactly as gateway.py does (simulated nodes).
  2. Gate explorer  - threshold vs accuracy vs % offloaded to the gateway (spectral demo model).
  3. Experiment results - reads results/results.json written by run_all.py.
  4. Fluorescence data  - tabular edge/gateway experiment on your own .npy files.

The live monitor and gate explorer use SYNTHETIC spectra (data.make_synthetic). They demonstrate the
system behaviour; they are not experimental results.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from config import CLASSES, CONF_THRESHOLD, EWMA_ALPHA, SEED, WL_AS7265X, WL_SRC
from data import make_synthetic, resample_to_sensor
from evaluate import perturb
from gate import EWMASmoother, decision_map
from models import build_edge, build_gateway

st.set_page_config(page_title="Plant stress AIoT", page_icon="🌿", layout="wide")
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SEV_COLOR = {0: "🟢", 1: "🟡", 2: "🟠", 3: "🔴"}


# ----------------------------------------------------------------------------- cached demo model
@st.cache_resource(show_spinner="Training demo edge + gateway models (once)...")
def load_demo():
    X, y, meta = make_synthetic(scans_per_plant=20)
    nb, nv = len(WL_SRC), len(WL_AS7265X)
    Xv = np.hstack([resample_to_sensor(X[:, :nb], WL_SRC, WL_AS7265X), X[:, nb:]])
    plants = meta["plant_id"].to_numpy()
    tr, te = next(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(Xv, y, plants))
    edge = build_edge(nv).fit(Xv[tr], y[tr])
    gw = build_gateway(nv).fit(Xv[tr], y[tr])
    gw.named_steps["clf"].n_jobs = 1                       # single-row predictions: avoid thread overhead
    return Xv, y, plants, te, edge, gw


@st.cache_data(show_spinner=False)
def test_probas(_edge, _gw, _Xv, te):
    return _edge.predict_proba(_Xv[te]), _gw.predict_proba(_Xv[te])


Xv, y_all, plants_all, te_idx, edge, gw = load_demo()
NV = len(WL_AS7265X)


# ----------------------------------------------------------------------------- live stream setup
def build_stream(n_plants, scans_per_plant):
    rng = np.random.default_rng(SEED)
    chosen = []
    per_class = max(1, n_plants // 4)
    for c in range(4):
        ps = np.unique(plants_all[te_idx][y_all[te_idx] == c])
        chosen += list(rng.choice(ps, size=min(per_class, len(ps)), replace=False))
    rows = {p: list(np.where(plants_all == p)[0][:scans_per_plant]) for p in chosen}
    order = [rows[p][k] for k in range(scans_per_plant) for p in chosen if k < len(rows[p])]
    return chosen, order


def reset_stream():
    chosen, order = build_stream(st.session_state.get("n_plants", 8), 20)
    st.session_state.update(chosen=chosen, order=order, pos=0, smoother=EWMASmoother(st.session_state.get("alpha", EWMA_ALPHA)),
                            log=[], latest={}, hist=[])


def process_one():
    s = st.session_state
    if s.pos >= len(s.order):
        s.playing = False
        return
    i = s.order[s.pos]; s.pos += 1
    x = Xv[i:i + 1].copy()
    if s.noise > 0:
        x = perturb(x, NV, "noise", s.noise, np.random.default_rng(SEED + s.pos))
    if s.gain > 0:
        x = perturb(x, NV, "gain", s.gain, np.random.default_rng(SEED + 7 * s.pos))
    p_edge = edge.predict_proba(x)[0]
    uncertain = p_edge.max() < s.thr
    p = gw.predict_proba(x)[0] if uncertain else p_edge
    plant = int(plants_all[i])
    s.smoother.alpha = s.alpha
    sm = s.smoother.update(plant, p)
    sev = int(sm.argmax())
    temp, hum = float(x[0, NV]), float(x[0, NV + 1])
    action = decision_map(sev, temp, hum)
    rec = {"scan": s.pos, "plant": plant, "true": CLASSES[int(y_all[i])], "raw_pred": CLASSES[int(p.argmax())],
           "conf": round(float(p.max()), 3), "tier": "gateway" if uncertain else "edge",
           "smoothed": CLASSES[sev], "action": action}
    s.log.append(rec)
    s.latest[plant] = {**rec, "temp": round(temp, 1), "hum": round(hum, 1)}
    s.hist.append({"scan": s.pos, "plant": f"plant {plant}", "expected_severity": float(sm @ np.arange(4))})


# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("🌿 Plant stress AIoT")
st.sidebar.caption("Edge (SNV-LDA) → confidence gate → gateway (RF) → EWMA → decision")
s = st.session_state
s.thr = st.sidebar.slider("Confidence threshold", 0.50, 0.99, CONF_THRESHOLD, 0.01,
                          help="Edge answers if max-probability ≥ threshold, otherwise the gateway does.")
s.alpha = st.sidebar.slider("EWMA α (decision smoothing)", 0.05, 1.0, EWMA_ALPHA, 0.05)
st.sidebar.markdown("**Inject sensor problems** (live tab)")
s.noise = st.sidebar.slider("Added noise (σ, reflectance)", 0.0, 0.03, 0.0, 0.002)
s.gain = st.sidebar.slider("Gain / contact error (σ)", 0.0, 0.15, 0.0, 0.01)
st.sidebar.markdown("---")
st.sidebar.caption("Live + gate tabs use synthetic spectra: they show behaviour, not results for the paper.")

if "order" not in s:
    s.playing = False
    reset_stream()

tab_live, tab_gate, tab_res, tab_fluo = st.tabs(["Live monitor", "Gate explorer", "Experiment results", "Fluorescence data"])

# ----------------------------------------------------------------------------- tab 1: live monitor
with tab_live:
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    if c1.button("▶ Play" if not s.playing else "⏸ Pause", width="stretch"):
        s.playing = not s.playing
        st.rerun()
    if c2.button("⏭ Step ×1", width="stretch"):
        process_one()
    if c3.button("↺ Reset", width="stretch"):
        s.playing = False
        reset_stream()
        st.rerun()
    speed = c4.select_slider("Speed (scans per second)", [1, 2, 4, 8], value=2)
    s.n_plants = st.number_input("Plants in stream (applied on reset)", 4, 16, s.get("n_plants", 8), 4)

    def live_body():
        if s.playing:
            for _ in range(speed):
                process_one()
        n = len(s.log)
        edge_n = sum(r["tier"] == "edge" for r in s.log)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Scans processed", f"{s.pos} / {len(s.order)}")
        m2.metric("Handled on edge", f"{100 * edge_n / n:.0f}%" if n else "–")
        m3.metric("Sent to gateway", f"{100 * (n - edge_n) / n:.0f}%" if n else "–")
        corr = np.mean([r["raw_pred"] == r["true"] for r in s.log]) if n else None
        m4.metric("Single-scan accuracy", f"{100 * corr:.0f}%" if corr is not None else "–")

        if s.latest:
            df = pd.DataFrame(s.latest.values()).sort_values("plant")
            df.insert(0, " ", df["smoothed"].map(lambda c: SEV_COLOR[CLASSES.index(c)]))
            st.subheader("Current state per plant")
            st.dataframe(df[[" ", "plant", "true", "smoothed", "conf", "tier", "temp", "hum", "action"]],
                         hide_index=True, width="stretch")
            st.subheader("Smoothed severity over time (expected class, 0 = healthy … 3 = severe)")
            h = pd.DataFrame(s.hist).pivot_table(index="scan", columns="plant", values="expected_severity")
            st.line_chart(h, height=260)
            with st.expander("Message log (latest 25)"):
                st.dataframe(pd.DataFrame(s.log[-25:][::-1]), hide_index=True, width="stretch")
        else:
            st.info("Press ▶ Play or ⏭ Step to start streaming scans from simulated leaf-clip nodes.")

    st.fragment(run_every=1.0 if s.playing else None)(live_body)()

# ----------------------------------------------------------------------------- tab 2: gate explorer
with tab_gate:
    st.subheader("Confidence gate: what does a threshold buy you?")
    pe, pg = test_probas(edge, gw, Xv, te_idx)
    yt = y_all[te_idx]
    thrs = np.round(np.arange(0.40, 1.0001, 0.01), 2)
    rows = []
    for t in thrs:
        unc = pe.max(1) < t
        p = np.where(unc[:, None], pg, pe)
        rows.append({"threshold": t, "offload_pct": 100 * unc.mean(), "macro_f1": f1_score(yt, p.argmax(1), average="macro")})
    sweep = pd.DataFrame(rows)
    cur = sweep.iloc[(sweep.threshold - s.thr).abs().argmin()]
    a, b, c = st.columns(3)
    a.metric("Threshold", f"{cur.threshold:.2f}")
    b.metric("Sent to gateway", f"{cur.offload_pct:.1f}%")
    c.metric("Macro-F1", f"{cur.macro_f1:.3f}")
    st.caption(f"Edge only: F1 {f1_score(yt, pe.argmax(1), average='macro'):.3f} · "
               f"Gateway only: F1 {f1_score(yt, pg.argmax(1), average='macro'):.3f} (held-out plants, synthetic data). "
               "Move the threshold slider in the sidebar.")
    l, r = st.columns(2)
    l.markdown("**Macro-F1 vs threshold**"); l.line_chart(sweep.set_index("threshold")["macro_f1"], height=260)
    r.markdown("**% sent to gateway vs threshold**"); r.line_chart(sweep.set_index("threshold")["offload_pct"], height=260)
    st.markdown("**Trade-off curve** (x = % sent to gateway, y = macro-F1)")
    st.scatter_chart(sweep, x="offload_pct", y="macro_f1", height=260)

# ----------------------------------------------------------------------------- tab 3: results viewer
with tab_res:
    st.subheader("Evaluation plan results (run_all.py)")
    path = os.path.join(RES, "results.json")
    if st.button("Run run_all.py now (synthetic data, ~20-60 s)"):
        with st.spinner("Running full evaluation..."):
            out = subprocess.run([sys.executable, "run_all.py"], cwd=HERE, capture_output=True, text=True)
        st.success("Done") if out.returncode == 0 else st.error(out.stderr[-1500:])
    if os.path.exists(path):
        R = json.load(open(path))
        st.markdown("**Group-aware nested CV (model ladder)**")
        st.dataframe(pd.DataFrame({k: {m: v.get(m) for m in ("macro_f1", "mcc", "ece")} for k, v in R["group_cv"].items()}).T.round(3),
                     width="stretch")
        if "leave_one_site_out" in R:
            st.markdown("**Leave-one-site-out**")
            st.dataframe(pd.DataFrame({k: {m: v.get(m) for m in ("macro_f1", "mcc", "ece")} for k, v in R["leave_one_site_out"].items()}).T.round(3),
                         width="stretch")
        c1, c2 = st.columns(2)
        c1.markdown("**Edge model under perturbation (macro-F1)**")
        c1.bar_chart(pd.Series(R["perturbation_edge"]), height=280)
        c2.markdown("**Channel-count ablation (macro-F1)**")
        c2.bar_chart(pd.Series(R["ablation"]["channel_count_macro_f1"]), height=280)
        c3, c4 = st.columns(2)
        c3.markdown("**Calibration transfer (virtual sensor → device)**")
        c3.bar_chart(pd.Series(R["calibration_transfer"]), height=260)
        c4.markdown("**Ablations**")
        ab = {k: v for k, v in R["ablation"].items() if isinstance(v, (int, float))}
        c4.dataframe(pd.Series(ab, name="macro_f1").round(3), width="stretch")
        st.markdown("**Cost & int8 export**")
        st.json({"cost_edge18": R["cost_edge18"], "export": {k: R["export"][k] for k in R["export"] if k != "path"}})
        st.caption("Numbers here come from the data used for the last run (synthetic unless you passed --csv).")
    else:
        st.info("No results/results.json yet. Click the button above or run `python run_all.py`.")

# ----------------------------------------------------------------------------- tab 4: fluorescence
with tab_fluo:
    st.subheader("Fluorescence (PSII colour + GLCM) features: edge vs gateway")
    st.caption("Upload XyWheatCW.npy (features) and YWheatCW.npy (labels). Random-split CV only: no group IDs exist, "
               "so this supports the architecture, not the spectral method.")
    fx = st.file_uploader("Features .npy", type="npy", key="fx")
    fy = st.file_uploader("Labels .npy", type="npy", key="fy")

    @st.cache_data(show_spinner="Cross-validating edge + gateway models...")
    def fluo_oof(xb, yb, edge_name, n_trees):
        import io
        from sklearn.base import clone
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.model_selection import StratifiedKFold
        from sklearn.tree import DecisionTreeClassifier
        X, y = np.load(io.BytesIO(xb)), np.load(io.BytesIO(yb))
        keep = [j for j in range(X.shape[1]) if not any(np.allclose(X[:, j], X[:, i]) for i in range(j))]
        X = X[:, keep]
        edges = {"LDA": LinearDiscriminantAnalysis(),
                 "Decision tree (depth 6)": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=SEED),
                 "Gradient boosting (small)": GradientBoostingClassifier(n_estimators=40, max_depth=3, random_state=SEED)}
        gwm = RandomForestClassifier(n_estimators=n_trees, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)
        pe, pg = np.zeros((len(y), 2)), np.zeros((len(y), 2))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
            pe[te] = clone(edges[edge_name]).fit(X[tr], y[tr]).predict_proba(X[te])
            pg[te] = clone(gwm).fit(X[tr], y[tr]).predict_proba(X[te])
        return y, pe, pg

    if fx and fy:
        e_name = st.selectbox("Edge model", ["Decision tree (depth 6)", "Gradient boosting (small)", "LDA"])
        n_trees = st.select_slider("Gateway RF trees", [50, 100, 200, 300], value=100)
        yf, pe, pg = fluo_oof(fx.getvalue(), fy.getvalue(), e_name, n_trees)
        t = st.slider("Gate threshold", 0.50, 1.0, 0.80, 0.01, key="fl_thr")
        unc = pe.max(1) < t
        p = np.where(unc[:, None], pg, pe)
        a, b, c, d = st.columns(4)
        a.metric("Edge only F1", f"{f1_score(yf, pe.argmax(1), average='macro'):.3f}")
        b.metric("Gateway only F1", f"{f1_score(yf, pg.argmax(1), average='macro'):.3f}")
        c.metric("Gated F1", f"{f1_score(yf, p.argmax(1), average='macro'):.3f}")
        d.metric("Sent to gateway", f"{100 * unc.mean():.1f}%")
        grid = np.round(np.arange(0.5, 1.0001, 0.01), 2)
        sw = pd.DataFrame([{"offload_pct": 100 * (pe.max(1) < g).mean(),
                            "macro_f1": f1_score(yf, np.where((pe.max(1) < g)[:, None], pg, pe).argmax(1), average="macro")}
                           for g in grid], index=grid)
        st.markdown("**Macro-F1 vs % sent to gateway**")
        st.scatter_chart(sw, x="offload_pct", y="macro_f1", height=280)
    elif os.path.exists(os.path.join(RES, "tabular_results.json")):
        T = json.load(open(os.path.join(RES, "tabular_results.json")))
        st.info("Showing saved results from run_tabular.py. Upload the two .npy files for an interactive version.")
        st.dataframe(pd.DataFrame(T["tiers"]).T.round(3), width="stretch")
        edge_pick = st.selectbox("Edge model", list(T["gate"]))
        g = pd.DataFrame(T["gate"][edge_pick])
        st.scatter_chart(g, x="offload_rate", y="macro_f1", height=280)
