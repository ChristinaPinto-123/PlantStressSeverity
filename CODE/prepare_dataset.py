"""Convert a public leaf-spectra dataset (wide CSV: one row per scan) into the format run_all.py
expects, resampled onto the 52-band grid (340-1050 nm).

Input: a CSV whose spectral columns have NUMERIC headers (wavelength in nm, e.g. 400, 401, ...)
plus metadata columns. Examples:

  # categorical stress labels, ordered mild -> severe
  python prepare_dataset.py raw.csv data.csv --label-col treatment \
      --label-order control,low,medium,high --group-col plant_id --site-col site

  # numeric trait (e.g. chlorophyll, relative water content) -> 4 severity classes by quantile
  python prepare_dataset.py raw.csv data.csv --label-col RWC --label-bins 4 --label-descending \
      --group-col plant_id

Severity 0 = healthy ... 3 = most stressed. Use --label-descending when LOW values mean MORE stress
(chlorophyll, water content).
"""
import argparse

import numpy as np
import pandas as pd

from config import WL_SRC

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("dst")
ap.add_argument("--label-col", required=True)
ap.add_argument("--label-order", help="comma-separated category names, healthy first")
ap.add_argument("--label-bins", type=int, help="quantile-bin a numeric label column into N classes")
ap.add_argument("--label-descending", action="store_true")
ap.add_argument("--group-col", help="plant / sample id (REQUIRED for honest group-aware CV)")
ap.add_argument("--site-col"); ap.add_argument("--temp-col"); ap.add_argument("--hum-col")
ap.add_argument("--percent", action="store_true", help="spectra are in % (divide by 100)")
a = ap.parse_args()

df = pd.read_csv(a.src)
spec_cols = []
for c in df.columns:
    try: spec_cols.append((float(c), c))
    except ValueError: pass
spec_cols.sort()
wl = np.array([w for w, _ in spec_cols]); S = df[[c for _, c in spec_cols]].to_numpy(float)
if a.percent or np.nanmax(S) > 1.5: S = S / 100.0
print(f"{len(df)} scans, {len(wl)} spectral columns, {wl.min():.0f}-{wl.max():.0f} nm")
if wl.min() > WL_SRC[0] + 1 or wl.max() < WL_SRC[-1] - 1:
    print(f"WARNING: data covers {wl.min():.0f}-{wl.max():.0f} nm but the grid is 340-1050 nm; "
          "edge bands are filled by repeating the end value. Prefer shrinking WL_SRC in config.py.")
B = np.vstack([np.interp(WL_SRC, wl, row) for row in S])

lab = df[a.label_col]
if a.label_order:
    order = [s.strip() for s in a.label_order.split(",")]
    y = lab.map({k: i for i, k in enumerate(order)})
    if y.isna().any(): raise SystemExit(f"labels not in --label-order: {sorted(set(lab[y.isna()]))}")
    y = y.astype(int).to_numpy()
elif a.label_bins:
    v = -lab if a.label_descending else lab
    y = pd.qcut(v.rank(method="first"), a.label_bins, labels=False).astype(int).to_numpy()
else:
    y = lab.astype(int).to_numpy()
if len(set(y)) != 4: print(f"WARNING: {len(set(y))} classes found; the code assumes 4 (0-3).")

out = pd.DataFrame(B, columns=[f"b{i}" for i in range(len(WL_SRC))])
if a.group_col: out.insert(0, "plant_id", df[a.group_col].to_numpy())
else:
    out.insert(0, "plant_id", np.arange(len(df)))
    print("WARNING: no --group-col; every scan is its own group, so CV may leak between scans of "
          "the same plant and give optimistic scores.")
out.insert(1, "site", df[a.site_col].to_numpy() if a.site_col else 0)
out.insert(2, "label", y)
if a.temp_col: out["temp"] = df[a.temp_col].to_numpy(float)
if a.hum_col: out["hum"] = df[a.hum_col].to_numpy(float)
out.to_csv(a.dst, index=False)
print(f"wrote {a.dst}; class counts {np.bincount(y).tolist()}; climate columns: "
      f"{[c for c in ('temp','hum') if c in out.columns] or 'none'}")
