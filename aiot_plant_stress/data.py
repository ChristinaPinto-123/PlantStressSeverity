"""Data loading.

* load_csv(): use this for real data (EcoSIS / LOPEX93 / ANGERS exports, or your own device).
  Expected columns: plant_id, site, label, b0..b{N-1} (reflectance); temp, hum optional
  (use prepare_dataset.py to convert a public dataset into this format)
* make_synthetic(): a physically-motivated stand-in so the whole pipeline can be run and
  tested before real data is available. NUMBERS FROM IT ARE NOT RESULTS FOR THE PAPER.
"""
import numpy as np
import pandas as pd

from config import SEED, WL_SRC


def load_csv(path, n_bands):
    df = pd.read_csv(path)
    band_cols = [f"b{i}" for i in range(n_bands)]
    clim = [c for c in ("temp", "hum") if c in df.columns]   # climate columns are optional
    X = np.hstack([df[band_cols].to_numpy(float), df[clim].to_numpy(float)])
    meta = df[["plant_id", "site"]].copy()
    return X, df["label"].to_numpy(int), meta


def resample_to_sensor(spectra, wl_src, wl_dst):
    """Virtual-sensor mapping: interpolate high-resolution spectra onto the channels of the
    cheap sensor (used to pretrain on public hyperspectral data)."""
    return np.vstack([np.interp(wl_dst, wl_src, s) for s in spectra])


def _base_spectrum(wl, severity, rng):
    sig = lambda x, c, w: 1.0 / (1.0 + np.exp(-(x - c) / w))
    edge_shift = -4.0 * severity + rng.normal(0, 1.5)          # red-edge blue shift with stress
    chl = 0.35 - 0.07 * severity                                # chlorophyll absorption depth
    refl = (0.05 + 0.08 * np.exp(-((wl - 550) / 35) ** 2)       # green peak
            + 0.45 * sig(wl, 715 + edge_shift, 12)               # red edge -> NIR plateau
            - (0.9 - 0.15 * severity) * 0.04 * np.exp(-((wl - 675) / 18) ** 2) * (chl / 0.35)
            + 0.03 * severity * np.exp(-((wl - 680) / 30) ** 2)  # red reflectance rises
            - 0.03 * severity * sig(wl, 900, 40)                 # NIR drop / water
            )
    return np.clip(refl, 0.01, 0.95)


def make_synthetic(n_sites=4, plants_per_site=30, scans_per_plant=6, wl=WL_SRC, seed=SEED):
    rng = np.random.default_rng(seed)
    rows, ys, plants, sites = [], [], [], []
    pid = 0
    for s in range(n_sites):
        site_gain = rng.normal(1.0, 0.05)
        site_temp = rng.normal(28, 3)
        site_hum = rng.normal(65, 8)
        for _ in range(plants_per_site):
            sev = int(rng.choice(4, p=[0.4, 0.25, 0.2, 0.15]))  # imbalanced classes
            plant_off = rng.normal(0, 0.01)
            plant_gain = rng.normal(1.0, 0.04)
            temp = site_temp + 1.2 * sev + rng.normal(0, 1.5)
            hum = site_hum - 3.0 * sev + rng.normal(0, 4)
            base = _base_spectrum(wl, sev + rng.normal(0, 0.25), rng)
            for _ in range(scans_per_plant):
                contact = 1.0 + rng.normal(0, 0.03) + rng.normal(0, 0.02) * (wl - wl.mean()) / wl.std()
                spec = base * site_gain * plant_gain * contact + plant_off + rng.normal(0, 0.004, wl.size)
                rows.append(np.r_[spec, temp + rng.normal(0, 0.3), hum + rng.normal(0, 1.0)])
                ys.append(sev); plants.append(pid); sites.append(s)
            pid += 1
    X = np.vstack(rows)
    meta = pd.DataFrame({"plant_id": plants, "site": sites})
    return X, np.array(ys), meta
