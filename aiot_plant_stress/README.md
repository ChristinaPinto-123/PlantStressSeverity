# AIoT plant-stress severity: code for the proposed methodology

Maps to Section III of the paper.

| Paper section | File |
|---|---|
| Data sources, virtual-sensor mapping (52 bands to AS7265x 18 ch) | `data.py` |
| Reference normalisation, QC, 3-pt smoothing, SNV, calibration transfer | `preprocess.py` |
| Model ladder: edge SNV-LDA, gateway RF, vegetation-index baseline; nested band selection | `models.py` |
| Confidence gate, EWMA / majority-vote smoothing, climate-aware decision mapping | `gate.py` |
| Group-aware CV, leave-one-site-out, perturbations, leakage ablation, latency, size, ECE | `evaluate.py` |
| Full evaluation plan (III-E) | `run_all.py` |
| int8 export of edge model to C header | `export_edge.py` |
| Leaf-clip node (ESP32 + AS7265x + SHT31) | `firmware/edge_node/edge_node.ino` |
| Gateway (MQTT, 2nd opinion, offline buffer) | `gateway.py` |
| Cloud retrain, benchmark validation, versioning | `cloud_retrain.py` |

## Using a real dataset
    python prepare_dataset.py raw.csv data.csv --label-col treatment \
        --label-order control,low,medium,high --group-col plant_id --site-col site
    python run_all.py --csv data.csv

`prepare_dataset.py` takes a wide CSV (numeric wavelength headers), resamples to the 52-band grid,
maps labels to severity 0-3 and writes `b0..b51, label, plant_id, site`. Temp/humidity columns are
optional (`--temp-col/--hum-col`); without them the model is spectral-only and the
climate-fusion ablation is skipped. Always pass `--group-col`, otherwise CV can leak between scans.

## Supplementary tabular experiment (fluorescence features, no spectra)
    python run_tabular.py --x XyWheatCW.npy --y YWheatCW.npy
Tests the edge/gateway confidence gate on 32 PSII-colour + GLCM features (control vs drought).
Random-split CV only (no group IDs), so it supports the architecture, not the spectral method.

## Run
    pip install -r requirements.txt
    python run_all.py                    # synthetic stand-in data
    python run_all.py --csv data.csv     # real data: plant_id, site, label, b0..b51, temp, hum

Then copy `results/edge_model.h` next to `edge_node.ino`.

## Important caveats
* The built-in dataset is SYNTHETIC. Its numbers (about 0.96 macro-F1) only show the code works;
  do not report them. Use EcoSIS / LOPEX93 / ANGERS or your own measurements via `--csv`.
  The synthetic task is easy, so leaky vs nested selection also look identical here.
* Energy per measurement, end-to-end latency, packet loss and on-ESP32 inference time need
  hardware. The firmware prints inference time (microseconds) over serial for this.
* Firmware is untested on hardware: verify pins, LED driver, gain and integration settings.
* The 1D-CNN gateway option and OTA delivery of model weights are not implemented
  (the gateway uses RF; the edge model is compiled in via `edge_model.h`).
