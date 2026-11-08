"""Shared configuration for the AIoT plant-stress pipeline."""
import numpy as np

SEED = 42

# Wavelength grid of the public / reference hyperspectral-style data (52 bands, 340-1050 nm)
WL_SRC = np.linspace(340, 1050, 52)

# AS7265x (AS72651/2/3) 18 channels, sorted by wavelength (nm).
# Firmware must use this same ordering.
WL_AS7265X = np.array(
    [410, 435, 460, 485, 510, 535, 560, 585, 610, 645, 680, 705, 730, 760, 810, 860, 900, 940],
    dtype=float,
)

CLASSES = ["healthy", "mild", "moderate", "severe"]

EDGE_K = 8               # number of spectral bands kept on the edge model
CONF_THRESHOLD = 0.80    # confidence gate: below this -> gateway second opinion
EWMA_ALPHA = 0.4         # decision-layer smoothing
SMOOTH_WINDOW = 3        # 3-sample moving average (must match firmware)
