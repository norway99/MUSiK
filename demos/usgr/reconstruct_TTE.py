import sys
import os

parent = os.path.dirname(os.path.realpath('../'))
sys.path.append(parent)

import numpy as np
import tqdm
from PIL import Image

from musik import *

from utils import geometry


# Experiment directory:
experiment_dir = '/data/trevor/usgr/simulations_raw/TTE_10k_fast_gaussian'


if not os.path.exists(os.path.join(experiment_dir, 'result_recon', 'signals.npz')):
    test_experiment = experiment.Experiment.load(experiment_dir)
    test_experiment.run(dry=True)
    test_reconstruction = reconstruction.DAS(experiment=test_experiment)
    signals = test_reconstruction.get_signals(
        dimensions=2, matsize=256, downsample=1, workers=8, tgc=10, save_dir=os.path.join(experiment_dir, 'result_recon')
    )

signals = np.load(os.path.join(experiment_dir, 'result_recon', 'signals.npz'))['signals']


# compute percentiles
p0 = np.nanpercentile(signals, 0)
p95 = np.nanpercentile(signals, 97.5)

# clip and normalize
signals_clipped = np.clip(signals, p0, p95)
signals_normalized = (signals_clipped - p0) / (p95 - p0) * 255

# replace NaNs
signals_normalized = np.nan_to_num(signals_normalized, nan=127).astype(np.uint8)  # or nan=127, nan=255, etc.

img_dir = os.path.join(experiment_dir, 'images')

# Save all the normalized signals as 8-bit PNG files
os.makedirs(img_dir, exist_ok=True)
for i in tqdm.tqdm(range(signals_normalized.shape[0])):
    Image.fromarray(signals_normalized[i]).save(f'{img_dir}/signals_{i:06d}.png')
