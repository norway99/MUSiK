import sys
import os

parent = os.path.dirname(os.path.realpath('../'))
sys.path.append(parent)

import numpy as np
# import open3d as o3d
import matplotlib.pyplot as plt

from musik import *

from utils import geometry


test_experiment = experiment.Experiment.load('/data/trevor/usgr/simulations_raw/TTE_5k')
test_experiment.run(dry=True)

test_reconstruction = reconstruction.DAS(experiment=test_experiment)

signals = test_reconstruction.get_signals(
    dimensions=2, matsize=256, downsample=1, workers=8, tgc=10, save_dir='/data/trevor/usgr/simulations_raw/TTE_5k/result_recon'
)

# # Save raw signals as npz files:
# np.savez('/data/trevor/usgr/simulations_raw/TTE_5k/signals.npz', signals=signals)