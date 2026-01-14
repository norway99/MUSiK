import sys
import os

parent = os.path.dirname(os.path.realpath('../'))
sys.path.append(parent)

import numpy as np
# import open3d as o3d
import matplotlib.pyplot as plt

from core import *

from utils import geometry


test_experiment = experiment.Experiment.load('/data/trevor/overflow/test_all_at_center')
test_experiment.run(dry=True)

test_reconstruction = reconstruction.DAS(experiment=test_experiment)

signals = test_reconstruction.get_signals(
    dimensions=2, matsize=512, downsample=1, workers=32, tgc=10
)

# Save raw signals as npz files:
np.savez('/data/trevor/overflow/test_all_at_center/signals.npz', signals=signals)