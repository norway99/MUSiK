import sys
import os

parent = os.path.dirname(os.path.realpath('../'))
sys.path.append(parent)

import numpy as np
import scipy
import tqdm
import open3d as o3d
import matplotlib.pyplot as plt
import glob

from core import *
from utils import phantom_builder
from utils import geometry
from utils import utils

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

test_experiment = experiment.Experiment.load('invitro_arm_experiment_1mhz_3D')
test_experiment.run(dry=True)

test_experiment.add_results()
test_reconstruction = reconstruction.Compounding(test_experiment)

z_heights = np.arange(-0.09, 0.09, 0.002)

for i in range(len(z_heights)):
    image = test_reconstruction.selective_compound(workers=30, transducers=[i*6,i*6+1,i*6+2,i*6+3,i*6+4,i*6+5], resolution_multiplier=2, local=True, combine=True)
    utils.save_array(image, f'invitro_arm_experiment_1mhz_3D/images/compounded_image_{i}.npy')