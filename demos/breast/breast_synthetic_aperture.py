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
import argparse


from core import *
from utils import phantom_builder
from utils import geometry
from utils import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--node', type=int, help='node index to run')
    
    args = parser.parse_args()
    
    test_experiment = experiment.Experiment.load('breast_synthetic_aperture_3')
    test_experiment.nodes = 1
    test_experiment.run(repeat=False, node=args.node)
    
    test_experiment.run(dry=True, dry_fast=True)
    test_experiment.add_results()
    test_reconstruction = reconstruction.Compounding(experiment=test_experiment)
    test_reconstruction.compound(workers=8, resolution_multiplier=1, combine=False, volumetric=True, attenuation_factor = 16, save_intermediates=True)
    
if __name__ == "__main__":
    sys.exit(main())
