import sys
sys.path.append('../utils')
sys.path.append('../core')
sys.path.append('../')
import numpy as np
import scipy
import tqdm
import utils
import geometry
from core import *

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import glob

test_phantom = phantom.Phantom(source_path = None,
            voxel_dims = (0.1e-3,0.1e-3,0.1e-3),
            matrix_dims = (400,400,400),
            baseline = (1500, 1000),
            seed = 5678,)

blood = tissue.Tissue(name='blood', c=1578, rho=1060, sigma=11.3, scale=0.00001, label=1)
myocardium = tissue.Tissue(name='myocardium', c=1561.3, rho=1081, sigma=30, scale=0.0001, label=2)
fat = tissue.Tissue(name='fat', c=1450, rho=920, sigma=10, scale=0.0001, label=3)

pos = 0.7

test_phantom.add_tissue_sphere((pos,-pos,0), 8, blood)
test_phantom.add_tissue_sphere((pos,0,0), 16, blood)
test_phantom.add_tissue_sphere((pos,pos,0), 32, blood)

test_phantom.add_tissue_sphere((-pos,-pos,0), 16, fat)
test_phantom.add_tissue_sphere((-pos,0,0), 32, fat)
test_phantom.add_tissue_sphere((-pos,pos,0), 8, fat)

test_phantom.add_tissue_sphere((0,-pos,0), 32, myocardium)
test_phantom.add_tissue_sphere((0,0,0), 8, myocardium)
test_phantom.add_tissue_sphere((0,pos,0), 16, myocardium)

n_transducers = 2

transducers = [transducer.Planewave(elements = 32, 
                       elevation = 1e-4 * 32, 
                       sensor_sampling_scheme = 'not_centroid', 
                       sweep = np.pi/3, 
                       ray_num = 64, 
                       imaging_ndims = 2,
                       focus_elevation = 20e-3,
                       ) for i in range(n_transducers)]

for t in transducers:
    t.make_sensor_coords(test_phantom.baseline[0])

test_transducer_set = transducer_set.TransducerSet(transducers, seed=8888)
test_transducer_set.generate_extrinsics(shape="spherical", extrinsics_kwargs={'r_mean': 10e-3, 'view_std': 0, 'yaw_fraction': 50, 'pitch_fraction': 0, 'roll_fraction': 0})

test_sensor = sensor.Sensor(transducer_set=test_transducer_set, aperture_type='extended')

simprops = simulation.SimProperties(grid_size   = (1024,256,256),
                 PML_size    = (32,8,8),
                 PML_alpha   = 2,
                 t_end       = 12e-5,          # [s]
                 bona        = 6,               # parameter b/a determining degree of nonlinear acoustic effects
                 alpha_coeff = 0.5, 	        # [dB/(MHz^y cm)]
                 alpha_power = 1.5,
                 )

test_experiment = experiment.Experiment(
                 simulation_path = '../experiment_files/pw_01',
                 sim_properties  = simprops,
                 phantom         = test_phantom,
                 transducer_set  = test_transducer_set,
                 sensor          = test_sensor,
                 nodes           = 1,
                 results         = None,
                 indices         = None,
                 workers         = 3,
                 )

test_experiment.save()

test_experiment.run(dry=True)

test_experiment.run()