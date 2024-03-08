
import sys
sys.path.append('../')
sys.path.append('../../k-wave-python/kwave')

import numpy as np
import scipy
import tqdm
import utils
import geometry
from phantom import Phantom
from tissue import Tissue
from transducer import Transducer, Focused, Planewave
from transducer_set import TransducerSet
from simulation import SimProperties, Simulation
from sensor import Sensor
from experiment import Experiment

import matplotlib.pyplot as plt

import kwave

import plotly.express as px

phantom = Phantom(source_path = None,
            voxel_dims = (0.2e-3,0.2e-3,0.2e-3),
            matrix_dims = (256,256,256),
            baseline = (1500, 1000),
            seed = 5678,)

blood = Tissue(name='blood', c=1578, rho=1060, sigma=11.3, scale=0.00001, label=1)
myocardium = Tissue(name='myocardium', c=1561.3, rho=1081, sigma=30, scale=0.0001, label=2)

phantom.add_tissue_sphere((0,0,0), 128, myocardium)
phantom.add_tissue_sphere((0.5,0,0), 32, blood)
phantom.add_tissue_sphere((-0.5,0,0), 32, blood)
phantom.add_tissue_sphere((0,0.5,0), 32, blood)
phantom.add_tissue_sphere((0,-0.5,0), 32, blood)
phantom.add_tissue_sphere((0,0,0.5), 32, blood)
phantom.add_tissue_sphere((0,0,-0.5), 32, blood)

transducers = [Focused(elements = 32, 
                       elevation = 1e-4 * 32, 
                       sensor_sampling_scheme = 'not_centroid', 
                       sweep = np.pi/3, 
                       ray_num = 128, 
                       imaging_ndims = 2,
                       focus_azimuth = 30e-3,
                       focus_elevation = 30e-3,
                       ) for i in range(3)]

for t in transducers:
    t.make_sensor_coords(phantom.baseline[0])

transducer_set = TransducerSet(transducers, seed=0000)
transducer_set.generate_extrinsics(shape="spherical", extrinsics_kwargs={'r_mean': 0.03, 'yaw_fraction': -0.5, 'pitch_fraction': 0, 'roll_fraction': 0})

transducer_set.assign_pose(0, geometry.Transform((0,0,0), (-30e-3,0,0), intrinsic=False))

sensor = Sensor(transducer_set=transducer_set, aperture_type="extended")

simprops = SimProperties(grid_size   = (1024,128,128),
                 PML_size    = (32,8,8),
                 PML_alpha   = 2,
                 t_end       = 1.2e-4,          # [s]
                 bona        = 6,               # parameter b/a determining degree of nonlinear acoustic effects
                 alpha_coeff = 0.75, 	        # [dB/(MHz^y cm)]
                 alpha_power = 1.5,
                 )

test_experiment = Experiment(
                 simulation_path = '../test_experiment_00',
                 sim_properties  = simprops,
                 phantom         = phantom,
                 transducer_set  = transducer_set,
                 sensor          = sensor,
                 nodes           = 1,
                 results         = None,
                 indices         = 28,
                 )

test_experiment.run(dry=True)

test_experiment.add_results()
