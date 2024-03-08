import sys
sys.path.append('../')
import numpy as np

from phantom import Phantom
from tissue import Tissue
from transducer import Transducer, Focused, Planewave
from transducer_set import TransducerSet
from simulation import SimProperties, Simulation
from sensor import Sensor
from experiment import Experiment
#from reconstruction import Reconstruction


print('Creating phantom...')
phantom = Phantom(source_path = None,
            voxel_dims = (0.1e-3,0.1e-3,0.1e-3),
            matrix_dims = (256,256,256),
            baseline = (1500, 1000),
            seed = 5678,)

dense = Tissue(name='dense', c=1650, rho=1100, sigma=0, scale=0.00001, label=1)

phantom.add_tissue_sphere((0.5,0,0), 8, dense)
phantom.add_tissue_sphere((-0.5,0,0), 8, dense)
phantom.add_tissue_sphere((0,0.5,0), 8, dense)
phantom.add_tissue_sphere((0,-0.5,0), 8, dense)
phantom.add_tissue_sphere((0,0,0.5), 8, dense)
phantom.add_tissue_sphere((0,0,-0.5), 8, dense)

print('Creating transducer_set...')
transducers = [Focused(elements = 32, 
                       elevation = 1e-4 * 32, 
                       sensor_sampling_scheme = 'not_centroid', 
                       sweep = np.pi/6, 
                       ray_num = 64,
                       imaging_ndims = 2,
                       focus_azimuth = 20e-3,
                       focus_elevation = 20e-3,
                       ) for i in range(4)]

for t in transducers:
    t.make_sensor_coords(phantom.baseline[0])


transducer_set = TransducerSet(transducers, seed=7777)
transducer_set.generate_extrinsics(shape="spherical", extrinsics_kwargs={'r_mean': 0.03, 'view_std': 0, 'yaw_fraction': 1, 'pitch_fraction': 0, 'roll_fraction': 0})
transducer_set.plot_transducers()
transducer_set.plot_transducer_fovs(scale=0.02)

print('Creating sensor...')
sensor = Sensor(transducer_set=transducer_set)

print('Setting simulation parameters...')
simprops = SimProperties(grid_size   = (1024,128,128),
                 PML_size    = (32,8,8),
                 PML_alpha   = 2,
                 t_end       = 10e-5,          # [s]
                 bona        = 6,               # parameter b/a determining degree of nonlinear acoustic effects
                 alpha_coeff = 0.75, 	        # [dB/(MHz^y cm)]
                 alpha_power = 1.5,
                 )

print('Creating experiment...')
test_experiment = Experiment(
                 simulation_path = '../calibration_experiment_00',
                 sim_properties  = simprops,
                 phantom         = phantom,
                 transducer_set  = transducer_set,
                 sensor          = sensor,
                 nodes           = 2,
                 results         = None,
                 indices         = None,
                 workers         = 2,
                 )

test_experiment.save()
print(f'Experiment saved to {test_experiment.simulation_path}')