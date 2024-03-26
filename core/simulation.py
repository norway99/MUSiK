import numpy as np
import scipy
import os
import tempfile
import time
import matplotlib.ticker
import matplotlib.pyplot as plt
from contextlib import contextmanager
import shutil
import glob

from .tissue import Tissue
from .phantom import Phantom
from .transducer import Transducer, Focused, Planewave
from .transducer_set import TransducerSet

import sys
sys.path.append('../utils')
import utils
import geometry

sys.path.append('../k-wave-python')
import kwave
import kwave.kmedium
import kwave.options.simulation_options
import kwave.options.simulation_execution_options
import kwave.kspaceFirstOrder3D


@contextmanager
def tempdir():
    path = tempfile.mkdtemp()
    try:
        yield path
    finally:
        try:
            shutil.rmtree(path)
        except IOError:
            sys.stderr.write('Failed to clean up temp dir at {}'.format(path))


class SimProperties:
    def __init__(self,
                 grid_size   = (1024,128,128),
                 PML_size    = (32,8,8),
                 PML_alpha   = 2,
                 t_end       = 2e-5,            # [s]
                 bona        = 6,               # parameter b/a determining degree of nonlinear acoustic effects
                 alpha_coeff = 0.75, 	        # [dB/(MHz^y cm)]
                 alpha_power = 1.5,
    ):
        self.grid_size   = grid_size
        self.PML_size    = PML_size
        self.PML_alpha   = PML_alpha
        self.t_end       = t_end
        self.bona        = bona
        self.alpha_coeff = alpha_coeff
        self.alpha_power = alpha_power
    
    
    def save(self, filepath):
        utils.dict_to_json(self.__dict__, filepath)
	
    @classmethod
    def load(cls, filepath):
        dictionary = utils.json_to_dict(filepath)
        simprops = cls()
        for key in dictionary.keys():
            simprops.__setattr__(key, dictionary[key])
        return simprops


class Simulation:

    def __init__(self, 
                 sim_properties,
                 phantom,
                 transducer_set,
                 sensor,
                 simulation_path = '',
                 index           = None,
                 gpu             = True,
                 dry             = False,
                 ):
        
        self.sim_properties = sim_properties
        self.phantom = phantom
        self.transducer_set = transducer_set
        self.sensor = sensor
        self.simulation_path = simulation_path
        self.index = index
        self.gpu = gpu
        self.dry = dry
        self.prepped_simulation = None
        
        
    def prep(self,):
        if self.prepped_simulation is not None:
            return
        self.prepped_simulation = self.__prep_by_index(self.index, dry=self.dry)
                        
        
    def run(self,):
        if self.prepped_simulation is None:
            self.prep()
        self.__run_by_index(self.index, dry=self.dry)
        
    
    # given a simulation index, return the simulation file
    def __prep_by_index(self, index, dry=False):
        start_time = time.time()
        # print(f'preparing simulation index: {index:18d}')
        for transducer_number, transducer in enumerate(self.transducer_set.transducers):
            if index - transducer.get_num_rays() < 0:
                steering_angle = transducer.steering_angles[index]
                # elapsed_time = time.time()
                sim_phantom = self.phantom.get_complete()
                # print('created simulation phantom in      {:11.2f} seconds'.format(round(time.time() - elapsed_time, 3)))
                # elapsed_time = time.time()
                sim_sensor = self.sensor
                
                if not dry:
                    affine = self.transducer_set.poses[transducer_number] * transducer.ray_transforms[index]
                    sim_phantom, center = affine.apply_to_array(sim_phantom, scale = self.phantom.voxel_dims[0], padwith=self.phantom.baseline)
                    sim_phantom = self.__crop_phantom(sim_phantom, center)
                    prepped = self.__prep_simulation(index, sim_phantom, transducer, sim_sensor, affine, steering_angle)
                    print('preparation for sim {:4d} completed in {:15.2f} seconds'.format(self.index, round(time.time() - start_time, 3)))
                    return prepped
                else:
                    affine = geometry.Transform()
                    sim_phantom = np.ones((2, self.sim_properties.grid_size[0], self.sim_properties.grid_size[1], self.sim_properties.grid_size[2])) * np.array([1540, 1000])[:,None,None,None]
                    self.__prep_simulation(index, sim_phantom, transducer, sim_sensor, affine, steering_angle, dry=True)
            else:
                index -= transducer.get_num_rays()
                
                
    # given a simulation index, return the simulation file
    def __run_by_index(self, index, dry=False):
        if not dry:
            start_time = time.time()
            # print(f'running simulation index:       {index:11d}')
            signals, time_array = self.__run_simulation(self.prepped_simulation)
            self.__write_signal(index, signals, time_array)
            print('simulation {:6d} completed in {:5.2f} seconds'.format(index, round(time.time() - start_time, 3)))
            return signals, time_array
                

    # crop phantom to the grid size of the simulation        
    def __crop_phantom(self, sim_phantom, center):
        crop_size = [self.sim_properties.grid_size[0] - 2 * self.sim_properties.PML_size[0],
                     self.sim_properties.grid_size[1] - 2 * self.sim_properties.PML_size[1],
                     self.sim_properties.grid_size[2] - 2 * self.sim_properties.PML_size[2],]
                
        axial_zero = int(center[-3])
        
        if crop_size[-3] > axial_zero:
            sim_phantom = np.stack(
                (np.pad(sim_phantom[0], ((0, crop_size[-3]),(0,0),(0,0)), 'constant', constant_values=(self.phantom.baseline[0],)),
                 np.pad(sim_phantom[1], ((0, crop_size[-3]),(0,0),(0,0)), 'constant', constant_values=(self.phantom.baseline[1],))),
                axis=0)
        
        if crop_size[-2]//2 > int(center[-2]) or crop_size[-2]//2 > sim_phantom.shape[-2] - int(center[-2]):
            expand = max(crop_size[-2]//2 - int(center[-2]), crop_size[-2]//2 - (sim_phantom.shape[-2] - int(center[-2])))
            sim_phantom = np.stack(
                (np.pad(sim_phantom[0], ((0,0),(expand, expand),(0,0)), 'constant', constant_values=(self.phantom.baseline[0],)),
                 np.pad(sim_phantom[1], ((0,0),(expand, expand),(0,0)), 'constant', constant_values=(self.phantom.baseline[1],))),
                axis=0)
            center = center + np.array([0, expand//2, 0])
            
        if crop_size[-1]//2 > int(center[-1]) or crop_size[-1]//2 > sim_phantom.shape[-1] - int(center[-1]):
            expand = max(crop_size[-1]//2 - int(center[-1]), crop_size[-1]//2 - (sim_phantom.shape[-1] - int(center[-1])))
            sim_phantom = np.stack(
                (np.pad(sim_phantom[0], ((0,0),(0,0),(expand, expand)), 'constant', constant_values=(self.phantom.baseline[0],)),
                 np.pad(sim_phantom[1], ((0,0),(0,0),(expand, expand)), 'constant', constant_values=(self.phantom.baseline[1],))),
                axis=0)
            center = center + np.array([0, 0, expand//2])
                                
        return sim_phantom[:,
                           axial_zero                                :  crop_size[-3] + axial_zero, 
                           int((center[-2] - crop_size[-2] / 2)) : int((center[-2] + crop_size[-2] / 2)), 
                           int((center[-1] - crop_size[-1] / 2)) : int((center[-1] + crop_size[-1] / 2)),]
    
    
    # given the simulation properties, phantom, and transducer, create the simulation file for the cuda binary and run
    def __prep_simulation(self, index, sim_phantom, sim_transducer, sim_sensor, affine, steering_angle, dry=False):
                
        # setup kgrid object
        pml_size_points = kwave.data.Vector(self.sim_properties.PML_size)  # [grid points]
        grid_size_points = kwave.data.Vector(self.sim_properties.grid_size) - 2 * pml_size_points  # [grid points]
        grid_size_meters = grid_size_points * self.phantom.voxel_dims  # [m]
        grid_spacing_meters = self.phantom.voxel_dims
        kgrid = kwave.kgrid.kWaveGrid(grid_size_points, grid_spacing_meters)
        t_end = self.sim_properties.t_end # [s]
        
        c0 = self.phantom.baseline[0] # [m/s]
        rho0 = self.phantom.baseline[1] # [kg/m^3]
                
        kgrid.makeTime(c0, t_end=t_end)
        
        # set up phantom
        sound_speed_map = sim_phantom[0]
        density_map     = sim_phantom[1]
        
        # fetch not_a_transducer object from transducer
        sim_transducer.make_pulse(kgrid.dt, c0, rho0)
        not_transducer = sim_transducer.make_notatransducer(kgrid, c0, steering_angle, self.sim_properties.PML_size)

        # setup medium object
        medium = kwave.kmedium.kWaveMedium(
            sound_speed=None,  # will be set later
            alpha_coeff=0.75,
            alpha_power=1.5,
            BonA=6
        )
        medium_position = 0
        
        medium.sound_speed = sound_speed_map
        medium.density = density_map
        
        sensor_mask, discretized_sensor_coords = sim_sensor.make_sensor_mask(not_transducer, self.phantom, self.sim_properties.grid_size, affine)
            
        return (medium, kgrid, not_transducer, sensor_mask, pml_size_points, discretized_sensor_coords, sim_transducer, sim_sensor)
        
        
    # given the simulation properties, phantom, and transducer, create the simulation file for the cuda binary and run
    def __run_simulation(self, prepped_simulation):
        
        medium, kgrid, not_transducer, sensor_mask, pml_size_points, discretized_sensor_coords, sim_transducer, sim_sensor = prepped_simulation
        
        # preallocate scan_line data array
        scan_line = np.zeros((not_transducer.number_active_elements, kgrid.Nt))
        
        # with tempfile.TemporaryDirectory() as temp_directory: # - encountering issue with this and multiprocessing, not removing temp directory
        with tempdir() as temp_directory:
            
            # set the input settings
            input_filename = f'input_{str(self.index).zfill(6)}.h5'
            input_file_full_path = os.path.join(temp_directory, input_filename)
            
            simulation_options = kwave.options.simulation_options.SimulationOptions(
                pml_inside=False,
                pml_size=pml_size_points,
                data_cast='single',
                data_recast=True,
                save_to_disk=True,  # This is broken in the current version of k-Wave, so we save to disk and manually delete to free memory
                input_filename=input_file_full_path,
                save_to_disk_exit=False
            )

            sensor_data = kwave.kspaceFirstOrder3D.kspaceFirstOrder3D(
                medium=medium,
                kgrid=kgrid,
                source=not_transducer,
                sensor=kwave.ksensor.kSensor(mask=sensor_mask),
                simulation_options=simulation_options,
                execution_options=kwave.options.simulation_execution_options.SimulationExecutionOptions(is_gpu_simulation=self.gpu)
            )
            

            utils.save_array(sensor_data['p'].T, 'sensor_data_woo', compression=False)
            utils.save_array(discretized_sensor_coords, 'disc_sensor_coords_woo', compression=False)
            
            # remove temporary files
            tmppath = tempfile.gettempdir()
            input_files = glob.glob(tempfile.gettempdir()+"/*input*.h5")
            if len(input_files) > 8: # ensure there are no more than 8 temporary files at a time - should work for up to 4 gpu's per node
                input_files = sorted(input_files, key=os.path.getmtime)
                for input_file in input_files[:1]:
                    os.remove(input_file)
                        
            element_signals = sim_sensor.voxel_to_element(self.sim_properties, sim_transducer, discretized_sensor_coords, sensor_data)
        
        return element_signals, kgrid.t_array
        
    
    def __write_signal(self, index, signal, time_array):
        utils.save_array(np.concatenate((time_array, signal), axis=0), os.path.join(self.simulation_path, f'results/signal_{str(index).zfill(6)}'), compression=False)
    
    
    def save(self, filepath):
        print('saving and loading simulations deemed unnecessary, not implemented')
    
    
    def load(self, filepath):
        print('saving and loading simulations deemed unnecessary, not implemented')
        

    def plot_medium_path(self, index, ax=None, save=False, save_path=None, cmap='viridis'):
        for transducer_number, transducer in enumerate(self.transducer_set.transducers):
            if index - transducer.get_num_rays() < 0:
                affine = self.transducer_set.poses[transducer_number] * transducer.ray_transforms[index]
                steering_angle = transducer.steering_angles[index]
                sim_phantom = self.phantom.get_complete()
                sim_phantom, center = affine.apply_to_array(sim_phantom, scale = self.phantom.voxel_dims[0], padwith=self.phantom.baseline)
                sim_phantom = self.__crop_phantom(sim_phantom, center)
                break
            else:
                index -= transducer.get_num_rays()
                                
        if ax is None:
            fig, ax = plt.subplots(2, 1, figsize=(10, 10 * sim_phantom.shape[2] / sim_phantom.shape[1] * 2 + 1))
        
        vmin = np.amin(sim_phantom[0])
        vmax = np.amax(sim_phantom[0])
                
        ax[0].imshow(sim_phantom[0, :, :, sim_phantom.shape[3]//2].T, cmap=cmap, vmin=vmin, vmax=vmax)
        ax[1].imshow(sim_phantom[0, :, sim_phantom.shape[2]//2, :].T, cmap=cmap, vmin=vmin, vmax=vmax)
        ax[0].plot([0, sim_phantom.shape[1]-1], [sim_phantom.shape[2]//2, sim_phantom.shape[2]//2], color='red', linewidth=1)
        ax[1].plot([0, sim_phantom.shape[1]-1], [sim_phantom.shape[3]//2, sim_phantom.shape[3]//2], color='red', linewidth=1)
        
        ax[0].set_ylabel('y')
        ax[1].set_ylabel('z')
        ax[1].set_xlabel('x')
        ax[0].get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: '{:.3f}'.format(float(x * self.phantom.voxel_dims[0]))))
        ax[1].get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: '{:.3f}'.format(float(x * self.phantom.voxel_dims[0]))))
        
        ax[0].get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: '{:.3f}'.format(float(x * self.phantom.voxel_dims[1] - (self.phantom.voxel_dims[1] * sim_phantom.shape[2] / 2)))))
        ax[1].get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: '{:.3f}'.format(float(x * self.phantom.voxel_dims[2] - (self.phantom.voxel_dims[2] * sim_phantom.shape[3] / 2)))))
        if save:
            plt.savefig(save_path)
        elif ax is None:
            plt.show()
