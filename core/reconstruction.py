import numpy as np
import scipy
import os
import glob
import matplotlib.pyplot as plt
import tqdm
import sys
import multiprocessing
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
sys.path.append('../utils')
import time

import utils
import geometry
from transducer import Focused, Planewave, Transducer
from phantom import Phantom
from transducer_set import TransducerSet
from simulation import Simulation, SimProperties
from sensor import Sensor
from .experiment import Experiment


class Reconstruction:
    
    def __init__(self, 
                 experiment = None,
                 ):
        if isinstance(experiment, Experiment):
            self.simulation_path = experiment.simulation_path
            self.sim_properties = experiment.sim_properties
            self.phantom = experiment.phantom
            self.transducer_set = experiment.transducer_set
            self.sensor = experiment.sensor
            self.results = experiment.results
            self.experiment = experiment
        elif isinstance(experiment, str):
            self.experiment = Experiment.load(experiment)
            self.simulation_path = self.experiment.simulation_path
            self.sim_properties = self.experiment.sim_properties
            self.phantom = self.experiment.phantom
            self.transducer_set = self.experiment.transducer_set
            self.sensor = self.experiment.sensor
            self.results = self.experiment.results
        else:
            assert False, 'Please provide an experiment to reconstruct.'
            
        
    def __len__(self):
        if self.transducer_set is None:
            return 0
        return sum([transducer.get_num_rays() for transducer in self.transducer_set.transducers])
    
        
    def add_results(self,):
        self.results = Results(os.path.join(self.simulation_path,'results'))


class DAS(Reconstruction):

    def __init__(self,
                 experiment = None):
        # for transducer in experiment.transducer_set.transducers:
            # if not isinstance(transducer, Focused):
            #     print("Warning: attempting to instantiate DAS reconstruction class but transducer set does not exclusively contain focused transducers.")
            #     break
        super().__init__(experiment)
            
        
    def plot_ray_path(self, index, ax=None, save=False, save_path=None, cmap='viridis'):
        if ax is None:
            fig, ax = plt.subplots(2,1, figsize=(15,5))
            self.experiment.plot_ray_path(index, ax=ax)
            centerline = self.sim_properties.grid_size[1] // 2 - self.sim_properties.PML_size[1]
            ax[0].plot(self.experiment.results[index][0] / self.experiment.phantom.voxel_dims[0] * 1540 / 2, self.experiment.results[index][1].T / 20 + centerline, linewidth=0.1, color='cyan', alpha=0.5)
            ax[0].set_ylim(0, centerline * 2)
            ax[1].plot(self.experiment.results[index][0] / self.experiment.phantom.voxel_dims[0] * 1540 / 2, self.experiment.results[index][1].T / 20 + centerline, linewidth=0.1, color='cyan', alpha=0.5)
            ax[1].set_ylim(0, centerline * 2)
            plt.show()
            
        
    def __time_to_coord(self, t, transform):
        dists = t * self.phantom.baseline[0] / 2
        
        dists = np.pad(dists[...,None], ((0,0),(0,2)), mode='constant', constant_values=0)
        coords = transform.apply_to_points(dists)
        return coords
    

    def process_line(self, index, transducer, transform, transmit_as_receive=True, attenuation_factor=1):
        processed = transducer.preprocess(transducer.make_scan_line(self.results[index][1], transmit_as_receive), self.results[index][0], self.sim_properties, attenuation_factor=attenuation_factor)
        coords = self.__time_to_coord(self.results[index][0], transform)
        times = self.results[index][0]
        return times, coords, processed


    def preprocess_data(self, global_transforms=True, workers=8, attenuation_factor=1):
        transducer_count = 0
        transducer, transducer_transform = self.transducer_set[transducer_count]
        running_index_list = np.cumsum([transducer.get_num_rays() for transducer in self.transducer_set.transducers])
        
        indices = []
        transducers = []
        transforms = []
        transmit_as_receive = []
        for index in range(len(self.results)):
            if index > running_index_list[transducer_count] - 1:
                transducer_count += 1
                transducer, transducer_transform = self.transducer_set[transducer_count]
            if global_transforms:
                transform = transducer_transform * transducer.ray_transforms[index - running_index_list[transducer_count]]
            else:
                transform = transducer.ray_transforms[index - running_index_list[transducer_count]]
            
            indices.append(index)
            transducers.append(transducer)
            transforms.append(transform)
            
                
        results = []
        sensor_receive = [self.sensor.aperture_type == "transmit_as_receive" for i in range(len(self.results))]
        attenuation_factor = [attenuation_factor for i in range(len(self.results))]
        inputs = list(zip(indices, transducers, transforms, sensor_receive, attenuation_factor))
        starttime = time.time()
        
        
        if workers > 1:
            with multiprocessing.Pool(8) as p:
                results = p.starmap(self.process_line, tqdm.tqdm(inputs, total=len(indices)))
        else:
            for input_data in inputs:
                results.append(self.process_line(input_data[0],input_data[1],input_data[2], input_data[3], input_data[4]))
        
        times = [r[0] for r in results]
        coords = [r[1] for r in results]
        processed = [r[2] for r in results]
        return times, coords, processed
    
        
    def plot_scatter(self, scale=5000, workers=1):
       
        colorme = lambda x: (     [1,0,0] if x % 7 == 0 
                             else [0,1,0] if x % 7 == 1 
                             else [0,0,1] if x % 7 == 2 
                             else [1,1,0] if x % 7 == 3 
                             else [1,0,1] if x % 7 == 4 
                             else [0,1,1] if x % 7 == 5 
                             else [1,1,1])
        
        times, coords, processed = self.preprocess_data(workers=workers)
        coords = np.stack(coords, axis=0)
        processed = np.stack(processed, axis=0)
        times = np.stack(times, axis=0)
        
        transducer_lens = [t.get_num_rays() for t in self.transducer_set.transducers]
        base_color = np.array([colorme(i) for i in range(len(transducer_lens)) for j in range(transducer_lens[i] * len(times[i]))])
        
        fig, ax = plt.subplots(1,1, figsize=(8,8))
        
        coords = np.reshape(coords, (-1,3))
        processed = np.reshape(processed, (-1,))
        times = np.reshape(times, (-1,))
        base_color = np.reshape(base_color, (-1,3))
        intensity = np.clip(processed, 0, scale)/scale
        
        colors = np.array([1,1,1]) - np.broadcast_to(intensity[:,None], base_color.shape) * base_color
        
        ax.scatter(coords[:,0], coords[:,1], c=colors, s=times*100000, alpha = 0.002)
        
        ax.set_aspect('equal')
        

    def get_image(self, bounds=None, matsize=256, dimensions=3, downsample = 1, workers=8, attenuation_factor=1):
        assert dimensions in [2,3], print("Image can be 2 or 3 dimensional")
        assert (downsample > 0 and downsample <= 1), print("Downsample must be a float on (0,1]")
        
        times, coords, processed = self.preprocess_data(workers=workers, attenuation_factor=attenuation_factor)
        coords = np.stack(coords, axis=0)
        processed = np.stack(processed, axis=0)

        if bounds is None:
            flat_coords = coords.reshape(-1,3)
            bounds = np.array([(np.min(flat_coords[:,0]),np.max(flat_coords[:,0])),
                            (np.min(flat_coords[:,1]),np.max(flat_coords[:,1])),
                            (np.min(flat_coords[:,2]),np.max(flat_coords[:,2]))])
        elif type(bounds) == list or type(bounds) == tuple or type(bounds) == np.ndarray:
            bounds = np.array(bounds)
        elif type(bounds) == float:
            bounds = np.array([(-bounds,bounds),(-bounds,bounds),(-bounds,bounds)])
        else:
            print("provide bounds as a list, tuple, numpy array, or float")
            return 0

        X = np.linspace(bounds[0,0], bounds[0,1], matsize)
        Y = np.linspace(bounds[1,0], bounds[1,1], matsize)
        Z = np.linspace(bounds[2,0], bounds[2,1], matsize)
        
        if dimensions == 2:
            X, Y = np.meshgrid(X, Y, indexing='ij') # worked before changing indexing to ij so maybe take this out if it doesn't work anymore :/
        else:
            X, Y, Z = np.meshgrid(X, Y, Z, indexing='ij')

        signals = []
        count = 0
        for transducer in tqdm.tqdm(self.transducer_set.transducers):
            subset_coords = coords[count:(count+transducer.get_num_rays()),:].reshape(-1,3)
            subset_processed = processed[count:(count+transducer.get_num_rays())].reshape(-1)
            
            if downsample != 1:
                subset_coords = subset_coords[::int(1/downsample)]
                subset_processed = subset_processed[::int(1/downsample)]
                            
            if dimensions == 2:
                interp = LinearNDInterpolator(subset_coords[:,:2], subset_processed)
                signals.append(interp(X, Y))
            else:
                interp = NearestNDInterpolator(subset_coords, subset_processed)
                signals.append(interp(X, Y, Z))
            count += transducer.get_num_rays()
        
        combined_signals = np.stack(signals, axis=0)
        masked_signals = np.ma.masked_array(combined_signals, np.isnan(combined_signals))
        image = np.ma.average(masked_signals, axis=0)
        image = image.filled(np.nan)
        
        return image, signals
    
    
    def get_signals(self, bounds=None, matsize=256, dimensions=3, downsample = 1, workers=8, tgc=1):
        assert dimensions in [2,3], print("Image can be 2 or 3 dimensional")
        assert (downsample > 0 and downsample <= 1), print("Downsample must be a float on (0,1]")
        
        times, coords, processed = self.preprocess_data(global_transforms=False, workers=workers, attenuation_factor=tgc)
        coords = np.stack(coords, axis=0)
        processed = np.stack(processed, axis=0)

        if bounds is None:
            flat_coords = coords.reshape(-1,3)
            bounds = np.array([(np.min(flat_coords[:,0]),np.max(flat_coords[:,0])),
                            (np.min(flat_coords[:,1]),np.max(flat_coords[:,1])),
                            (np.min(flat_coords[:,2]),np.max(flat_coords[:,2]))])
        elif type(bounds) == list or type(bounds) == tuple or type(bounds) == np.ndarray:
            bounds = np.array(bounds)
        elif type(bounds) == float:
            bounds = np.array([(-bounds,bounds),(-bounds,bounds),(-bounds,bounds)])
        else:
            print("provide bounds as a list, tuple, numpy array, or float")
            return 0
        

        bounds_avg = (bounds[0,1] - bounds[0,0] + bounds[1,1] - bounds[1,0] + bounds[2,1] - bounds[2,0])/3
        X = np.linspace(bounds[0,0], bounds[0,1], int((bounds[0,1]-bounds[0,0]) / bounds_avg * matsize))
        Y = np.linspace(bounds[1,0], bounds[1,1], int((bounds[1,1]-bounds[1,0]) / bounds_avg * matsize))
        Z = np.linspace(bounds[2,0], bounds[2,1], int((bounds[2,1]-bounds[2,0]) / bounds_avg * matsize))
        
        if dimensions == 2:
            X, Y = np.meshgrid(X, Y, indexing='ij') # worked before changing indexing to ij so maybe take this out if it doesn't work anymore :/
        else:
            X, Y, Z = np.meshgrid(X, Y, Z, indexing='ij')

        signals = []
        count = 0
        for transducer in tqdm.tqdm(self.transducer_set.transducers):
            subset_coords = coords[count:(count+transducer.get_num_rays()),:].reshape(-1,3)
            subset_processed = processed[count:(count+transducer.get_num_rays())].reshape(-1)
            
            if downsample != 1:
                subset_coords = subset_coords[::int(1/downsample)]
                subset_processed = subset_processed[::int(1/downsample)]
                            
            if dimensions == 2:
                interp = LinearNDInterpolator(subset_coords[:,:2], subset_processed)
                signals.append(interp(X, Y))
            else:
                interp = NearestNDInterpolator(subset_coords, subset_processed)
                signals.append(interp(X, Y, Z))
            count += transducer.get_num_rays()
        
        return signals
        
        
        
class Compounding(Reconstruction):

    def __init__(self,
                 experiment = None):

        super().__init__(experiment)

    def __get_element_centroids(self): # in global coordinates
        sensor_coords = self.sensor.sensor_coords
        sensors_per_el = self.sensor.sensors_per_el
        element_centroids = np.zeros((sensors_per_el.size, 3))
        pos = 0
        for entry in range(sensors_per_el.size):
            element_centroids[entry] = np.mean(sensor_coords[int(pos):int(pos+sensors_per_el[entry]), :], axis = 0)
            pos += sensors_per_el[entry]
        return element_centroids
    
    def compound(self): # not just plane-wave compounding, also works for saft (extended aperture with focused transducers)

        if isinstance(self.transducer_set[0], Focused):
            # do nothing
            pass
        else:
            # still do nothing
            pass

        matrix_dims = self.phantom.matrix_dims
        voxel_dims = self.phantom.voxel_dims
        c0 = self.phantom.baseline[0]
        # dt = self.kgrid.dt # not the correct way to access the kgrid
        # ------------------------------------------------------------------------------------------------------------
        # fix this sometime
        dt = (self.results[0][0][-1] - self.results[0][0][0]) / self.results[0][0].shape[0]
        # ------------------------------------------------------------------------------------------------------------
        
        # need to choose lateral, axial, and elevation resolutions
        
        resolution = max(dt*c0, 2*c0/self.transducer_set.get_lowest_frequency()) / 4 # make sure this works
        
        x = np.arange(-matrix_dims[0]*voxel_dims[0]/2, matrix_dims[0]*voxel_dims[0]/2, step=resolution)
        y = np.arange(-matrix_dims[1]*voxel_dims[1]/2, matrix_dims[1]*voxel_dims[1]/2, step=resolution)
        z = np.arange(-matrix_dims[2]*voxel_dims[2]/2, matrix_dims[2]*voxel_dims[2]/2, step=resolution)
                
        image_matrix = np.zeros((len(x), len(y), len(z)))
        
        # note that origin is at center of the 3d image in global coordinate system
        
        element_centroids = self.__get_element_centroids()
        transducer_count = 0
        transducer, transducer_transform = self.transducer_set[transducer_count]
        running_index_list = np.cumsum([transducer.get_num_rays() for transducer in self.transducer_set.transducers])
        steering_angle = transducer.steering_angles[0]
        
        for index in tqdm.tqdm(range(len(self.results))):
            if index > running_index_list[transducer_count] - 1:
                transducer_count += 1
                transducer, transducer_transform = self.transducer_set[transducer_count]
            
            steering_angle = transducer.steering_angles[index - running_index_list[transducer_count]]

            dt = (self.results[index][0][-1] - self.results[index][0][0]) / self.results[index][0].shape[0]
            preprocessed_data = transducer.preprocess(self.results[index][1], self.results[index][0], self.sim_properties, window_factor=8)
            
            t_start = int(np.ceil(transducer.width / 2 *  np.abs(np.sin(steering_angle)) / c0 / dt + len(transducer.get_pulse())/2))
            
            if len(preprocessed_data.shape) == 2:
                preprocessed_data = preprocessed_data[:, t_start:]
                preprocessed_data = np.pad(preprocessed_data, ((0,0),(0,int(preprocessed_data.shape[1]*1.73))),)
            else:
                preprocessed_data = preprocessed_data[:, :, t_start:]
                                
            transmit_position = transducer_transform.translation
                        
            if isinstance(transducer, Planewave):
                transmit_rotation = transducer_transform.get()[:3, :3]
                pw_rotation = np.array([[np.cos(steering_angle), -np.sin(steering_angle), 0], [np.sin(steering_angle), np.cos(steering_angle), 0], [0, 0, 1]])
                rotation = np.matmul(pw_rotation, transmit_rotation)
                normal = np.matmul(pw_rotation, np.array([1, 0, 0])) 
            else:
                transmit_rotation = transducer_transform.rotation.as_euler('ZYX')
                nl_transform = geometry.Transform(rotation = transmit_rotation) * transducer.ray_transforms[index - running_index_list[transducer_count]]
                normal = nl_transform.apply_to_point((1, 0, 0)) # do we need this for the focused case?????
                
            xxx, yyy, zzz = np.meshgrid(x - transmit_position[0], y - transmit_position[1], z - transmit_position[2], indexing='ij')
            distances = np.stack([xxx, yyy, zzz], axis=0)
            
            transmit_dists = np.abs(np.einsum('ijkl,i->jkl', distances, normal))
            
            for centroid, rf_series in zip(element_centroids, preprocessed_data):
                xxx, yyy, zzz = np.meshgrid(x - centroid[0], y - centroid[1], z - centroid[2], indexing='ij')
                element_dists = np.sqrt(xxx**2 + yyy**2 + zzz**2)
                travel_times = ((transmit_dists + element_dists)/c0/dt).astype(np.int32)

                # image_matrix += rf_series[travel_times]
                
                for i in range(len(x)):
                    for j in range(len(y)):
                        for k in range(len(z)):
                            image_matrix[i][j][k] += rf_series[travel_times[i][j][k]]

        return image_matrix
                                                                                
                
                
            
            

