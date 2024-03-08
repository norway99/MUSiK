import numpy as np
import scipy
import os
import glob
import matplotlib.pyplot as plt
import tqdm
import sys
sys.path.insert(0, '../utils')

import utils
import geometry
from phantom import Phantom
from transducer_set import TransducerSet
from simulation import Simulation, SimProperties
from sensor import Sensor
from experiment import Experiment


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
        for transducer in experiment.transducer_set.transducers:
            if not isinstance(transducer, Focused):
                print("Warning: attempting to instantiate DAS reconstruction class but transducer set does not exclusively contain focused transducers.")
                break
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
    
        
    def preprocess_data(self,):
        coords = []
        time = []
        processed = []
        transducer_count = 0
        transducer, transducer_transform = self.transducer_set[transducer_count]
        running_index_list = np.cumsum([transducer.get_num_rays() for transducer in self.transducer_set.transducers])
        for index in tqdm.tqdm(range(len(self.results))):
            if index > running_index_list[transducer_count] - 1:
                transducer_count += 1
                transducer, transducer_transform = self.transducer_set[transducer_count]
                transform = transducer_transform * transducer.ray_transforms[index - running_index_list[transducer_count]]
                
            processed.append(transducer.preprocess(transducer.make_scan_line(self.results[index][1]), self.results[index][0], self.sim_properties))
            coords.append(self.__time_to_coord(self.results[index][0], transform))
            time.append(self.results[index][0])
            
        return time, coords, processed
    
    
    # def plot_scatter(self,):
    #     coords, processed, time = self.preprocess_data()
    #     coords = np.stack(coords, axis=0)
    #     processed = np.stack(processed, axis=0)
        
    #     fig, ax = plt.subplots(1,1, figsize=(8,8))
        
    #     coords = np.reshape(coords, (-1,3))
    #     processed = np.reshape(processed, (-1,))
        
    #     ax.scatter(coords[:,0], coords[:,1], c=np.clip(processed, 0, 10000), s=2.5, cmap='gray')
        
    #     ax.set_aspect('equal')
        
        
# broken
    def plot_scatter(self, scale=5000):
       
        colorme = lambda x: (     [1,0,0] if x % 7 == 0 
                             else [0,1,0] if x % 7 == 1 
                             else [0,0,1] if x % 7 == 2 
                             else [1,1,0] if x % 7 == 3 
                             else [1,0,1] if x % 7 == 4 
                             else [0,1,1] if x % 7 == 5 
                             else [1,1,1])
        
        time, coords, processed = self.preprocess_data()
        coords = np.stack(coords, axis=0)
        processed = np.stack(processed, axis=0)
        time = np.stack(time, axis=0)
        
        transducer_lens = [t.get_num_rays() for t in self.transducer_set.transducers]
        base_color = np.array([colorme(i) for i in range(len(transducer_lens)) for j in range(transducer_lens[i] * len(time[i]))])
        
        fig, ax = plt.subplots(1,1, figsize=(8,8))
        
        coords = np.reshape(coords, (-1,3))
        processed = np.reshape(processed, (-1,))
        time = np.reshape(time, (-1,))
        base_color = np.reshape(base_color, (-1,3))
        intensity = np.clip(processed, 0, scale)/scale
        
        colors = np.array([1,1,1]) - np.broadcast_to(intensity[:,None], base_color.shape) * base_color
        
        ax.scatter(coords[:,0], coords[:,1], c=colors, s=time*100000, alpha = 0.002)
        
        ax.set_aspect('equal')
        
class Compounding(Reconstruction):

    def __init__(self,
                 experiment = None):
        ts = experiment.transducer_set
    for transducer in ts:
        if not isinstance(tranducer, Planewave):
            print("Warning: attempting to instantiate Compounding reconstruction class but transducer set does not exclusively contain plane-wave tranducers.")
            break
        super().__init__(experiment)

    def __get_element_centroids(self): # in global coordinates
        sensor_coords = self.sensor.sensor_coords
        sensors_per_el = self.sensors_per_el
        element_centroids = np.zeros((sensors_per_el.size, 3))
        pos = 0
        for entry in range(sensors_per_el.size):
            element_centroids[entry] = np.mean(sensor_coords[pos:pos+sensors_per_el[entry], :], axis = 0)
            pos += sensors_per_el[entry]
        return element_centroids
    
    def compound(self):

        matrix_dims = self.phantom.matrix_dims
        voxel_dims = self.phantom.voxel_dims
        c0 = self.phantom.baseline[0]
        dt = self.kgrid.dt # not the correct way to access the kgrid
        
        # need to choose lateral, axial, and elevation resolutions
        
        resolution = max(dt*c0, 2*c0/self.transducer_set.get_lowest_frequency()) # make sure this works
        
        x = np.linspace(-matrix_dims[0]*voxel_dims[0]/2, matrix_dims[0]*voxel_dims[0]/2, retstep=resolution[0])
        y = np.linspace(-matrix_dims[1]*voxel_dims[1]/2, matrix_dims[1]*voxel_dims[1]/2, retstep=resolution[1])
        z = np.linspace(-matrix_dims[2]*voxel_dims[2]/2, matrix_dims[2]*voxel_dims[2]/2, retstep=resolution[2])
        
        xxx, yyy, zzz = np.meshgrid(x, y, z)

        image_matrix = np.zeros(len(x), len(y), len(z))
        
        # note that origin is at center of the 3d image in global coordinate system
        
        element_centroids = self.__get_element_centroids()
        transducer_count = 0
        for index in tqdm.tqdm(range(len(self.results))):
            if index > running_index_list[transducer_count] - 1:
                transducer_count += 1
                transducer, transducer_transform = self.transducer_set[transducer_count]
                steering_angle = transducer.steering_angles[index - running_index_list[transducer_count]]

            preprocessed_data = transducer.preprocess(self.results[index][1], self.results[index][0], self.sim_properties)
            
            transmit_position = transducer_transform.translation
            transmit_rotation = transducer_transform.rotation
            nl_transform = Transform(rotation = transmit_rotation) * transducer.ray_transforms[index - running_index_list[transducer_count]]
            normal = nl_transform.apply_to_point((1, 0, 0)) # normal vector to plane wave shot

            transmit_dists = np.absolute(np.dot(np.array([xxx-transmit_position[0], yyy-transmit_position[1], zzz-transmit_position[2]]), normal))

            for centroid, rf_series in zip(element_centroids, preprocessed_data):
                element_dists = np.sqrt((xxx-centroid[0])**2 + (yyy-centroid[1])**2 + (zzz-centroid[2])**2)
                travel_times = (transmit_dists + element_dists)/c0

                #parallelize this (with multithreading?)
                for i in range(len(x)):
                    for j in range(len(y)):
                        for k in range(len(z)):
                            image_matrix[i][j][k] += rf_series[travel_times[i][j][k]]

        return image_matrix
                                                                                
                
                
            
            

