import numpy as np
import scipy
import os
import glob
import matplotlib.pyplot as plt
import tqdm
import sys
import multiprocessing
from scipy import interpolate
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.ndimage import gaussian_filter1d
import time

from utils import utils
from utils import geometry
from transducer import Focused, Planewave, Transducer
from phantom import Phantom
from transducer_set import TransducerSet
from simulation import Simulation, SimProperties
from sensor import Sensor
from .experiment import Experiment

from scipy.signal import hilbert

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
        
        
        if workers > 1:
            with multiprocessing.Pool(workers) as p:
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
        

    def get_image(self, bounds=None, matsize=256, dimensions=3, downsample = 1, workers=8, tgc=1):
        assert dimensions in [2,3], print("Image can be 2 or 3 dimensional")
        assert (downsample > 0 and downsample <= 1), print("Downsample must be a float on (0,1]")
        
        times, coords, processed = self.preprocess_data(global_transforms=True, workers=workers, attenuation_factor=tgc)

        if bounds is None:
            flat_coords = np.concatenate(coords, axis=0).reshape(-1,3)
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
            X, Y = np.meshgrid(X, Y, indexing='ij')
        else:
            X, Y, Z = np.meshgrid(X, Y, Z, indexing='ij')

        signals = []
        count = 0
        
        for i, transducer in tqdm.tqdm(enumerate(self.transducer_set.transducers)):
            subset_coords = np.stack(coords[count:int(count+transducer.get_num_rays())], axis=0).reshape(-1,3)
            subset_processed = np.stack(processed[count:int(count+transducer.get_num_rays())], axis=0).reshape(-1)
            
            if downsample != 1:
                subset_processed = gaussian_filter1d(subset_processed.reshape(-1,transducer.get_num_rays()), int(1/downsample), axis=-1).reshape(-1)
                subset_coords = subset_coords[::int(1/downsample)]
                subset_processed = subset_processed[::int(1/downsample)]
            
            if dimensions == 2:
                interp = LinearNDInterpolator(subset_coords[:,:2], subset_processed)
                signals.append(interp(X, Y))
            else:
                interp = NearestNDInterpolator(subset_coords, subset_processed)
                signal = interp(X, Y, Z)
                
                ray_length = subset_coords.shape[0] / transducer.get_num_rays()
                convex_coords = np.stack([subset_coords[0]] + [subset_coords[int(i*ray_length-1)] for i in range(1, transducer.get_num_rays())] + [subset_coords[-1]])
                convex_hull_mask = utils.compute_convex_hull_mask(convex_coords, np.stack([X,Y,Z], axis=-1))
                
                signals.append(signal * convex_hull_mask)
                
            count += transducer.get_num_rays()
        
        combined_signals = np.stack(signals, axis=0)
        masked_signals = np.ma.masked_array(combined_signals, np.isnan(combined_signals))
        image = np.ma.average(masked_signals, axis=0)
        image = image.filled(np.nan)
        
        return image
    
    
    def get_signals(self, bounds=None, matsize=256, dimensions=3, downsample = 1, workers=8, tgc=1):
        assert dimensions in [2,3], print("Image can be 2 or 3 dimensional")
        assert (downsample > 0 and downsample <= 1), print("Downsample must be a float on (0,1]")
        
        times, coords, processed = self.preprocess_data(global_transforms=False, workers=workers, attenuation_factor=tgc)

        if bounds is None:
            flat_coords = np.concatenate(coords, axis=0).reshape(-1,3)
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
            X, Y = np.meshgrid(X, Y, indexing='ij')
        else:
            X, Y, Z = np.meshgrid(X, Y, Z, indexing='ij')

        signals = []
        count = 0
        for i, transducer in tqdm.tqdm(enumerate(self.transducer_set.transducers)):
            subset_coords = np.stack(coords[count:int(count+transducer.get_num_rays())], axis=0).reshape(-1,3)
            subset_processed = np.stack(processed[count:int(count+transducer.get_num_rays())], axis=0).reshape(-1)
            
            if downsample != 1:
                subset_processed = gaussian_filter1d(subset_processed.reshape(-1,transducer.get_num_rays()), int(1/downsample), axis=-1).reshape(-1)
                subset_coords = subset_coords[::int(1/downsample)]
                subset_processed = subset_processed[::int(1/downsample)]
                            
            if dimensions == 2:
                interp = LinearNDInterpolator(subset_coords[:,:2], subset_processed)
                signals.append(interp(X, Y))
            else:
                interp = NearestNDInterpolator(subset_coords, subset_processed)
                signal = interp(X, Y, Z)
                # interp = LinearNDInterpolator(subset_coords, subset_processed)
                
                ray_length = subset_coords.shape[0] / transducer.get_num_rays()
                convex_coords = np.stack([subset_coords[0]] + [subset_coords[int(i*ray_length-1)] for i in range(1, transducer.get_num_rays())] + [subset_coords[-1]])
                convex_hull_mask = utils.compute_convex_hull_mask(convex_coords, np.stack([X,Y,Z], axis=-1))
                
                signals.append(signal * convex_hull_mask)
                
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
    
    def compound(self, workers=8, resolution_multiplier=1, local=False, combine=True, pressure_field=None, pressure_field_resolution=None, return_local=False, attenuation_factor=1, volumetric=False):

        if isinstance(self.transducer_set[0], Focused):
            # do nothing
            pass
        else:
            # still do nothing
            pass

        matrix_dims = self.phantom.matrix_dims
        voxel_dims = self.phantom.voxel_dims
        
        c0 = self.phantom.baseline[0]
        dt = (self.results[0][0][-1] - self.results[0][0][0]) / self.results[0][0].shape[0]
                
        resolution = max(dt*c0, 2*c0/self.transducer_set.get_lowest_frequency()) / 4 / resolution_multiplier # make sure this works
        
        x = np.arange(-matrix_dims[0]*voxel_dims[0]/2 + voxel_dims[0]/2, matrix_dims[0]*voxel_dims[0]/2 + voxel_dims[0]/2, step=resolution)
        y = np.arange(-matrix_dims[1]*voxel_dims[1]/2 + voxel_dims[1]/2, matrix_dims[1]*voxel_dims[1]/2 + voxel_dims[1]/2, step=resolution)
        z = np.arange(-matrix_dims[2]*voxel_dims[2]/2 + voxel_dims[2]/2, matrix_dims[2]*voxel_dims[2]/2 + voxel_dims[2]/2, step=resolution)
        
        image_matrix = np.zeros((len(x), len(y), len(z)))
        
        # note that origin is at center of the 3d image in global coordinate system

        element_centroids = self.__get_element_centroids()

        arguments = []

        transducer_count = 0
        transducer, transducer_transform = self.transducer_set[transducer_count]
        running_index_list = np.cumsum([transducer.get_num_rays() for transducer in self.transducer_set.transducers])

        for index in tqdm.tqdm(range(len(self.results))):
            
            if index > running_index_list[transducer_count] - 1:
                transducer_count += 1
                transducer, transducer_transform = self.transducer_set[transducer_count]
            
            arguments.append((index, running_index_list, transducer_count, transducer, transducer_transform, x, y, z, c0, dt, element_centroids, resolution, return_local, pressure_field, pressure_field_resolution, attenuation_factor, volumetric))
        
        with multiprocessing.Pool(workers) as p:
            if not local:
                image_matrices = list(p.starmap(self.scanline_reconstruction_canonical, arguments))
            else:
                image_matrices = list(p.starmap(self.scanline_reconstruction_refined, arguments))

        if combine:
            return np.sum(np.stack(image_matrices, axis=0), axis=0)
        else:
            return image_matrices
    
    
    # def scanline_reconstruction_canonical(self, index, running_index_list, transducer_count, transducer, transducer_transform, x, y, z, c0, dt, element_centroids, resolution, return_local, pressure_field=None, pressure_field_resolution=None):
    #     # fetch steering angle
    #     if index > running_index_list[transducer_count] - 1:
    #         transducer_count += 1
    #         transducer, transducer_transform = self.transducer_set[transducer_count]
    #     steering_angle = transducer.steering_angles[index - running_index_list[transducer_count]]

    #     # get dt
    #     dt = (self.results[index][0][-1] - self.results[index][0][0]) / (self.results[index][0].shape[0]-1)
        
    #     # run transducer signal preprocessing with signal enveloping (according to canonical reconstruction)
    #     preprocessed_data = transducer.preprocess(self.results[index][1], self.results[index][0], self.sim_properties, window_factor=8, saft=True, demodulate=True)
        
    #     # pad the timesignal if duration < long diagonal
    #     if len(preprocessed_data.shape) == 2:
    #         preprocessed_data = np.pad(preprocessed_data, ((0,0),(0,int(preprocessed_data.shape[1]*1.73))),)
                
    #     if isinstance(transducer, Planewave):
    #         steering_transform = geometry.Transform(rotation=[steering_angle,0,0])
    #         timedelay = transducer.width / 2 * np.abs(np.sin(np.max(transducer.steering_angles))) # timedelay gets padded on according to the max delay
    #     else:
    #         steering_transform = transducer.ray_transforms[index - running_index_list[transducer_count]]
    #         timedelay = 0

    #     beam_transform = transducer_transform * steering_transform
        
    #     global_bounds = np.array([[np.min(x), np.max(x)], [np.min(y), np.max(y)],[np.min(z), np.max(z)]])
    #     xs, ys, zs = np.meshgrid(global_bounds[0], global_bounds[1], global_bounds[2], indexing='ij')
    #     global_vertices = np.stack((xs.flatten(), ys.flatten(), zs.flatten()), axis=-1)
    #     local_vertices = beam_transform.apply_to_points(global_vertices, inverse=True)
        
    #     local_mins = np.min(local_vertices, axis=0)
    #     local_maxs = np.max(local_vertices, axis=0)
    #     local_x = np.arange(local_mins[0], local_maxs[0]+resolution, step=resolution)
    #     local_y = np.arange(local_mins[1], local_maxs[1]+resolution, step=resolution)
    #     # local_z = np.arange(local_mins[2], local_maxs[2]+resolution, step=resolution)
    #     local_z = np.array([0])
    #     xxx, yyy, zzz = np.meshgrid(local_x, local_y, local_z, indexing='ij')

    #     if isinstance(transducer, Focused): 
    #         assert self.sensor.aperture_type == "extended_aperture", "For focused transducers, the sensor aperture type must be 'extended_aperture'"
    #         sensors_per_el = transducer.get_sensors_per_el()
    #         transmit_centroids = np.zeros((transducer.get_num_elements(), 3))
    #         pos = 0
    #         for entry in range(transducer.get_num_elements()):
    #             transmit_centroids[entry] = np.mean(transducer.sensor_coords[pos:pos+sensors_per_el, :], axis = 0)
    #             pos += sensors_per_el
    #         element_centroids = beam_transform.apply_to_points(element_centroids, inverse=True)
    #         label = transducer.get_label()
    #         index = self.transducer_set.find_transducer(label)
    #         start_element = 0
    #         for t in self.transducer_set.transducers[:index]:
    #             start_element += t.get_num_elements()
    #         element_centroids[start_element:transducer.get_num_elements(), :] = transmit_centroids
    #         transmit_dists = np.sqrt(xxx**2 + yyy**2 + zzz**2)
    #         # timedelays = np.round(((np.linalg.norm(transmit_centroids - np.array(((transducer.focus_azimuth,0,0))), axis=1)) - transducer.focus_azimuth) / c0 / dt).astype(np.int32)
    #         # timedelays = transducer.not_transducer.beamforming_delays
    #     else:
    #         if self.sensor.aperture_type == "transmit_as_receive":
    #             # recreate the sensor mask
    #             transmit_centroids = np.linspace(-transducer.not_transducer.transducer.element_pitch * len(transducer.not_transducer.active_elements) / 2,
    #                                                                          transducer.not_transducer.transducer.element_pitch * len(transducer.not_transducer.active_elements) / 2,
    #                                                                          len(transducer.not_transducer.active_elements))
    #             # Account for PML?
    #             transmit_centroids = np.stack((np.zeros_like(transmit_centroids), transmit_centroids, np.zeros_like(transmit_centroids)), axis=1)
    #             element_centroids = steering_transform.apply_to_points(transmit_centroids, inverse=True)
    #         else:
    #             element_centroids = beam_transform.apply_to_points(element_centroids, inverse=True)
                
    #         distances = np.stack([xxx, yyy, zzz], axis=0)
    #         transmit_dists = np.abs(np.einsum('ijkl,i->jkl', distances,  np.array((1,0,0))))

    #     local_image_matrix = np.zeros((len(local_x), len(local_y), len(local_z)))
        
    #     apodizations = np.ones((len(local_x), len(local_y), len(local_z)))
            
    #     el2el_dists = (np.sqrt(element_centroids[:,0] ** 2 + element_centroids[:,1] ** 2 + element_centroids[:,2] ** 2) + transducer.width / 2) * 1.25
                
    #     lx, ly, lz = np.meshgrid(local_x, local_y, local_z, indexing='ij')
    #     for i, (centroid, rf_series) in enumerate(zip(element_centroids, preprocessed_data)):
    #         element_dists = np.sqrt((lx - centroid[0]) ** 2 + (ly - centroid[1]) ** 2 + (lz - centroid[2]) ** 2)
            
    #         if self.sensor.aperture_type == "transmit_as_receive":
    #             travel_times = np.round((transmit_dists + element_dists)/c0/dt).astype(np.int32)
    #             windowed_times = travel_times
    #             denominator = len(preprocessed_data)
    #         else:
    #             travel_times = np.round((transmit_dists + element_dists + timedelay)/c0/dt).astype(np.int32)
    #             windowed_times = np.where(transmit_dists + element_dists + timedelay < el2el_dists[i], 0, travel_times)
    #             denominator = transducer.get_num_elements()
                
    #         local_image_matrix[:len(local_x), :len(local_y), :len(local_z)] += rf_series[windowed_times[:len(local_x), :len(local_y), :len(local_z)]] * apodizations[:len(local_x), :len(local_y), :len(local_z)] / denominator
        
    #     if return_local:
    #         return local_image_matrix
    #     flat = local_image_matrix.flatten()
    #     local_coords = np.stack((xxx.flatten(), yyy.flatten(), zzz.flatten()), axis=-1)
    #     local_2_global = beam_transform.apply_to_points(local_coords)
    #     interpolator = NearestNDInterpolator(local_2_global, flat)
    #     z = np.array([0]) # for 2d
    #     gx,gy,gz = np.meshgrid(x, y, z, indexing='ij')
    #     global_signal = interpolator(gx,gy,gz).reshape(len(x), len(y), len(z))
    #     return global_signal
    
    
    def scanline_reconstruction_refined(self, index, running_index_list, transducer_count, transducer, transducer_transform, x, y, z, c0, dt, element_centroids, resolution, return_local, pressure_field=None, pressure_field_resolution=None, attenuation_factor=None, volumetric=False):
        # fetch steering angle
        if index > running_index_list[transducer_count] - 1:
            transducer_count += 1
            transducer, transducer_transform = self.transducer_set[transducer_count]
        steering_angle = transducer.steering_angles[index - running_index_list[transducer_count]]

        # get dt
        dt = (self.results[index][0][-1] - self.results[index][0][0]) / (self.results[index][0].shape[0]-1)
        
        # run transducer signal preprocessing
        preprocessed_data = transducer.preprocess(self.results[index][1], self.results[index][0], self.sim_properties, window_factor=8, saft=True, attenuation_factor=attenuation_factor)
        
        # pad the timesignal if duration < long diagonal
        if len(preprocessed_data.shape) == 2:
            preprocessed_data = np.pad(preprocessed_data, ((0,0),(0,int(preprocessed_data.shape[1]*1.73))),)
                
        if isinstance(transducer, Planewave):
            steering_transform = geometry.Transform(rotation=[steering_angle,0,0])
            timedelay = transducer.width / 2 * np.abs(np.sin(np.max(transducer.steering_angles))) # timedelay gets padded on according to the max delay
        else:
            steering_transform = transducer.ray_transforms[index - running_index_list[transducer_count]]
            timedelay = 0

        if return_local:
            distance = np.linalg.norm(transducer_transform.translation)
            beam_transform = geometry.Transform([0,0,0],[-distance,0,0]) * steering_transform
        else:
            beam_transform = transducer_transform * steering_transform
        
        global_bounds = np.array([[np.min(x), np.max(x)], [np.min(y), np.max(y)],[np.min(z), np.max(z)]])
        xs, ys, zs = np.meshgrid(global_bounds[0], global_bounds[1], global_bounds[2], indexing='ij')
        global_vertices = np.stack((xs.flatten(), ys.flatten(), zs.flatten()), axis=-1)
        local_vertices = beam_transform.apply_to_points(global_vertices, inverse=True)
        
        local_mins = np.min(local_vertices, axis=0)
        local_maxs = np.max(local_vertices, axis=0)
        local_x = np.arange(local_mins[0], local_maxs[0]+resolution, step=resolution)
        local_y = np.arange(local_mins[1], local_maxs[1]+resolution, step=resolution)
        if volumetric:
            local_z = np.arange(local_mins[2], local_maxs[2]+resolution, step=resolution)
        else:
            local_z = np.array([0])
        xxx, yyy, zzz = np.meshgrid(local_x, local_y, local_z, indexing='ij')

        if isinstance(transducer, Focused): 
            assert self.sensor.aperture_type == "extended_aperture", "For focused transducers, the sensor aperture type must be 'extended_aperture'"
            sensors_per_el = transducer.get_sensors_per_el()
            transmit_centroids = np.zeros((transducer.get_num_elements(), 3))
            pos = 0
            for entry in range(transducer.get_num_elements()):
                transmit_centroids[entry] = np.mean(transducer.sensor_coords[pos:pos+sensors_per_el, :], axis = 0)
                pos += sensors_per_el
            element_centroids = beam_transform.apply_to_points(element_centroids, inverse=True)
            label = transducer.get_label()
            index = self.transducer_set.find_transducer(label)
            start_element = 0
            for t in self.transducer_set.transducers[:index]:
                start_element += t.get_num_elements()
            element_centroids[start_element:transducer.get_num_elements(), :] = transmit_centroids
            transmit_dists = np.sqrt(xxx**2 + yyy**2 + zzz**2)
            # timedelays = np.round(((np.linalg.norm(transmit_centroids - np.array(((transducer.focus_azimuth,0,0))), axis=1)) - transducer.focus_azimuth) / c0 / dt).astype(np.int32)
            # timedelay = transducer.not_transducer.beamforming_delays * dt
            timedelay = 0
        else:
            if self.sensor.aperture_type == "transmit_as_receive":
                # recreate the sensor mask
                transmit_centroids = np.linspace(-transducer.not_transducer.transducer.element_pitch * len(transducer.not_transducer.active_elements) / 2,
                                                                             transducer.not_transducer.transducer.element_pitch * len(transducer.not_transducer.active_elements) / 2,
                                                                             len(transducer.not_transducer.active_elements))
                # Account for PML?
                transmit_centroids = np.stack((np.zeros_like(transmit_centroids), transmit_centroids, np.zeros_like(transmit_centroids)), axis=1)
                element_centroids = steering_transform.apply_to_points(transmit_centroids, inverse=True)
            else:
                element_centroids = beam_transform.apply_to_points(element_centroids, inverse=True)
                
            distances = np.stack([xxx, yyy, zzz], axis=0)
            transmit_dists = np.abs(np.einsum('ijkl,i->jkl', distances,  np.array((1,0,0))))

        local_image_matrix = np.zeros((len(local_x), len(local_y), len(local_z)))
        # local_image_matrix = np.ones((len(local_x), len(local_y), len(local_z))) # for multiplication, initialize to ones
        
        if pressure_field is not None:
            assert pressure_field_resolution is not None, "Pressure field resolution must be provided if pressure field is provided"
            vox_size = pressure_field_resolution
            normalized_pfield = pressure_field/np.sum(pressure_field)
            pfield_xs = np.arange(0, normalized_pfield.shape[0]*vox_size, step=vox_size)
            pfield_ys = np.arange(-normalized_pfield.shape[1]/2*vox_size+vox_size/2, normalized_pfield.shape[1]/2*vox_size+vox_size/2, step=vox_size)
            f = interpolate.interp2d(pfield_ys, pfield_xs, normalized_pfield, kind='linear', fill_value=0)
            apodizations = np.repeat(f(local_y, local_x,), len(local_z)).reshape(len(local_x), len(local_y), len(local_z))
        else:
            apodizations = np.ones((len(local_x), len(local_y), len(local_z)))
            
        el2el_dists = (np.sqrt(element_centroids[:,0] ** 2 + element_centroids[:,1] ** 2 + element_centroids[:,2] ** 2) + transducer.width / 2) * 1.25
                
        lx, ly, lz = np.meshgrid(local_x, local_y, local_z, indexing='ij')
        for i, (centroid, rf_series) in enumerate(zip(element_centroids, preprocessed_data)):
            element_dists = np.sqrt((lx - centroid[0]) ** 2 + (ly - centroid[1]) ** 2 + (lz - centroid[2]) ** 2)
            
            if self.sensor.aperture_type == "transmit_as_receive":
                travel_times = np.round((transmit_dists + element_dists)/c0/dt).astype(np.int32)
                windowed_times = travel_times
                denominator = len(preprocessed_data)
            else:
                travel_times = np.round((transmit_dists + element_dists + timedelay)/c0/dt).astype(np.int32)
                windowed_times = np.where(transmit_dists + element_dists + timedelay < el2el_dists[i], 0, travel_times)
                denominator = transducer.get_num_elements()
                
            local_image_matrix[:len(local_x), :len(local_y), :len(local_z)] += rf_series[windowed_times[:len(local_x), :len(local_y), :len(local_z)]] * apodizations[:len(local_x), :len(local_y), :len(local_z)] / denominator
            
            # At some point, test reconstruction with multiplication here, requires normalization
            # normalized = rf_series[travel_times[:len(local_x), :len(local_y), :len(local_z)]] * apodizations[:len(local_x), :len(local_y), :len(local_z)]
            # normalized = normalized / np.sum(rf_series[travel_times[:len(local_x), :len(local_y), :len(local_z)]] * apodizations[:len(local_x), :len(local_y), :len(local_z)]) / dt
            # local_image_matrix[:len(local_x), :len(local_y), :len(local_z)] *= normalized

        local_image_matrix = np.abs(hilbert(local_image_matrix, axis = 0))
        
        flat = local_image_matrix.flatten()
        local_coords = np.stack((xxx.flatten(), yyy.flatten(), zzz.flatten()), axis=-1)
        local_2_global = beam_transform.apply_to_points(local_coords)
        interpolator = NearestNDInterpolator(local_2_global, flat)
        if not volumetric:
            z = np.array([0])
        gx,gy,gz = np.meshgrid(x, y, z, indexing='ij')
        global_signal = interpolator(gx,gy,gz).reshape(len(x), len(y), len(z))
        return global_signal
    
    
    def selective_compound(self, transducers, workers=8, resolution_multiplier=1, local=False, pressure_field=None, combine=True):

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
                
        resolution = max(dt*c0, 2*c0/self.transducer_set.get_lowest_frequency()) / 4 / resolution_multiplier # make sure this works
        
        x = np.arange(-matrix_dims[0]*voxel_dims[0]/2 + voxel_dims[0]/2, matrix_dims[0]*voxel_dims[0]/2 + voxel_dims[0]/2, step=resolution)
        y = np.arange(-matrix_dims[1]*voxel_dims[1]/2 + voxel_dims[1]/2, matrix_dims[1]*voxel_dims[1]/2 + voxel_dims[1]/2, step=resolution)
        z = np.arange(-matrix_dims[2]*voxel_dims[2]/2 + voxel_dims[2]/2, matrix_dims[2]*voxel_dims[2]/2 + voxel_dims[2]/2, step=resolution)
        
        image_matrix = np.zeros((len(x), len(y), len(z)))
        
        # note that origin is at center of the 3d image in global coordinate system

        element_centroids = self.__get_element_centroids()

        arguments = []

        transducer_count = 0
        transducer, transducer_transform = self.transducer_set[transducer_count]
        running_index_list = np.cumsum([transducer.get_num_rays() for transducer in self.transducer_set.transducers])

        for index in tqdm.tqdm(range(len(self.results))):
            if index > running_index_list[transducer_count] - 1:
                transducer_count += 1
                transducer, transducer_transform = self.transducer_set[transducer_count]
            
            if transducer_count not in transducers:
                    continue
            arguments.append((index, running_index_list, transducer_count, transducer, transducer_transform, x, y, z, c0, dt, element_centroids, resolution, pressure_field))
        
        print(f'running reconstruction on {len(arguments)} rays')
        with multiprocessing.Pool(workers) as p:
            if not local:
                image_matrices = list(p.starmap(self.scanline_reconstruction, arguments))
            else:
                image_matrices = list(p.starmap(self.scanline_reconstruction_refined, arguments))

        if combine:
            return np.sum(np.stack(image_matrices, axis=0), axis=0)
        else:
            return image_matrices