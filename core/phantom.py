import numpy as np
import scipy
import scipy.ndimage
import os

import sys
sys.path.append('../utils')
import utils
import geometry
from .tissue import Tissue
from scipy.interpolate import RegularGridInterpolator, NearestNDInterpolator


class Phantom:
    """
    A class representing a phantom for simulating ultrasound imaging.

    Attributes:
    - source_path (str): The path to the source directory containing the phantom data.
    - voxel_dims (tuple): The dimensions of each voxel in the phantom, in meters.
    - matrix_dims (tuple): The dimensions of the phantom matrix.
    - seed (int): The seed for the random number generator used to generate the phantom.
    - rng (numpy.random.Generator): The random number generator used to generate the phantom.
    - mask (numpy.ndarray): A 3D array representing the phantom mask.
    - tissues (dict): A dictionary of Tissue objects representing the different types of tissue in the phantom.

    Methods:
    - save(filepath): Saves the phantom to the specified file path.
    - load(source_path): Loads the phantom from the specified source directory.
    - create_from_image(): Sets the phantom mask and estimates the tissues from a Hounsfield unit-scaled image.
    - create_from_list(): Sets the phantom mask and reads the tissues by supplying a list of shapes and corresponding tissues.
    - add_tissue(tissue): Adds a tissue to the phantom.
    - remove_tissue(tissue): Removes a tissue from the phantom.
    - add_tissue_sphere(centroid, radius, tissue): Adds a spherical region of tissue to the phantom.
    - get_tissues(): Returns a dictionary of the different types of tissue in the phantom.
    - get_tissue_mask(): Returns a mask containing only tissues of a particular type.
    - get_mask(): Returns the phantom mask.
    - get_dimensions(): Returns the dimensions of the phantom matrix.
    - get_voxel_dims(): Returns the dimensions of each voxel in the phantom, in meters.
    - generate_tissue(tissue, rand_like=None, order=1): Generates a 3D array representing a tissue in the phantom.
    - get_complete(): Returns a 3D array representing the complete phantom.
    - render(): Renders the phantom.
    """

    def __init__(self, 
                 source_path = None,
                 voxel_dims = (1e-3,1e-3,1e-3),
                 matrix_dims = (256,256,256),
                 baseline = (1500, 1000),
                 seed = 5678,
                 from_mask=True,
                 ):
        
        # initialize from source if exists
        if source_path is not None:
            self.load(source_path)
            return 1
        
        # otherwise initialize empty water phantom
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.voxel_dims = np.array(voxel_dims)
        self.mask = np.zeros(matrix_dims, dtype = np.float32)
        self.tissues = {'water':Tissue(name = 'water', label = 0, c=1500, rho=1000, sigma=0, scale=0.1)}
        self.matrix_dims = np.array(matrix_dims)
        self.baseline = baseline
        self.from_mask = from_mask
        self.complete = None
        
        
    # save phantom to source dir containing tissues, mask, and source
    def save(self, filepath):
        os.makedirs(filepath, exist_ok=True)
        utils.dict_to_json(self.__save_tissues(), filepath + '/tissues.json')
        utils.save_array(self.mask, filepath + '/mask.npz')
        if self.complete is not None:
            utils.save_array(self.complete, filepath + '/complete.npy')
        dictionary = self.__dict__.copy()
        dictionary.pop('mask')
        dictionary.pop('tissues')
        dictionary.pop('rng')
        dictionary.pop('complete')
        utils.dict_to_json(dict(dictionary), filepath + '/source.json')
    

	# load phantom from source dir
    @classmethod
    def load(cls, source_path):
        phantom = cls()
        assert os.path.exists(source_path), 'provided path does not exist'
        assert os.path.exists(source_path+'/tissues.json'), 'missing tissues file'
        assert os.path.exists(source_path+'/source.json'), 'missing source file'
        assert os.path.exists(source_path+'/mask.npz') or os.path.exists(source_path+'/mask.npy'), 'missing mask file'
        
        phantom.tissues = phantom.__load_tissues(utils.json_to_dict(source_path+'/tissues.json'))
        phantom.mask = utils.load_array(source_path+'/mask')
        if os.path.exists(source_path+'/complete.npy'):
            phantom.complete = utils.load_array(source_path+'/complete.npy')
        elif os.path.exists(source_path+'/complete.npz'):
            phantom.complete = utils.load_array(source_path+'/complete.npz')
        else:
            phantom.complete = None
        source = utils.json_to_dict(source_path+'/source.json')
        
        phantom.rng = np.random.default_rng(source['seed'])
        phantom.voxel_dims = source['voxel_dims']
        phantom.matrix_dims = source['matrix_dims']
        return phantom
    
    
    # save tissues to dictionary
    def __save_tissues(self,):
        dictionary = {}
        for tissue in self.tissues.keys():
            dictionary[tissue] = self.tissues[tissue].save()
        return dictionary
    
    
    def __load_tissues(self, dictionary):
        for tissue in dictionary.keys():
            t = Tissue().load(dictionary[tissue])
            self.tissues[tissue] = t
        return self.tissues
	
 
	# set mask and estimate tissues from a houndsfield unit-scaled image
    def create_from_image(self, image, input_voxel_size, target_voxel_size=None, transfer_fn=None):
        if target_voxel_size is None:
            target_voxel_size = self.voxel_dims
        if transfer_fn is None:
            transfer_fn = lambda x: (x + np.amin(x)) / (np.amax(x) - np.amin(x))
            
        if type(image) == np.ndarray:
            data = image
        
        data = transfer_fn(data)

        x = np.arange(0, data.shape[0])
        y = np.arange(0, data.shape[1])
        z = np.arange(0, data.shape[2])

        assert len(input_voxel_size) == 3, 'input voxel size must be a tuple of length 3'
        assert len(target_voxel_size) == 3, 'target voxel size must be a tuple of length 3'

        transformed_x = x * input_voxel_size[0] / target_voxel_size[0]
        transformed_y = y * input_voxel_size[1] / target_voxel_size[1]
        transformed_z = z * input_voxel_size[2] / target_voxel_size[2]

        from scipy.interpolate import RegularGridInterpolator
        interp = RegularGridInterpolator((transformed_x, transformed_y, transformed_z), data)
        # interp = NearestNDInterpolator((transformed_x, transformed_y, transformed_z), data)

        points = np.stack(np.meshgrid(np.arange(0, transformed_x[-1]), np.arange(0, transformed_y[-1]), np.arange(0, transformed_z[-1]), indexing='ij'), axis=-1)
        
        if points.shape[0] * points.shape[1] * points.shape[2] > 5e8:
            print('desired phantom array size is very large (>500,000,000 voxels), consider supplying a larger target_voxel_size or cropping the input image')
        if points.shape[0] * points.shape[1] * points.shape[2] > 2e9:
            print('desired phantom array size is too large (>2e9 voxels), consider supplying a larger target_voxel_size or cropping the input image')
            return 0
        
        new_phantom = interp(points)
        self.complete = np.stack((new_phantom * self.baseline[0]/self.baseline[1], new_phantom), axis = 0)        


	# set mask and read tissues by supplying a list of shapes and corresponding tissues
    def create_from_list(self, ):
        print('not yet implemented')
        
    
    def __getitem__(self, key):
        if isinstance(key, slice) or isinstance(key, int) or isinstance(key, tuple):
            return self.mask[key]
        
    def __setitem__(self, key, value):
        assert value in [t.label for t in self.tissues.values()], "value must be a valid tissue label"
        if isinstance(key, slice) or isinstance(key, int) or isinstance(key, tuple):
            self.mask[key] = value
        
        
    # add tissue
    def add_tissue(self, tissue):
        self.from_mask = True
        self.tissues[tissue.name] = tissue
        
        
    def remove_tissue(self, name):
        if name in self.tissues.keys():
            del self.tissues[name]
        else:
            print('tissue not found')
        
        
    def add_tissue_sphere(self, centroid, radius, tissue):
        region = geometry.create_sphere(centroid, radius, self.matrix_dims)
        if tissue.name not in self.tissues.keys():
            self.tissues[tissue.name] = tissue
        self.mask = np.where(region, tissue.label, self.mask)


    def get_tissues(self, ):
        return self.tissues


    def get_tissue_mask(self, ): # return a mask containing only tissues of a particular type
        print('not yet implemented')


    def get_mask(self, ):
        return self.mask


    def get_dimensions(self, ):
        return self.matrix_dims


    def get_voxel_dims(self, ):
        return self.voxel_dims
    
    
    def __randn_like(self, array):
        return self.rng.standard_normal(array.shape).astype(array.dtype)
    
    
    def __rand_like(self, array):
        return self.rng.uniform(array.shape).astype(array.dtype) * 2 - 1
    
    
    def generate_tissue(self, tissue, shape, voxel_size, rand_like = None, order = 1):
        if rand_like is None:
            rand_like = self.__randn_like
            
        if tissue.sigma == 0:
            return np.stack((np.ones(shape, dtype = np.float16) * tissue.c, np.ones(shape, dtype = np.float16) * tissue.rho), axis = 0)
            
        scaling_factor = tissue.scale / ((voxel_size[0] + voxel_size[1] + voxel_size[2]) / 3)
        scaling_factor = np.clip(scaling_factor, 1/np.mean(shape), np.mean(shape) * 100)
        size = (int(shape[0] // scaling_factor + 1),
				int(shape[1] // scaling_factor + 1),
			    int(shape[2] // scaling_factor + 1),)
        
		# Implement anisotropy later

        # if multiple scatterers in one voxel, modify std of noise: in 3 dimensions, std of iid gaussians scales with ^3/2
        if np.mean(size) > np.mean(shape):
            size = shape
            std_scale = scaling_factor ** 0.5
            scaling_factor = 1
        else:
            std_scale = 1
        
        tissue_matrix = np.ones(size, dtype = np.float32) * tissue.c
        tissue_matrix = tissue_matrix + rand_like(tissue_matrix) * tissue.sigma * std_scale

        tissue_matrix = scipy.ndimage.zoom(tissue_matrix, scaling_factor, order=order)
        tissue_matrix = tissue_matrix[:shape[0],
									  :shape[1],
									  :shape[2],]
        return np.stack((tissue_matrix, tissue_matrix/tissue.c * tissue.rho), axis = 0)
    
    
    def make_complete(self, mask, voxel_size): # edit this and generate_tissue, include the matrix size and the voxel size in the argument - very important
        complete = np.zeros(mask.shape, dtype = np.float16)
        for key in self.tissues.keys():
            complete = np.where(mask == self.tissues[key].label, self.generate_tissue(self.tissues[key], mask.shape, voxel_size), complete)
        return complete


    def get_complete(self):
        if self.complete is None:
            self.complete = self.make_complete(self.mask, self.voxel_dims)
        return self.complete
        
    
    # voxel_size and matrix_size refer to the size of a voxel (m,m,m) in the computational grid and the matrix size of the computational grid
    def crop_rotate_crop(self, bounds, transform, voxel_size, matrix_size):
        # keep a running log of discretization bias
        bias = np.array([0,0,0], dtype=np.float32)
        
        # need to swap order of rotation and translation according to homogeneous coordinates convention
        rotation = -transform.rotation.as_euler('ZYX')
        translation = -np.matmul(transform.get(inverse=True)[:3,:3], transform.translation)
        swapped_transform = geometry.Transform(rotation=rotation, translation=translation)
        transform = swapped_transform
        transformed_bounds = transform.apply_to_points(bounds, inverse=True) # need to edit this - homogeneous matrix definitions should be double checked
        
        # compute bounding box in global coords that contains the bounds
        first_crop_bounds_coords = np.array([(np.min(transformed_bounds[:,0]), np.max(transformed_bounds[:,0])),
                                            (np.min(transformed_bounds[:,1]), np.max(transformed_bounds[:,1])),
                                            (np.min(transformed_bounds[:,2]), np.max(transformed_bounds[:,2]))])
        
        # convert bounding coords to matrix indices so as to crop
        first_crop_bounds_indices = (first_crop_bounds_coords / np.broadcast_to(self.voxel_dims, (2,3)).T + np.broadcast_to(self.matrix_dims, (2,3)).T/2)
        first_crop_bounds_indices[:,0] = np.floor(first_crop_bounds_indices[:,0])
        first_crop_bounds_indices[:,1] = np.ceil(first_crop_bounds_indices[:,1])
        first_crop_bounds_indices = first_crop_bounds_indices.astype(np.int32)
        # buffer = np.amax(matrix_size) // 100
        # print(f'buffer {buffer}')
        # first_crop_bounds_indices = np.stack((first_crop_bounds_indices[:,0] - 5, first_crop_bounds_indices[:,1] + 5)).T # helpful to extend the initial cropped region slightly to avoid discretization truncation error
        
        
        # Given some bounds of the computational region, crop a cube centered at the same centroid, ithe long (diagonal) hypotenuse of the 
        
        bias += np.mean(first_crop_bounds_indices.astype(np.float32), axis=1) * self.voxel_dims - np.mean(first_crop_bounds_coords, axis=1)    
            
        # If self from_mask or self_from image, get either the mask or the matrix
        if self.from_mask:
            medium = self.mask
        else:
            medium = self.get_complete() # retrieve the matrix to transform - This should really be mask not complete in most cases
        
        # pad if indices extend out of the computational region
        pad_x = max(-first_crop_bounds_indices[0,0], first_crop_bounds_indices[0,1] - self.matrix_dims[0], 0)
        pad_y = max(-first_crop_bounds_indices[1,0], first_crop_bounds_indices[1,1] - self.matrix_dims[1], 0)
        pad_z = max(-first_crop_bounds_indices[2,0], first_crop_bounds_indices[2,1] - self.matrix_dims[2], 0)
        
        if self.from_mask:
            if pad_x:
                medium = np.pad(medium, ((pad_x,pad_x),(0,0),(0,0)), 'constant', constant_values=0)
            if pad_y:
                medium = np.pad(medium, ((0,0),(pad_y,pad_y),(0,0)), 'constant', constant_values=0)
            if pad_z:
                medium = np.pad(medium, ((0,0),(0,0),(pad_z,pad_z)), 'constant', constant_values=0)
        else:
            if pad_x:
                medium = np.stack(
                    (np.pad(medium[0], ((pad_x,pad_x),(0,0),(0,0)), 'constant', constant_values=(self.baseline[0],)),
                        np.pad(medium[1], ((pad_x,pad_x),(0,0),(0,0)), 'constant', constant_values=(self.baseline[1],))),
                    axis=0)
            if pad_y:
                medium = np.stack(
                    (np.pad(medium[0], ((0,0),(pad_y,pad_y),(0,0)), 'constant', constant_values=(self.baseline[0],)),
                        np.pad(medium[1], ((0,0),(pad_y,pad_y),(0,0)), 'constant', constant_values=(self.baseline[1],))),
                    axis=0)
            if pad_z:
                medium = np.stack(
                    (np.pad(medium[0], ((0,0),(0,0),(pad_z,pad_z)), 'constant', constant_values=(self.baseline[0],)),
                        np.pad(medium[1], ((0,0),(0,0),(pad_z,pad_z)), 'constant', constant_values=(self.baseline[1],))),
                    axis=0)

        first_crop_bounds_indices = first_crop_bounds_indices + np.stack((np.array((pad_x, pad_y, pad_z)),np.array((pad_x, pad_y, pad_z)))).T

        # compute the grid size:
        grid_size = matrix_size * voxel_size / self.voxel_dims
        
        if self.from_mask:
            cropped_matrix = medium[first_crop_bounds_indices[0,0]:first_crop_bounds_indices[0,1],
                                    first_crop_bounds_indices[1,0]:first_crop_bounds_indices[1,1],
                                    first_crop_bounds_indices[2,0]:first_crop_bounds_indices[2,1]]
        else:
            cropped_matrix = medium[:,  first_crop_bounds_indices[0,0]:first_crop_bounds_indices[0,1],
                                        first_crop_bounds_indices[1,0]:first_crop_bounds_indices[1,1],
                                        first_crop_bounds_indices[2,0]:first_crop_bounds_indices[2,1]]
                                
        # Perform rotation of the cropped region
        if self.from_mask:
            rotated_matrix = transform.rotate_array(cropped_matrix, padwith=0)
        else:
            rotated_matrix = transform.rotate_array(cropped_matrix, padwith=self.baseline)
        bias = np.matmul(transform.get(inverse=False)[:3,:3], bias).squeeze()
        
        # Perform a crop to get the rough grid matrix in global coordinates
        rough_crop = self.crop_matrix(rotated_matrix, grid_size)
        bias = bias + self.compute_bias(np.array(rotated_matrix.shape[-3:]), matrix_size) * self.voxel_dims
        
        # interpolate up to the correct simulation voxel_size
        sampled_matrix = self.interpolate_up(rough_crop, self.voxel_dims, voxel_size)
        bias = bias / voxel_size / matrix_size
        
        # finally perform a final crop to the desired computational matrix_size while correcting for accumulated bias
        # final = self.crop_matrix(sampled_matrix, matrix_size, bias=bias)
        final = self.crop_matrix(sampled_matrix, matrix_size)
                        
        # If self from_mask, then sample complete, else, return final
        if self.from_mask:
            final = self.make_complete(mask=final, voxel_size=voxel_size)
        
        return final
    
        
    def crop_matrix(self, matrix, matrix_size, bias=np.zeros(3)):
        centroid = np.round(np.array(matrix.shape[-3:]) / 2 + bias)
        start = (centroid - np.array(matrix_size) / 2)
        end = (start + np.array(matrix_size))
        start = np.floor(start).astype(np.int32)
        end = np.ceil(end).astype(np.int32)
        
        cropped_matrix = matrix[..., start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        return cropped_matrix
    
    
    def compute_bias(self, input_size, target_size):
        bias = []
        for i in range(len(input_size)):
            bias.append(((int(input_size[i]) & 0x1) != (int(target_size[i]) & 0x1)) / 2)
        return np.array(bias)
    
    def interpolate_up(self, matrix, input_voxel_size, target_voxel_size):
        x = np.linspace(0, matrix.shape[-3], matrix.shape[-3])
        y = np.linspace(0, matrix.shape[-2], matrix.shape[-2])
        z = np.linspace(0, matrix.shape[-1], matrix.shape[-1])
        
        transformed_x = x * input_voxel_size[0] / target_voxel_size[0]
        transformed_y = y * input_voxel_size[1] / target_voxel_size[1]
        transformed_z = z * input_voxel_size[2] / target_voxel_size[2]
        
        points = np.stack(np.meshgrid(np.arange(0, transformed_x[-1]), np.arange(0, transformed_y[-1]), np.arange(0, transformed_z[-1]), indexing='ij'), axis=-1)
        if points.shape[0] * points.shape[1] * points.shape[2] > 5e8:
            print('desired phantom array size is very large (>500,000,000 voxels), consider reducing the simulation grid size or increasing the simulation voxel size')
        if points.shape[0] * points.shape[1] * points.shape[2] > 2e9:
            assert False, 'desired phantom array size is too large (>2e9 voxels), consider reducing the simulation grid size or increasing the simulation voxel size'
        
        if len(matrix.shape) > 3:
            sampled_matrix = []
            for i in range(matrix.shape[0]):
                interp = RegularGridInterpolator((transformed_x, transformed_y, transformed_z), matrix[i], method='nearest')
                sampled_matrix.append(interp(points))
            sampled_matrix = np.stack(sampled_matrix, axis=0)
        else:
            interp = RegularGridInterpolator((transformed_x, transformed_y, transformed_z), matrix, method='nearest')
            sampled_matrix = interp(points)
        return sampled_matrix
    
    
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


    def render(self, ):
        print('not yet implemented')