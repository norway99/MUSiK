import numpy as np
import scipy
import scipy.ndimage
import os

import sys
sys.path.append('../utils')
import utils
import geometry
from tissue import Tissue
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
                 voxel_dims = (0.1e-3,0.1e-3,0.1e-3),
                 matrix_dims = (256,256,256),
                 baseline = (1500, 1000),
                 seed = 5678,
                 ):
        
        # initialize from source if exists
        if source_path is not None:
            self.load(source_path)
            return 1
        
        # otherwise initialize empty water phantom
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.voxel_dims = voxel_dims
        self.mask = np.zeros(matrix_dims, dtype = np.float16)
        self.tissues = {'water':Tissue(name = 'water', label = 0, c=1500, rho=1000, sigma=0, scale=0.1)}
        self.matrix_dims = matrix_dims
        self.baseline = baseline
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
            
        # if not os.path.exists(image_path):
        #     print('image file not found')
        #     return 0
        # if os.path.splitext(image_path)[1] == '.mhd':
        #     import SimpleITK
        #     data = SimpleITK.ReadImage(image_path)
        # elif os.path.splitext(image_path)[1] == '.nii' or os.path.splitext(image_path)[1] == '.nii.gz':
        #     import nibabel
        #     data = nibabel.load(image_path).get_fdata()
        # else:
        #     print('wants image file of either .mhd, .nii, or .nii.gz format')
        #     return 0
        
        if type(image) == np.ndarray:
            data = image
        
        # data = transfer_fn(data)

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
        self.complete = np.stack((new_phantom * self.baseline[0], new_phantom * self.baseline[1]), axis = 0)        


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
    
    
    def generate_tissue(self, tissue, rand_like = None, order = 1):
        if rand_like is None:
            rand_like = self.__randn_like
            
        if tissue.sigma == 0:
            return np.stack((np.ones(self.matrix_dims, dtype = np.float16) * tissue.c, np.ones(self.matrix_dims, dtype = np.float16) * tissue.rho), axis = 0)
            
        scaling_factor = tissue.scale / ((self.voxel_dims[0] + self.voxel_dims[1] + self.voxel_dims[2]) / 3)
        scaling_factor = np.clip(scaling_factor, 1/np.mean(self.matrix_dims), np.mean(self.matrix_dims) * 100)
        size = (int(self.mask.shape[0] // scaling_factor + 1),
				int(self.mask.shape[1] // scaling_factor + 1),
			    int(self.mask.shape[2] // scaling_factor + 1),)
        
		# Implement anisotropy later

        # if multiple scatterers in one voxel, modify std of noise: in 3 dimensions, std of iid gaussians scales with ^3/2
        if np.mean(size) > np.mean(self.matrix_dims):
            size = self.matrix_dims
            std_scale = scaling_factor ** 0.5
            scaling_factor = 1
        else:
            std_scale = 1
        
        tissue_matrix = np.ones(size, dtype = np.float32) * tissue.c
        tissue_matrix = tissue_matrix + rand_like(tissue_matrix) * tissue.sigma * std_scale

        tissue_matrix = scipy.ndimage.zoom(tissue_matrix, scaling_factor, order=order)
        tissue_matrix = tissue_matrix[:self.matrix_dims[0],
									  :self.matrix_dims[1],
									  :self.matrix_dims[2],]
        return np.stack((tissue_matrix, tissue_matrix/tissue.c * tissue.rho), axis = 0)
    
    
    def make_complete(self):
        complete = np.zeros(self.matrix_dims, dtype = np.float16)
        for key in self.tissues.keys():
            complete = np.where(self.mask == self.tissues[key].label, self.generate_tissue(self.tissues[key]), complete)
        self.complete = complete


    def get_complete(self):
        if self.complete is None:
            self.make_complete()
            return self.complete
        else:
            return self.complete


    def render(self, ):
        print('not yet implemented')