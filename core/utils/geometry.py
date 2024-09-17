import numpy as np
import scipy
from scipy import spatial
from scipy import ndimage
import matplotlib.pyplot as plt
import copy

import utils

class Transform:
    def __init__(self, 
                rotation = (0,0,0),
                translation = (0,0,0),
                from_matrix = False,
                about_axis = False,
                intrinsic = True,       # define intrinsic or extrinsic while initializing from euler angles
                ordering = None,        # define the order of euler angles
                ):
        
        # initialize from rotation and translation  
        if from_matrix:
            if isinstance(rotation, np.ndarray):
                self.rotation = scipy.spatial.transform.Rotation.from_matrix(rotation)
            else:
                raise Exception("Please supply the rotation in matrix format")
        elif about_axis:
            if rotation == (0,0,0):
                self.rotation = scipy.spatial.transform.Rotation.from_matrix(np.eye(3))
            else:
                self.rotation = scipy.spatial.transform.Rotation.from_rotvec(rotation)
        elif intrinsic:
            if ordering is not None: 
                self.rotation = scipy.spatial.transform.Rotation.from_euler(ordering, rotation, degrees=False)
            else:
                self.rotation = scipy.spatial.transform.Rotation.from_euler("ZYX", rotation, degrees=False)  # intrinsic
        else:
            if ordering is not None:
                self.rotation = scipy.spatial.transform.Rotation.from_euler(ordering, rotation, degrees=False)
            else:
                self.rotation = scipy.spatial.transform.Rotation.from_euler("zyx", rotation, degrees=False)  # extrinsic
        self.translation = np.array(translation)

    @classmethod
    def make_from_heading_vector(cls, heading_vector, trans):
 
        e = np.array([1, 0, 0])
        rej = (heading_vector - np.dot(e, heading_vector) * e)/np.linalg.norm(heading_vector - np.dot(e, heading_vector) * e)
        cross = np.cross(heading_vector, e)

        f_inv = np.column_stack((e, rej, cross))

        f = np.linalg.inv(f_inv)

        g = np.array([[np.dot(e, heading_vector), -1*np.linalg.norm(np.cross(e, heading_vector)), 0], 
                      [np.linalg.norm(np.cross(e, heading_vector)), np.dot(e, heading_vector), 0], 
                      [0, 0, 1]])

        rot_matrix = np.matmul(f_inv, np.matmul(g, f))

        return cls(rotation=rot_matrix, translation=trans, from_matrix=True)
    
    
    # return homogeneous transformation matrix
    def __get_matrix(self, inverse=False, scale=1,):
        # figure out how to invert the transformation matrix here: https://mathematica.stackexchange.com/questions/106257/how-do-i-get-the-inverse-of-a-homogeneous-transformation-matrix
        if inverse:
            transform = np.zeros((4,4), dtype=np.float32)
            transform[:-1,:-1] = self.rotation.as_matrix().T
            transform[:-1,-1] = -self.rotation.apply(self.translation, inverse=True) / scale
            transform[-1,:] = np.array([0,0,0,1])
            return transform
        else:
            transform = np.zeros((4,4), dtype=np.float32)
            transform[:-1,:-1] = self.rotation.as_matrix()
            transform[:-1,-1] = self.translation / scale
            transform[-1,:] = np.array([0,0,0,1])
            return transform
        
    def get(self, inverse=False, scale=1):
        return self.__get_matrix(inverse=inverse, scale=scale)
        
    
    # pad to the minimum bounding box necessary to contain the phantom for a given transformation
    def __greedy_padding(self, array, scale, padwith):
                
        cent = np.array((array.shape[-3] / 2, array.shape[-2] / 2, array.shape[-1] / 2))
        vert = np.array([
            (0,0,0),
            (0,0,array.shape[-1]),
            (0,array.shape[-2],0),
            (array.shape[-3],0,0),
            (array.shape[-3],array.shape[-2],0),
            (0,array.shape[-2],array.shape[-1]),
            (array.shape[-3],0,array.shape[-1]),
            (array.shape[-3],array.shape[-2],array.shape[-1]), 
        ]) - cent - self.translation/scale
        
        inverse_vert = self.apply_to_points(vert, inverse=True, scale=scale) + cent
        vert = vert + cent
        
        min_x = np.floor(np.min(np.concatenate((inverse_vert[:,0],vert[:,0]), axis=0)))
        max_x = np.ceil( np.max(np.concatenate((inverse_vert[:,0],vert[:,0]), axis=0)))
        min_y = np.floor(np.min(np.concatenate((inverse_vert[:,1],vert[:,1]), axis=0)))
        max_y = np.ceil( np.max(np.concatenate((inverse_vert[:,1],vert[:,1]), axis=0)))
        min_z = np.floor(np.min(np.concatenate((inverse_vert[:,2],vert[:,2]), axis=0)))
        max_z = np.ceil( np.max(np.concatenate((inverse_vert[:,2],vert[:,2]), axis=0)))
                
        if min_x < 0: cent[0] = cent[0] - min_x
        if min_y < 0: cent[1] = cent[1] - min_y
        if min_z < 0: cent[2] = cent[1] - min_z
        
        x_pad = (int(max(0,-min_x)), int(max(0,max_x - array.shape[-3])))
        y_pad = (int(max(0,-min_y)), int(max(0,max_y - array.shape[-2])))
        z_pad = (int(max(0,-min_z)), int(max(0,max_z - array.shape[-1])))

        if len(array.shape) == 3:
            padded = np.pad(array, (x_pad, y_pad, z_pad), mode='constant', constant_values=(padwith,))
        elif len(array.shape) == 4:
            padded = []
            for i in range(array.shape[0]):
                padded.append(np.pad(array[i], (x_pad, y_pad, z_pad), mode='constant', constant_values=(padwith[i],)))
            padded = np.stack(padded, axis=0)
        else:
            print('padding currently only supported for transformations of 3 dimensional single or multichannel arrays')
            return 0
        
        return padded, cent
    
    
    # for applying to an image matrix in 3D or a multichannel image matrix in 3D + 1D
    def apply_to_array(self, array, scale=1, padwith=0, inverse=False, order=0, mode='nearest'):
        transform = self.__get_matrix(inverse=inverse)
        translation = self.translation / -scale
        
        # padding to avoid losing information
        if padwith is not None:
            array, center = self.__greedy_padding(array, scale, padwith)
                    
        if len(array.shape) > 3:
            translated = []
            for i in range(array.shape[0]):
                translated.append(scipy.ndimage.shift(array[i], translation, order=order, mode='constant', cval=padwith[i]))
            translated = np.stack(translated, axis=0)
        else:
            translated = scipy.ndimage.shift(array, translation, order=order, mode='constant', cval=padwith)
        
        offset = center - self.rotation.apply(center)
        
        if len(array.shape) > 3:
            tarray = []
            for i in range(array.shape[0]):
                tarray.append(scipy.ndimage.affine_transform(translated[i], transform[:-1,:-1], order=order, mode='constant', cval=padwith[i], offset=offset))
            tarray = np.stack(tarray, axis=0)
        else:
            tarray = scipy.ndimage.affine_transform(translated, transform[:-1,:-1], order=order, mode='constant', cval=padwith, offset=offset)
                    
        return tarray, center
    
    def padtocube(self, array):
        shape = array.shape[-3:]
        hypot = int(max(shape) * np.sqrt(3))
        left_pad1 = (hypot - shape[0]) // 2
        right_pad1 = hypot - shape[0] -  left_pad1
        left_pad2 = (hypot - shape[1]) // 2
        right_pad2 = hypot - shape[1] -  left_pad2
        left_pad3 = (hypot - shape[2]) // 2
        right_pad3 = hypot - shape[2] -  left_pad3
        if len(array.shape) > 3:
            if len(array.shape) > 4:
                print("why does the array have more than four dimensions?")
            parray = []
            for i in range(array.shape[0]):
                parray.append(np.pad(array[i], ((left_pad1, right_pad1), (left_pad2, right_pad2), (left_pad3, right_pad3)), mode='edge'))
            parray = np.stack(parray, axis=0)
        else:
            parray = np.pad(array, ((left_pad1, right_pad1), (left_pad2, right_pad2), (left_pad3, right_pad3)), mode='edge')
        return parray
    
    
    # for applying to an image matrix in 3D or a multichannel image matrix in 3D + 1D
    def rotate_array(self, array, padwith=0, inverse=False, order=0):
        padded_array = self.padtocube(array)
        transform = self.__get_matrix(inverse=inverse)
        

        center = np.array(padded_array.shape[-3:])/2
        # print((padded_array.shape[-3] - array.shape[-3]) / 2)
        # print(padded_array.shape)
        # print(array.shape)
        # center = np.array(((padded_array.shape[-3] - array.shape[-3]) / 2, padded_array.shape[-2] / 2, padded_array.shape[-1] / 2))
        
        
        offset = center - self.rotation.apply(center, inverse=True)
        
        if len(padded_array.shape) > 3:
            tarray = []
            for i in range(padded_array.shape[0]):
                tarray.append(scipy.ndimage.affine_transform(padded_array[i], transform[:-1,:-1].T, order=order, mode='constant', cval=padwith[i], offset=offset))
            tarray = np.stack(tarray, axis=0)
        else:
            tarray = scipy.ndimage.affine_transform(padded_array, transform[:-1,:-1].T, order=order, mode='constant', cval=padwith, offset=offset)
            
        return tarray
        
        
    # for applying to a single point in R^3
    def apply_to_point(self, v, inverse=False, scale=1):
        if inverse:
            return np.matmul(self.__get_matrix(inverse=True, scale=scale) , np.array([v[0], v[1], v[2] ,1]).T)[:3]
        return np.matmul(self.__get_matrix(scale=scale) , np.array([v[0], v[1], v[2] ,1]).T)[:3]
        
        
    # for applying to an array of points
    def apply_to_points(self, v, inverse=False, scale=1):
        if inverse:
            return np.einsum('ij,mj->mi', self.__get_matrix(inverse=True, scale=scale), np.array([v[:,0], v[:,1], v[:,2], np.ones_like(v[:,0])]).T)[:,:3]
        return np.einsum('ij,mj->mi', self.__get_matrix(scale=scale), np.array([v[:,0], v[:,1], v[:,2], np.ones_like(v[:,0])]).T)[:,:3]
    
    
    # define composition of transformations
    def __mul__(self, other):
        assert type(self) is type(other), "Can only compose transformations of the same type"
        return self.__class__((self.rotation * other.rotation).as_euler("ZYX", degrees=False), self.translation + other.translation)

    def plot_self(self, save=False, filepath='outputs/rotation.png'):
        ax = plt.figure().add_subplot(projection="3d", proj_type="ortho")
        self.__plot_rotated_axes(ax, self.rotation, name="r", offset=(0, 0, 0))
        ax.set(xlim=(-1.25, 1.25), ylim=(-1.25, 1.25), zlim=(-1.25, 1.25))
        ax.set(xticks=[-1, 0, 1], yticks=[-1, 0, 1], zticks=[-1, 0, 1])
        ax.set_aspect("equal", adjustable="box")
        ax.figure.set_size_inches(3, 3)
        if save:
            plt.savefig(filepath)
        else:
            plt.show()

    def __plot_rotated_axes(self, ax, r, name=None, offset=(0, 0, 0), scale=1):
        colors = ("#FF6666", "#005533", "#1199EE")  # Colorblind-safe RGB
        loc = np.array([offset, offset])
        for i, (axis, c) in enumerate(zip((ax.xaxis, ax.yaxis, ax.zaxis),
                                        colors)):
            axlabel = axis.axis_name
            axis.set_label_text(axlabel)
            axis.label.set_color(c)
            axis.line.set_color(c)
            axis.set_tick_params(colors=c)
            line = np.zeros((2, 3))
            line[1, i] = scale
            line_rot = r.apply(line)
            line_plot = line_rot + loc
            ax.plot(line_plot[:, 0], line_plot[:, 1], line_plot[:, 2], c)
            text_loc = line[1]*1.2
            text_loc_rot = r.apply(text_loc)
            text_plot = text_loc_rot + loc[0]
            ax.text(*text_plot, axlabel.upper(), color=c,
                    va="center", ha="center")
        ax.text(*offset, name, color="k", va="center", ha="center",
                bbox={"fc": "w", "alpha": 0.8, "boxstyle": "circle"})
        
    def save(self):
        return {"rotation": tuple(self.rotation.as_euler("ZYX", degrees=False)), "translation": tuple(self.translation)}


    @classmethod
    def load(cls, dictionary):
        # load will ALWAYS default to intrinsic definition of euler angles, if using the save() function, this will be consistent
        rot = dictionary["rotation"]
        trans = dictionary["translation"]
        return cls(rotation = rot, translation = trans)
    
    
    def __copy__(self,):
        return self.__class__(rotation = self.rotation.as_euler("ZYX", degrees=False), translation = self.translation)
        








def create_sphere(centroid, radius, voxel_dims, matrix_dims):
    
    # convert centroid from global coordinates to matrix indices
    centroid = (int(centroid[0] / voxel_dims[0] + matrix_dims[0] / 2), 
                int(centroid[1] / voxel_dims[1] + matrix_dims[1] / 2), 
                int(centroid[2] / voxel_dims[2] + matrix_dims[2] / 2))
        
    x, y, z = np.mgrid[0:matrix_dims[0]:1, 0:matrix_dims[1]:1, 0:matrix_dims[2]:1]
    r = np.sqrt((x - centroid[0])**2 + (y - centroid[1])**2 + (z - centroid[2])**2)
    r[r==0] = 1
    r[r > radius] = 0
    
    return r



def generate_pose_spherical(r_mean=5e-3, r_std=0, view_std=0, yaw_fraction=0.5, pitch_fraction=1, roll_fraction=0, rng=None):
    if rng is not None:
        rand = rng.random
        randn = rng.normal
    else:
        rand = np.random.rand
        randn = np.random.randn
    theta = 2 * np.pi * rand() * yaw_fraction    
    phi = (np.arccos(2 * rand() - 1) - np.pi / 2) * pitch_fraction + np.pi / 2
    r = randn() * r_std + r_mean
    theta_p = theta #- np.pi  # assign direction to be towards origin with some variance
    
    if pitch_fraction == 0:
        phi_p = -phi + np.pi / 2  # assign direction to be towards origin with some variance
    else:
        phi_p = - phi + np.pi / 2  # assign direction to be towards origin with some variance
    
    gamma_p = (0.5 - rand()) * 2 * np.pi * roll_fraction  # assign roll randomly
    
    orientation = np.array([theta_p, phi_p, gamma_p])
    transform = Transform(orientation, (0, 0, 0))
    
    position = transform.apply_to_point(np.array([-r, 0, 0]))
    
    if view_std != 0:
        if pitch_fraction == 0:
            orientation[0] = randn() * view_std + theta  # assign direction to be towards origin with some variance
        else:
            orientation[0] = randn() * view_std + theta
            orientation[1] = randn() * view_std - phi - np.pi/2
            orientation[2] = randn() * view_std + gamma_p
            orientation = orientation + randn(3) * view_std  # assign direction to be towards origin with some variance
    
    return orientation, position



def generate_pose_cylindrical(r_mean=5e-3, r_std=0, view_std=0, z_range=None, rng=None):
    if rng is not None:
        rand = rng.random
        randn = rng.normal
    else:
        rand = np.random.rand
        randn = np.random.randn
    if z_range is None:
        z_range = r_mean
    theta = 2 * np.pi * rand()
    z = (rand() - 0.5) * z_range * 2
    r = randn() * r_std + r_mean

    theta_p = np.mod((randn() * view_std - theta + 2 * np.pi), 2 * np.pi)  # assign direction to be towards origin with some variance
    phi_p = np.mod((randn() * view_std - theta + 2 * np.pi), 2 * np.pi)  # assign direction to be towards origin with some variance
    gamma_p = rand() * 2 * np.pi  # assign roll randomly

    theta_p = np.pi + theta  # assign direction to be towards origin with some variance
    phi_p = 0  # assign direction to be towards origin with some variance
    gamma_p = 0  # assign roll randomly
    
    # position = np.array([-r, 0, z])
    orientation = np.array([theta_p, phi_p, gamma_p])
    transform = Transform(orientation, (0, 0, 0))
    position = transform.apply_to_point(np.array([-r, 0, z]))

    return orientation, position

def generate_random(size=0.1, rng=None):
    if rng is not None:
        rand = rng.random
    else:
        rand = np.random.rand
    return 2 * np.pi * rand(3), (2 * rand(3) - 1) * size