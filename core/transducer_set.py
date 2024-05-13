import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import sys
sys.path.append('../utils')

import geometry
import utils
from transducer import Transducer, Focused, Planewave


class TransducerSet:

    def __init__(self,
                 transducers = [],
                 poses       = [],
                 seed        = None,
                 ):
        self.transducers = transducers # list of transducers
        self.n_transducers = len(transducers)
        if len(poses) == 0:
            self.poses = [None for i in range(self.n_transducers)]
        else:
            self.poses = poses
        self.seed = seed
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            
    
    def __getitem__(self, index):
        return self.transducers[index], self.poses[index]
    
            
    @classmethod
    def load(cls, transducer_set_file, c0 = 1540):
        transducer_set_dict = utils.json_to_dict(transducer_set_file)
        transducer_list = []
        pose_list = []
        for my_dict in transducer_set_dict['transducers']:
            if my_dict[0]['type'] == 'focused':
                transducer = Focused.load(my_dict[0])
            elif my_dict[0]['type'] == 'planewave':
                transducer = Planewave.load(my_dict[0])
            else:
                transducer = Transducer.load(my_dict[0])
            transducer.make_sensor_coords(c0)
            transducer_list.append(transducer)
            pose_list.append(geometry.Transform.load(my_dict[1]))
        transducer_set = cls(transducer_list, pose_list)
        transducer_set_dict.pop('transducers')
        for key, value in transducer_set_dict.items():
            setattr(transducer_set, key, value) 
        transducer_set.rng = np.random.default_rng(transducer_set.seed)
        return transducer_set
            
            
    def save(self, save_file):
        transducer_set_dict = self.__dict__.copy()
        transducer_set_dict.pop('transducers')
        transducer_set_dict.pop('poses')
        transducer_set_dict.pop('rng')
        transducer_set_dict['transducers'] = []
        for t, p in zip(self.transducers, self.poses): # consider getting the dictionary obj for transducer and pose before zipping and saving
            my_dict = [t.save(), p.save()]
            transducer_set_dict['transducers'].append(my_dict)
        utils.dict_to_json(transducer_set_dict, save_file)
        
        
    def generate_extrinsics(self, shape=None, extrinsics_kwargs=None): # put tissue_mask = None, target = None, attenuation_threshold = None into extrinsics_kwargs

        if shape == "constrained":
            
            if tissue_mask is None or target is None or attenuation_threshold is None:
                raise Exception("In order to use option 'constrained,' \
                please supply a tissue mask, target point triple, and scalar attenuation threshold")
            self.optimize_extrinsics(**extrinsics_kwargs)
            
        elif shape == "spherical":
            for i in range(self.n_transducers):
                orientation, position = geometry.generate_pose_spherical(**extrinsics_kwargs, rng=self.rng)
                transform = geometry.Transform(orientation, position, intrinsic=True)
                self.assign_pose(i, transform)
        elif shape == "flat":
            pass
        elif shape == "cylindrical":
            for i in range(self.n_transducers):
                orientation, position = geometry.generate_pose_cylindrical(**extrinsics_kwargs, rng=self.rng)
                transform = geometry.Transform(orientation, position, intrinsic=True)
                self.assign_pose(i, transform)
        else: # random 
            for i in range(self.n_transducers):
                orientation, position = geometry.generate_random(**extrinsics_kwargs, rng=self.rng)
                transform = geometry.Transform(orientation, position, intrinsic=True)
                self.assign_pose(i, transform)
                        
    
    def optimize_extrinsics(self, tissue_mask, target, attenuation_threshold):
        pass
        
        
    def assign_pose(self, index, transform):
        assert index <= len(self), "Index out of range. No transducer exists at index {}.".format(index)
        self.poses[index] = transform
    
    
    def remove_pose(self, index):
        print('removing poses without removing transducers is not supported, either overwrite the pose (assign_pose) or remove the transducer (remove_transducer)')
        return 0   
                 
                    
    def find_transducer(self, label):
        ctr = 0
        for transducer in self.transducers:
            if transducer.label == label:
                break
            ctr += 1
        if ctr == len(self.transducer_list):
            return None
        else:
            return ctr
        
        
    def add_transducer(self, trans = None, load_file = None):
        if trans is None and load_file is None:
            raise Exception("Please supply either a transducer object or a .json file from which to load a transducer object")
        elif trans is None:
            trans = Transducer.load(load_file)
        self.transducers.append(trans)
        self.n_transducers += 1


    def remove_transducer(self, label = None, index = None): # remove the transducer at position i

        if label is None and index is None:
            raise Exception("Please supply a valid label or index to remove a transducer")
        elif label is not None:
            index = self.find_transducer(label)
        del self.transducers[index]
        self.n_transducers -= 1


    def get_transducers(self):
        return self.transducers


    def get_poses(self):
        return self.poses

    # def build_sensor_mask(self, transmit, transmit_transform, aperture, kgrid) -> np.ndarray:

    #     sensor_mask = transmit.not_transducer.indexed_mask

    #     all_sensor_coords = np.empty(1, 3)
        
    #     for key in aperture:
    #         t = self.transducers[key]
    #         p = self.poses[key]
    #         sensor_coords = t.get_sensor_coords()
    #         transformed_sensor_coords = transmit_transform.apply_to_points(p.apply_to_points(sensor_coords, inverse=True))
    #         all_sensor_coords = np.vstack((all_sensor_coords, np.ndarray.astype(np.round(transformed_sensor_coords), int)))
    #         for coord in transformed_sensor_coords:
    #             sensor_mask[coord[0], coord[1], coord[2]] = 1

    #     all_sensor_coords = all_sensor_coords[1:, :]
                
    #     return sensor_mask, all_sensor_coords
        
        
    def plot_transducer_coords(self, ax=None, save=False, save_path=None, scale=0.1):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.set_xlim(-scale, scale)
            ax.set_ylim(-scale, scale)
            ax.set_zlim(-scale, scale)
            ax.set_aspect('equal')
            ax.grid(False)
        cmap = plt.get_cmap('tab20b')
        for i in range(len(self.transducers)):
            self.transducers[i].plot_sensor_coords(ax=ax, transform=self.poses[i], color=cmap(i/len(self.transducers)))
        if ax is None:
            if save:
                plt.savefig(save_path)
            else:
                plt.show()

    def plot_transducers(self):
        coords = []
        for i in range(len(self.transducers)):
            coords.append(self.poses[i].apply_to_points(self.transducers[i].sensor_coords))
    
        fig = go.Figure(data=[go.Scatter3d(x=np.concatenate(coords, axis=0)[:,0],
                                           y=np.concatenate(coords, axis=0)[:,1],
                                           z=np.concatenate(coords, axis=0)[:,2],
                                           mode='markers',
                                           marker=dict(size=1, opacity=0.8,),
                )])
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        fig.update_layout(scene_aspectmode='data')
        fig.show()
                
    def plot_transducer_fovs(self, ax=None, save=False, save_path=None, scale=0.1):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.set_xlim(-scale, scale)
            ax.set_ylim(-scale, scale)
            ax.set_zlim(-scale, scale)
            ax.set_aspect('equal')
            ax.grid(False)
        cmap = plt.get_cmap('tab20b')
        for i in range(len(self.transducers)):
            self.transducers[i].plot_fov(ax=ax, transform=self.poses[i], length=0.02, color=cmap(i/len(self.transducers)))
        if ax is None:
            if save:
                plt.savefig(save_path)
            else:
                plt.show()
        

    def __len__(self,):
        return self.n_transducers

    def get_lowest_frequency(self):
        min_f = self.transducers[0].max_frequency
        for transducer in self.transducers:
            f = transducer.max_frequency
            if f < min_f:
                min_f = f
        return min_f
