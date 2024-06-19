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

    def _snap_to_surface(self, point, surface):   
        import open3d as o3d
             
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(surface)
        query_pt = o3d.core.Tensor([point], dtype=o3d.core.Dtype.Float32)
        closest_pt = scene.compute_closest_points(query_pt)
        closest_triangle = closest_pt['primitive_ids'][0].item()
        return closest_pt['points'][0].numpy(), closest_triangle
    
    def place_on_mesh_voxel(self, transducer_index, surface_mesh, voxel, voxel_size):
        import open3d as o3d
        
        min_coord = surface_mesh.get_min_bound()
        coord = np.multiply(voxel,voxel_size) + min_coord
        return self.place_on_mesh(transducer_index, surface_mesh, point = coord)
        
    def place_on_mesh(self, transducer_index, surface_mesh, vertex_id = None, triangle_id = None, point = None):
        import open3d as o3d
        
        if surface_mesh is None:
            raise Exception("Must provide a surface on which to place the transducer")
        if vertex_id is None and triangle_id is None and point is None:
            raise Exception("Must provide a heuristic for transducer placement")
        if not isinstance(surface_mesh, o3d.t.geometry.TriangleMesh):
            surface_mesh = o3d.t.geometry.TriangleMesh.from_legacy(surface_mesh)
        surface_mesh.compute_vertex_normals()
        surface_mesh.compute_triangle_normals()
        surface_mesh.normalize_normals()
        if vertex_id is not None:
            pt = surface_mesh.vertex.positions[vertex_id]
            normal= surface_mesh.vertex.normals[vertex_id]
        else:   
            if triangle_id is not None:
                normal = surface_mesh.triangle.normals[triangle_id]
                triangle_vertices = surface_mesh.triangle.indices[triangle_id].numpy()
                triangle_vertex_coords = np.vstack(surface_mesh.vertex.positions[triangle_vertices[0]].numpy(), 
                                                   surface_mesh.vertex.positions[triangle_vertices[1]].numpy(),
                                                   surface_mesh.vertex.positions[triangle_vertices[2]].numpy())
                pt = np.mean(triangle_vertex_coords, axis=0)
            else:                
                pt, id = self._snap_to_surface(point, surface_mesh)
                normal = surface_mesh.triangle.normals[id].numpy()
        min_coord = surface_mesh.get_min_bound().numpy()
        max_coord = surface_mesh.get_max_bound().numpy()
        mesh_origin = np.mean([min_coord, max_coord], axis=0)
        pt = pt - mesh_origin     
        return pt, normal

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
        self.transducers.pop(index)
        self.poses.pop(index)
        self.n_transducers -= 1


    def get_transducers(self):
        return self.transducers


    def get_poses(self):
        return self.poses


    def plot_transducer_coords(self, ax=None, save=False, save_path=None, scale=0.1, view=None, phantom_coords=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.set_xlim(-scale, scale)
            ax.set_ylim(-scale, scale)
            ax.set_zlim(-scale, scale)
            ax.set_aspect('equal')
            ax.grid(False)
        if view is not None:
            ax.view_init(view[0], view[1])
        cmap = plt.get_cmap('tab20b')
        for i in range(len(self.transducers)):
            self.transducers[i].plot_sensor_coords(ax=ax, transform=self.poses[i], color=cmap(i/len(self.transducers)))
        if phantom_coords is not None:
            ax.scatter(phantom_coords[:,0], phantom_coords[:,1], phantom_coords[:,2], s=0.1, color='b')
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
                
    def plot_transducer_fovs(self, ax=None, save=False, save_path=None, scale=0.1, view=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.set_xlim(-scale, scale)
            ax.set_ylim(-scale, scale)
            ax.set_zlim(-scale, scale)
            ax.set_aspect('equal')
            ax.grid(False)
        if view is not None:
            ax.view_init(view[0], view[1])
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
