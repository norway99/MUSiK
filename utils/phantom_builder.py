import sys
import os
import numpy as np
import scipy
import tqdm

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import glob
import pydicom
import functools
import mcubes

from sklearn.cluster import KMeans
from skimage.morphology import rectangle,disk, erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage import segmentation
import open3d as o3d
from open3d import io, visualization



def read_dicom(dicom_folder_path, HU = True, crop = None, axis = None):
    HU_image = []
    slice_position = []
    dicom_path = os.path.join(dicom_folder_path, '*.dcm')
    for dicom in tqdm.tqdm(sorted(glob.glob(dicom_path))):
        dcm = pydicom.dcmread(dicom)
        b = 0
        m = 1
        if hasattr(dcm, "RescaleIntercept"):
            b = dcm.RescaleIntercept
        if hasattr(dcm, "RescaleSlope"):
            m = dcm.RescaleSlope
        HU_image.append(m * dcm.pixel_array + b)
        slice_position.append(dcm.SliceLocation)
        
    HU_image = np.stack(HU_image, axis=-1)
    voxel_size = np.array([
        float(dcm.PixelSpacing[0]),
        float(dcm.PixelSpacing[1]),
        np.abs((float(slice_position[0]) - float(slice_position[-1])) / (len(slice_position) - 1))
    ])/1000

    HU_image = np.where(HU_image > -1000, HU_image, -1000)

    if crop is not None:
        if type(crop) is int or len(crop) == 1:
            if axis is None:
                raise Exception("Please specify a cropping axis.")
            elif axis == 0:
                HU_image = HU_image[:crop, :, :]
            elif axis == 1:
                HU_image = HU_image[:, :crop, :]
            elif axis == 2:
                HU_image = HU_image[:, :, :crop]
            else:
                raise Exception("Axis out of bounds.")
        elif len(crop == 2):
            HU_image = HU_image[:crop[0], :crop[1], :]
        else:
            HU_image = HU_image[:crop[0], :crop[1], crop[2]]

    if HU: # convert to grayscale
        min = np.min(HU_image)
        max = np.max(HU_image)
        image =  255 * (HU_image - min)/ (max - min)
    else:
        image = HU_image

    return image, voxel_size

def morph_open(image, footprint_size, footprint_shape = 'disk'):
    if footprint_shape == 'rectangle':
        if isinstance(footprint_size, int) or isinstsance(footprint_size, float):
            footprint = rectangle(footprint_size, footprint_size)
        else:
            footprint = rectangle(footprint_size[0], footprint_size[1])
    else:    
        footprint = disk(footprint_size)
    opened = image
    for i in range(image.shape[2]):
        opened[:, :, i] = opening(image[:, :, i], footprint)
    return opened

def flood_fill(image, slice_end):
    new_image = image
    for slice in range(slice_end): 
            for j in range(new_image.shape[1]):
                i_start = 0
                i_end = 0
                for i in range(new_image.shape[0]):
                    if new_image[i, j, slice] > 0:
                        i_start = i
                        break
                for i in range(1, new_image.shape[0]+1):
                    if new_image[-i, j, slice] > 0:
                        i_end = -i
                        break
                new_image[i_start:i_end, j, slice] = 1
    return new_image

def make_fg_mask(image, slice_end, use_kmeans = False, fg_threshold = None):
    fg_mask = np.zeros(image.shape)
    for slice in range(slice_end):
        if use_kmeans:
            kmeans = KMeans(n_clusters=2).fit(image[:, :, slice])
            fg_threshold = np.mean(kmeans.cluster_centers_, axis=0)
        else:
            if fg_threshold is None:
                raise Exception("Please supply a threshold pixel value.")
        fg_mask[:, :, slice] = np.where(image[:, :, slice] > fg_threshold, 1, 0)

    fg_mask = flood_fill(fg_mask, slice_end)
    return fg_mask

def make_surface_mesh(fg_mask, voxel_size, save_path):

    smoothed_mask = mcubes.smooth(fg_mask)
    vertices, triangles = mcubes.marching_cubes(smoothed_mask, 0)
    vertices = vertices * np.array([voxel_size[0], 1., 1.])
    vertices = vertices * np.array([1., voxel_size[1], 1.])
    vertices = vertices * np.array([1., 1., voxel_size[2]]) 
    mcubes.export_obj(vertices, triangles, save_path) # this works well 
    return vertices, triangles

def voxelize(voxel_size, mesh=None, mesh_file=None, min_bounds=None, max_bounds=None, grid_shape=None):
    if mesh is None and mesh_file is None:
        print('need to supply either mesh or mesh_file')
        return 0
    elif mesh is not None:
        legacy_mesh = mesh
    else:
        legacy_mesh = io.read_triangle_mesh(mesh_file)
    # if make_convex is True:
    #     ans = legacy_mesh.compute_convex_hull()
    #     legacy_mesh = ans[0]
    if min_bounds is None or max_bounds is None or grid_shape is None:
        voxels = o3d.geometry.VoxelGrid.create_from_triangle_mesh(legacy_mesh, voxel_size).get_voxels()
        voxel_indices = np.stack(list(vx.grid_index for vx in voxels))
        voxel_mask = np.zeros(np.max(voxel_indices, axis=0) + 1)
        for index in voxel_indices:
            voxel_mask[index[0], index[1], index[2]] = 1
        occupancy = voxel_mask
    else:
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(legacy_mesh)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh)
        xs = np.linspace(min_bounds[0], max_bounds[0], grid_shape[0]+1) + voxel_size/2
        xs = xs[:-1] 
        ys = np.linspace(min_bounds[1], max_bounds[1], grid_shape[1]+1) + voxel_size/2
        ys = ys[:-1]    
        zs = np.linspace(min_bounds[2], max_bounds[2], grid_shape[2]+1) + voxel_size/2
        zs = zs[:-1]
        query_points = np.stack(np.meshgrid(xs, ys, zs, indexing='ij'), axis=-1).astype(np.float32)
        occupancy = scene.compute_occupancy(query_points).numpy()
    return occupancy
        

    
