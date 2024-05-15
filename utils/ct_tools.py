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
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image


def read_dicom(dicom_folder_path, HU = True, crop = None, axis = None):
    HU_image = []
    slice_position = []
    dicom_path = os.path.join(dicom_folder_path, '*.dcm')
    for dicom in tqdm.tqdm(sorted(glob.glob(dicom_path))):
        dcm = pydicom.dcmread(dicom)
        b = dcm.RescaleIntercept
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

    return image

def flood_fill(fg_mask, slice_end):
    for slice in range(slice_end):  
        for j in range(fg_mask.shape[1]):
            i_start = 0
            i_end = 0
            for i in range(fg_mask.shape[0]):
                if fg_mask[i, j, slice] == 1:
                    i_start = i
                    break
            for i in range(1, fg_mask.shape[0]+1):
                if fg_mask[-i, j, slice] == 1:
                    i_end = -i
                    break
            fg_mask[i_start:i_end, j, slice] = 1

def make_fg_mask(image, slice_end, use_kmeans = True, fg_threshold = None):
    fg_mask = np.zeros(image.shape)
    for slice in range(slice_end):
        if use_kmeans:
            kmeans = KMeans(n_clusters=2).fit(image[:, :, slice])
            fg_threshold = np.mean(kmeans.cluster_centers_, axis=0)
        else:
            if fg_threshold is None:
                raise Exception("Please supply a threshold pixel value.")
        fg_mask[:, :, slice] = np.where(image[:, :, slice] > fg_threshold, 1, 0)

    flood_fill(fg_mask, slice_end)
    return fg_mask

def make_surface_mesh(fg_mask, save_path):

    smoothed_mask = mcubes.smooth(fg_mask)
    vertices, triangles = mcubes.marching_cubes(smoothed_mask, 0)
    mcubes.export_obj(vertices, triangles, save_path) # this works well 
    return vertices, triangles


    
