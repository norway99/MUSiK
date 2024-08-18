import json
import numpy as np
import os
import mrcfile
from scipy.ndimage import binary_fill_holes
from scipy.spatial import ConvexHull, Delaunay


# for serializing numpy arrays to json
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

# save dictionary to filepath as json
def dict_to_json(dictionary, filepath):
    json_object = json.dumps(dictionary, indent=4, cls=NpEncoder)
    with open(filepath, "w") as outfile:
        outfile.write(json_object)

# read json into dictionary
def json_to_dict(filepath):
    with open(filepath, "r") as infile:
        return json.load(infile)
    
# save to numpy file
def save_array(array, filepath, compression=True):
    filepath = os.path.splitext(filepath)[0]
    if compression:
        filepath = filepath + ".npz"
        np.savez_compressed(filepath, array=array)
    else:
        filepath = filepath + ".npy"
        np.save(filepath, array)
    
# load numpy file
def load_array(filepath):
    ext = os.path.splitext(filepath)[1]
    if ext == ".npz":
        return np.load(filepath)["array"]
    elif ext == ".npy":
        return np.load(filepath)
    elif len(ext) == 0:
        if os.path.isfile(filepath+".npz"):
            return np.load(filepath+".npz")["array"]
        elif os.path.isfile(filepath+".npy"):
            return np.load(filepath+".npy")
    else:
        print("Error: file extension not recognized, must be .npy or .npz")
        return None

def save_mrc(array, filepath):
    array = np.where(np.isnan(array), 0, array)
    with mrcfile.new(filepath, overwrite=True) as mrc:
        mrc.set_data(array.astype(np.float32))

def generate_distance_matrix(size, center=None) -> np.array:
    if center == None:
        center = [size[dim]//2 for dim in range(len(size))]
    else:
        assert len(center) == len(size)
        
    dist_arrays = []
    for dim in range(len(size)):
        dist_arrays.append(np.arange(size[dim]) - center[dim])

    coord_arrays = np.meshgrid(*dist_arrays, indexing='ij')
    dist = np.sqrt(sum([coord**2 for coord in coord_arrays]))
    return dist


def fill_3d_holes(binary_mask):
    binary_mask = np.where(binary_mask > 0, 1, 0)
    filled_region = binary_fill_holes(binary_mask)
    binary_mask[filled_region > 0] = 1
    return binary_mask

def compute_convex_hull_mask(points, meshgrid_obj):    
    hull = ConvexHull(points)
    deln = Delaunay(points[hull.vertices]) 
    out_idx = np.nonzero(deln.find_simplex(meshgrid_obj) + 1)
    out_img = np.empty(meshgrid_obj.shape[:-1])
    out_img[:] = np.nan
    out_img[out_idx] = 1
    return out_img