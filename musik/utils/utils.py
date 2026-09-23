import json
import numpy as np
import os
import mrcfile
from scipy.ndimage import binary_fill_holes
from scipy.spatial import ConvexHull, Delaunay


class NpEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types.

    Converts NumPy integers, floats, and arrays to Python-native types
    for JSON serialization.

    Example:
        >>> data = {'array': np.array([1, 2, 3]), 'value': np.float64(3.14)}
        >>> json.dumps(data, cls=NpEncoder)
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def dict_to_json(dictionary, filepath):
    """Save a dictionary to a JSON file.

    Handles NumPy types automatically using NpEncoder.

    Args:
        dictionary: Dictionary to save.
        filepath: Path to output JSON file.
    """
    json_object = json.dumps(dictionary, indent=4, cls=NpEncoder)
    with open(filepath, "w") as outfile:
        outfile.write(json_object)


def json_to_dict(filepath):
    """Load a dictionary from a JSON file.

    Args:
        filepath: Path to JSON file.

    Returns:
        Dictionary loaded from file.
    """
    with open(filepath, "r") as infile:
        return json.load(infile)


def save_array(array, filepath, compression=True):
    """Save a NumPy array to disk.

    Args:
        array: NumPy array to save.
        filepath: Output path (extension will be added automatically).
        compression: If True, use compressed .npz format. Otherwise .npy.
    """
    filepath = os.fspath(filepath)
    filepath = os.path.splitext(filepath)[0]
    if compression:
        filepath = filepath + ".npz"
        np.savez_compressed(filepath, array=array)
    else:
        filepath = filepath + ".npy"
        np.save(filepath, array)


def load_array(filepath):
    """Load a NumPy array from disk.

    Automatically handles both .npy and .npz formats, and will search
    for either extension if none is provided.

    Args:
        filepath: Path to array file (.npy or .npz).

    Returns:
        NumPy array, or None if file not found.
    """
    filepath = os.fspath(filepath)
    ext = os.path.splitext(filepath)[1]
    if ext == ".npz":
        return np.load(filepath)["array"]
    elif ext == ".npy":
        return np.load(filepath)
    elif len(ext) == 0:
        if os.path.isfile(filepath + ".npz"):
            return np.load(filepath + ".npz")["array"]
        elif os.path.isfile(filepath + ".npy"):
            return np.load(filepath + ".npy")
    else:
        print("Error: file extension not recognized, must be .npy or .npz")
        return None


def save_mrc(array, filepath):
    """Save a 3D array to MRC format (cryo-EM standard).

    Replaces NaN values with zeros before saving.

    Args:
        array: 3D NumPy array to save.
        filepath: Output path with .mrc extension.
    """
    array = np.where(np.isnan(array), 0, array)
    with mrcfile.new(filepath, overwrite=True) as mrc:
        mrc.set_data(array.astype(np.float32))


def generate_distance_matrix(size, center=None) -> np.array:
    """Generate a matrix of Euclidean distances from a center point.

    Creates an N-dimensional array where each element contains the
    distance from that position to the center.

    Args:
        size: Tuple of array dimensions.
        center: Center point coordinates. If None, uses array center.

    Returns:
        NumPy array of distances with shape `size`.
    """
    if center == None:
        center = [size[dim] // 2 for dim in range(len(size))]
    else:
        assert len(center) == len(size)

    dist_arrays = []
    for dim in range(len(size)):
        dist_arrays.append(np.arange(size[dim]) - center[dim])

    coord_arrays = np.meshgrid(*dist_arrays, indexing="ij")
    dist = np.sqrt(sum([coord**2 for coord in coord_arrays]))
    return dist


def fill_3d_holes(binary_mask):
    """Fill holes in a 3D binary mask.

    Args:
        binary_mask: 3D binary array (values > 0 are True).

    Returns:
        Binary mask with internal holes filled.
    """
    binary_mask = np.where(binary_mask > 0, 1, 0)
    filled_region = binary_fill_holes(binary_mask)
    binary_mask[filled_region > 0] = 1
    return binary_mask


def compute_convex_hull_mask(points, meshgrid_obj):
    """Create a mask for points inside the convex hull.

    Args:
        points: (N, 3) array of points defining the hull vertices.
        meshgrid_obj: Meshgrid array of shape (..., 3) to test.

    Returns:
        Array with 1 inside hull, NaN outside.
    """
    hull = ConvexHull(points)
    deln = Delaunay(points[hull.vertices])
    out_idx = np.nonzero(deln.find_simplex(meshgrid_obj) + 1)
    out_img = np.empty(meshgrid_obj.shape[:-1])
    out_img[:] = np.nan
    out_img[out_idx] = 1
    return out_img
