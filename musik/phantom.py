import numpy as np
import scipy
import scipy.ndimage
import os
from os import listdir
from .utils import utils
from .utils import geometry
from .tissue import Tissue
from scipy.interpolate import RegularGridInterpolator
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Any, Union, List, Callable

# phantom bounds in global coords are NOT necessarily -1 to 1

_MIX_1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX_2 = np.uint64(0x94D049BB133111EB)
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)


def _mix64(x):
    """SplitMix64 finalizer: a fast, well-distributed hash of a uint64 array."""
    x = (x ^ (x >> np.uint64(30))) * _MIX_1
    x = (x ^ (x >> np.uint64(27))) * _MIX_2
    return x ^ (x >> np.uint64(31))


def _texture_key(seed, label):
    """64-bit key identifying one tissue's texture field under a phantom seed."""
    return (int(seed) * 0x9E3779B97F4A7C15 + int(label) * 0xD1B54A32D192ED03 + 1) % 2**64


def _hashed_normal(cells, key):
    """Standard normal values, float32, that depend only on integer lattice cells (n, 3) and `key`."""
    h = _mix64(np.full(len(cells), key, dtype=np.uint64))
    for axis in range(3):
        h = _mix64(h ^ cells[:, axis].astype(np.int64).view(np.uint64))
    # Box-Muller from two uniforms taken from independent hashes.
    u1 = ((h >> np.uint64(11)).astype(np.float64) + 0.5) * 2.0**-53
    u2 = (_mix64(h ^ _GOLDEN) >> np.uint64(11)).astype(np.float64) * 2.0**-53
    return (np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)).astype(np.float32)


def _world_noise(points, spacing, key, interpolate):
    """Unit-variance lattice noise at world `points` (n, 3), fixed in space.

    Each lattice cell of size `spacing` holds one hashed normal value. Without
    `interpolate`, a point takes its cell's value (white noise at the lattice
    resolution); with it, values are trilinear between lattice nodes, matching the
    linearly upsampled coarse grid of `Phantom.generate_tissue`.
    """
    scaled = points / spacing
    if not interpolate:
        return _hashed_normal(np.floor(scaled).astype(np.int64), key)
    # Hash each lattice node once over the points' bounding box, then interpolate.
    lo = np.floor(scaled.min(axis=0)).astype(np.int64)
    hi = np.floor(scaled.max(axis=0)).astype(np.int64) + 2
    nodes = np.stack(
        np.meshgrid(*[np.arange(a, b) for a, b in zip(lo, hi)], indexing="ij"), axis=-1
    )
    node_noise = _hashed_normal(nodes.reshape(-1, 3), key).reshape(nodes.shape[:3])
    return scipy.ndimage.map_coordinates(node_noise, (scaled - lo).T, order=1)



@dataclass
class Phantom:
    """
    A dataclass representing a phantom for simulating ultrasound imaging.

    Attributes:
        voxel_dims: The dimensions of each voxel in the phantom, in meters.
        matrix_dims: The dimensions of the phantom matrix.
        baseline: Tuple of baseline sound speed and density values.
        seed: The seed for the random number generator used to generate the phantom.
        from_mask: Whether the phantom was created from a mask.
        world_texture: Generate tissue texture as a deterministic function of world position
            (requires a seed), so every simulation that overlaps a region sees the same scatterers.
        mask: A 3D array representing the phantom mask.
        tissues: A dictionary of Tissue objects representing the different types of tissue in the phantom.
        complete: The complete phantom representation.
        default_tissue: The default tissue label.
        rng: The random number generator used to generate the phantom.

    Methods:
        save(filepath): Saves the phantom to the specified file path.
        load(source_path): Loads the phantom from the specified source directory.
        create_from_image(): Sets the phantom mask and estimates the tissues from a Hounsfield unit-scaled image.
        create_from_list(): Sets the phantom mask and reads the tissues by supplying a list of shapes and corresponding tissues.
        add_tissue(tissue): Adds a tissue to the phantom.
        remove_tissue(tissue): Removes a tissue from the phantom.
        add_tissue_sphere(centroid, radius, tissue): Adds a spherical region of tissue to the phantom.
        get_tissues(): Returns a dictionary of the different types of tissue in the phantom.
        get_tissue_mask(): Returns a mask containing only tissues of a particular type.
        get_mask(): Returns the phantom mask.
        get_dimensions(): Returns the dimensions of the phantom matrix.
        get_voxel_dims(): Returns the dimensions of each voxel in the phantom, in meters.
        generate_tissue(tissue, rand_like=None, order=1): Generates a 3D array representing a tissue in the phantom.
        get_complete(): Returns a 3D array representing the complete phantom.
        render(): Renders the phantom.
    """

    voxel_dims: np.ndarray = field(default_factory=lambda: np.array((1e-3, 1e-3, 1e-3)))
    matrix_dims: np.ndarray = field(default_factory=lambda: np.array((256, 256, 256)))
    baseline: Tuple[float, float] = (1500, 1000)
    seed: Optional[int] = None
    from_mask: bool = True
    world_texture: bool = False
    mask: np.ndarray = field(init=False)
    tissues: Dict[str, Tissue] = field(default_factory=dict, init=False)
    complete: Optional[np.ndarray] = field(default=None, init=False)
    default_tissue: int = field(default=0, init=False)
    rng: np.random.Generator = field(init=False)

    def __post_init__(self):
        """Initialize fields after the dataclass initialization."""
        # Cast inputs to correct types
        self.voxel_dims = np.array(self.voxel_dims)
        self.matrix_dims = np.array(self.matrix_dims, dtype=int)
        
        # Initialize random number generator
        self.rng = np.random.default_rng(self.seed)
        
        # Initialize mask
        self.mask = np.zeros(self.matrix_dims, dtype=np.float32)
        
        # Initialize tissues dictionary with water as default
        self.tissues = {
            "water": Tissue(name="water", label=0, c=1500, rho=1000, sigma=0, scale=0.1)
        }

    @classmethod
    def from_source(cls, source_path):
        """Create a phantom instance from a source directory."""
        if source_path is not None:
            return cls.load(source_path)
        return cls()

    # save phantom to source dir containing tissues, mask, and source
    def save(self, filepath):
        filepath = os.fspath(filepath)
        os.makedirs(filepath, exist_ok=True)
        utils.dict_to_json(
            self.__save_tissues(), os.path.join(filepath, "tissues.json")
        )
        utils.save_array(self.mask, os.path.join(filepath, "mask.npz"))
        if self.complete is not None:
            utils.save_array(
                self.complete, os.path.join(filepath, "complete.npy"), compression=False
            )
        dictionary = self.__dict__.copy()
        dictionary.pop("mask")
        dictionary.pop("tissues")
        dictionary.pop("rng")
        if "complete" in dictionary.keys():
            dictionary.pop("complete")
        utils.dict_to_json(dict(dictionary), os.path.join(filepath, "source.json"))

    # load phantom from source dir
    @classmethod
    def load(cls, source_path):
        source_path = os.fspath(source_path)
        phantom = cls()
        assert os.path.exists(source_path), "provided path does not exist"
        assert os.path.exists(
            os.path.join(source_path, "tissues.json")
        ), "missing tissues file"
        assert os.path.exists(
            os.path.join(source_path, "source.json")
        ), "missing source file"
        assert os.path.exists(os.path.join(source_path, "mask.npz")) or os.path.exists(
            os.path.join(source_path, "mask.npy")
        ), "missing mask file"

        phantom.tissues = phantom.__load_tissues(
            utils.json_to_dict(os.path.join(source_path, "tissues.json"))
        )
        phantom.mask = utils.load_array(os.path.join(source_path, "mask"))
        if os.path.exists(os.path.join(source_path, "complete.npy")):
            phantom.complete = utils.load_array(
                os.path.join(source_path, "complete.npy")
            )
        elif os.path.exists(os.path.join(source_path, "complete.npz")):
            phantom.complete = utils.load_array(
                os.path.join(source_path, "complete.npz")
            )
        else:
            phantom.complete = None
        source = utils.json_to_dict(os.path.join(source_path, "source.json"))

        phantom.rng = np.random.default_rng(source["seed"])
        phantom.voxel_dims = np.array(source["voxel_dims"])
        phantom.matrix_dims = np.array(source["matrix_dims"], dtype=int)
        phantom.seed = source["seed"]
        phantom.from_mask = source["from_mask"]
        phantom.world_texture = source.get("world_texture", False)
        phantom.baseline = source["baseline"]
        if "default_tissue" in source.keys():
            phantom.default_tissue = source["default_tissue"]
        else:
            phantom.default_tissue = 0
        return phantom

    # save tissues to dictionary
    def __save_tissues(
        self,
    ):
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
    def create_from_image(
        self,
        image,
        input_voxel_size,
        target_voxel_size=None,
        transfer_fn=None,
        scale_w_noise=False,
    ):
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

        assert len(input_voxel_size) == 3, (
            "input voxel size must be a tuple of length 3"
        )
        assert len(target_voxel_size) == 3, (
            "target voxel size must be a tuple of length 3"
        )

        transformed_x = x * input_voxel_size[0] / target_voxel_size[0]
        transformed_y = y * input_voxel_size[1] / target_voxel_size[1]
        transformed_z = z * input_voxel_size[2] / target_voxel_size[2]

        from scipy.interpolate import RegularGridInterpolator

        interp = RegularGridInterpolator(
            (transformed_x, transformed_y, transformed_z), data
        )
        # interp = NearestNDInterpolator((transformed_x, transformed_y, transformed_z), data)

        points = np.stack(
            np.meshgrid(
                np.arange(0, transformed_x[-1]),
                np.arange(0, transformed_y[-1]),
                np.arange(0, transformed_z[-1]),
                indexing="ij",
            ),
            axis=-1,
        )

        if points.shape[0] * points.shape[1] * points.shape[2] > 5e8:
            print(
                "desired phantom array size is very large (>500,000,000 voxels), consider supplying a larger target_voxel_size or cropping the input image"
            )
        if points.shape[0] * points.shape[1] * points.shape[2] > 2e9:
            print(
                "desired phantom array size is too large (>2e9 voxels), consider supplying a larger target_voxel_size or cropping the input image"
            )
            return 0

        new_phantom = interp(points)

        if scale_w_noise:
            scaled_phantom = (new_phantom - np.amin(new_phantom)) / (
                np.amax(new_phantom) - np.amin(new_phantom)
            )
            noise_vec = (
                self.rng.standard_normal(new_phantom.shape) * scaled_phantom * 50
            )
            new_phantom = new_phantom + noise_vec
        self.complete = np.stack(
            (new_phantom * self.baseline[0] / self.baseline[1], new_phantom), axis=0
        )
        self.voxel_dims = np.array(target_voxel_size)
        self.matrix_dims = np.array(new_phantom.shape)
        self.mask = np.zeros(self.matrix_dims)
        self.from_mask = False
        self.tissues = {}

    # set mask and read tissues by supplying a list of shapes and corresponding tissues
    def create_from_list(
        self,
    ):
        print("not yet implemented")

    def __getitem__(self, key):
        if isinstance(key, slice) or isinstance(key, int) or isinstance(key, tuple):
            return self.mask[key]

    def __setitem__(self, key, value):
        assert value in [t.label for t in self.tissues.values()], (
            "value must be a valid tissue label"
        )
        if isinstance(key, slice) or isinstance(key, int) or isinstance(key, tuple):
            self.mask[key] = value

    def build_organ_from_mesh(
        self, surface_mesh, voxel_size, tissue_list, dir_path=None, file_list=None
    ):
        from utils import phantom_builder

        if dir_path is not None:
            all_files = [
                f
                for f in listdir(dir_path)
                if os.path.isfile(os.path.join(dir_path, f))
            ]
            files = [f for f in all_files if os.path.splitext(f)[-1].lower() == ".obj"]
            if not len(files):
                files = [
                    f for f in all_files if os.path.splitext(f)[-1].lower() == ".ply"
                ]
            if not len(files):
                raise Exception(
                    "No valid mesh files found in directory, looking for '.obj' or '.ply' files"
                )
            files.sort()
            file_list = [os.path.join(dir_path, f) for f in files]
        else:
            if file_list is None:
                raise Exception(
                    "Please supply either a directory or a list of file paths"
                )
        min_bound = surface_mesh.get_min_bound()
        max_bound = surface_mesh.get_max_bound()
        for tissue, file in zip(tissue_list, file_list):
            self.add_tissue(
                tissue,
                mask=phantom_builder.voxelize(
                    voxel_size,
                    mesh_file=file,
                    min_bounds=min_bound,
                    max_bounds=max_bound,
                    grid_shape=self.matrix_dims,
                ),
            )
            print(f"Added {tissue.name}")
        self.matrix_dims = np.array(self.mask.shape)

    # add tissue
    def add_tissue(self, tissue, mask=None):
        self.from_mask = True
        self.tissues[tissue.name] = tissue
        if mask is not None:
            self.mask = np.where(mask, tissue.label, self.mask)
            self.complete = None

    def remove_tissue(self, name):
        if name in self.tissues.keys():
            self.mask = np.where(self.mask == self.tissues[name].label, 0, self.mask)
            del self.tissues[name]
            self.complete = None
        else:
            print("tissue not found")

    def set_default_tissue(self, name):
        if type(name) == str and name in self.tissues.keys():
            self.default_tissue = self.tissues[name].label
        elif hasattr(name, "label"):
            self.add_tissue(name)
            self.default_tissue = name.label
        else:
            print("tissue not found")
        self.mask = np.where(self.mask == 0, self.default_tissue, self.mask)

    def add_tissue_sphere(self, centroid, radius, tissue):
        region = geometry.create_sphere(
            centroid, radius, self.voxel_dims, self.matrix_dims
        )
        if tissue.name not in self.tissues.keys():
            self.tissues[tissue.name] = tissue
        self.mask = np.where(region, tissue.label, self.mask)
        self.complete = None

    def get_tissues(
        self,
    ):
        return self.tissues

    def get_tissue_mask(
        self,
    ):  # return a mask containing only tissues of a particular type
        print("not yet implemented")

    def get_mask(
        self,
    ):
        return self.mask

    def get_dimensions(
        self,
    ):
        return self.matrix_dims

    def get_voxel_dims(
        self,
    ):
        return self.voxel_dims

    def __randn_like(self, array):
        return self.rng.standard_normal(array.shape).astype(array.dtype)

    def __rand_like(self, array):
        return self.rng.uniform(array.shape).astype(array.dtype) * 2 - 1

    def generate_tissue(self, tissue, shape, voxel_size, rand_like=None, order=1):
        if rand_like is None:
            rand_like = self.__randn_like

        if tissue.sigma == 0:
            return np.stack(
                (
                    np.ones(shape, dtype=np.float16) * tissue.c,
                    np.ones(shape, dtype=np.float16) * tissue.rho,
                ),
                axis=0,
            )

        size, scaling_factor, std_scale = self.__texture_parameters(tissue, shape, voxel_size)

        tissue_matrix = np.ones(size, dtype=np.float32) * tissue.c
        tissue_matrix = (
            tissue_matrix + rand_like(tissue_matrix) * tissue.sigma * std_scale
        )

        if scaling_factor != 1:
            tissue_matrix = scipy.ndimage.zoom(tissue_matrix, scaling_factor, order=order)
        tissue_matrix = tissue_matrix[
            : shape[0],
            : shape[1],
            : shape[2],
        ]
        return np.stack((tissue_matrix, tissue_matrix / tissue.c * tissue.rho), axis=0)

    def __texture_parameters(self, tissue, shape, voxel_size):
        """Coarse noise grid size, its upsampling factor, and the noise std multiplier.

        Scatterers larger than a voxel are drawn on a coarse grid and linearly upsampled,
        giving texture correlated over `tissue.scale`. Smaller ones collapse to per-voxel
        white noise (upsampling factor 1) with a reduced std.
        """
        scaling_factor = tissue.scale / np.mean(voxel_size)
        scaling_factor = float(
            np.clip(scaling_factor, 1 / np.mean(shape), np.mean(shape) * 100)
        )
        size = tuple(int(n // scaling_factor + 1) for n in shape)

        # Implement anisotropy later

        # if multiple scatterers in one voxel, modify std of noise: in 3 dimensions, std of iid gaussians scales with ^3/2
        # Note: the exponent below is 1/2, not the 3/2 described above. With 1/2, speckle intensity
        # scales as sigma^2 * scale * voxel^2, so it shifts with voxel size (on the simulation grid,
        # voxel ~ 1/f, giving ~f^2 rather than the Rayleigh f^4). With 3/2 it would be
        # sigma^2 * scale^3: independent of voxel size and consistent with the correlated branch.
        # Changing it re-balances tissues with different `scale` in existing phantoms.
        if np.mean(size) > np.mean(shape):
            return tuple(shape), 1, scaling_factor**0.5
        return size, scaling_factor, 1.0

    def __tissue_values(self, tissue, selection, voxel_size, rng, positions, order=1):
        """Sound speed for the voxels in `selection`, matching `generate_tissue` statistically.

        Noise is drawn only where the tissue is: per selected voxel for white noise, or,
        for correlated texture, on the part of `generate_tissue`'s coarse grid that the
        selection touches, sampled at the same coordinates `zoom` would use. With
        `world_texture`, noise instead comes from a hash of world-space lattice cells.
        """
        count = int(np.count_nonzero(selection))
        if tissue.sigma == 0:
            return np.full(count, tissue.c, dtype=np.float32)

        size, scaling_factor, std_scale = self.__texture_parameters(
            tissue, selection.shape, voxel_size
        )
        if self.world_texture:
            key = _texture_key(self.seed, tissue.label)
            spacing = scaling_factor * np.asarray(voxel_size, dtype=np.float64)
            noise = _world_noise(positions(selection), spacing, key, interpolate=scaling_factor != 1)
            return np.float32(tissue.c) + noise * np.float32(tissue.sigma * std_scale)

        if scaling_factor == 1:
            noise = rng.standard_normal(count, dtype=np.float32)
            return np.float32(tissue.c) + noise * np.float32(tissue.sigma * std_scale)

        # `zoom` maps output index i to input coordinate i * (n_in - 1) / (n_out - 1).
        size = np.array(size)
        zoomed = np.array([int(round(n * scaling_factor)) for n in size])
        ratio = (size - 1) / np.maximum(zoomed - 1, 1)
        coarse_coords = [index * r for index, r in zip(np.nonzero(selection), ratio)]
        lo = np.array([int(np.floor(p.min())) for p in coarse_coords])
        hi = np.array([int(np.floor(p.max())) + 2 for p in coarse_coords])
        hi = np.minimum(hi, size)
        coarse = np.float32(tissue.c) + rng.standard_normal(tuple(hi - lo), dtype=np.float32) * np.float32(tissue.sigma)
        return scipy.ndimage.map_coordinates(
            coarse, [p - l for p, l in zip(coarse_coords, lo)], order=order
        )

    def simulation_rng(self, index):
        """Random stream for one simulation, independent across indices and worker processes.

        Forked workers inherit identical copies of `self.rng`, so drawing from it per
        simulation repeats noise between workers. With a seed, the stream is derived
        from (seed, index) and is reproducible; without one, it is fresh entropy.
        """
        if self.seed is None:
            return np.random.default_rng()
        return np.random.default_rng(np.random.SeedSequence(self.seed, spawn_key=(int(index),)))

    def make_complete(self, mask, voxel_size, rng=None, world=None):
        """Sound speed and density maps, shape (2, *mask.shape), for a label mask.

        Each tissue's texture is generated only over the voxels it occupies (see
        `generate_tissue` for the texture model). Voxels whose label matches no tissue
        take the default tissue.

        Args:
            mask: Label array.
            voxel_size: Voxel size of `mask` in meters.
            rng: Random generator for the texture (default `self.rng`); unused with `world_texture`.
            world: World positions (*mask.shape, 3) of the voxels, used with `world_texture`.
                Defaults to `mask` being this phantom's own grid, centered on the origin.
        """
        if rng is None:
            rng = self.rng
        if self.world_texture and self.seed is None:
            raise ValueError("world_texture needs a seed so every simulation sees the same scatterers")

        def positions(selection):
            if world is not None:
                return world[selection]
            voxel = np.asarray(voxel_size, dtype=np.float64)
            return (np.argwhere(selection) - np.array(mask.shape) / 2) * voxel

        complete = np.empty((2, *mask.shape), dtype=np.float32)
        known = [t.label for t in self.tissues.values()]
        unmatched = ~np.isin(mask, known)
        if unmatched.any() and self.default_tissue not in known:
            raise ValueError("mask contains labels with no tissue and no default tissue to fill them")

        for tissue in self.tissues.values():
            selection = mask == tissue.label
            if tissue.label == self.default_tissue:
                selection |= unmatched
            if not selection.any():
                continue
            values = self.__tissue_values(tissue, selection, voxel_size, rng, positions)
            complete[0][selection] = values
            if tissue.sigma == 0:
                complete[1][selection] = tissue.rho
            else:
                complete[1][selection] = values / np.float32(tissue.c) * np.float32(tissue.rho)
        return complete

    def get_complete(self):
        if self.complete is None:
            self.complete = self.make_complete(self.mask, self.voxel_dims)
        return self.complete

    def interpolate_phantom(self, bounds, transform, voxel_size, matrix_size, rng=None):
        X, Y, Z = np.meshgrid(
            np.linspace(bounds[0][0], bounds[-1][0], matrix_size[0]),
            np.linspace(bounds[0][1], bounds[-1][1], matrix_size[1]),
            np.linspace(bounds[0][2], bounds[-1][2], matrix_size[2]),
            indexing="ij",
        )

        local_points = np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=-1)

        global_points = transform.apply_to_points(local_points, inverse=False)

        global_indices = global_points / self.voxel_dims.T + self.matrix_dims.T / 2

        # Nearest-neighbour lookup on the voxel lattice: round each index and clamp to the
        # grid, which is how a nearest-neighbour interpolator extrapolates past the edges.
        nearest = np.clip(np.rint(global_indices), 0, self.matrix_dims - 1).astype(np.intp)
        nearest = tuple(nearest.T)

        if self.from_mask:
            final_mask = self.mask[nearest].reshape(matrix_size)
            world = global_points.reshape(*final_mask.shape, 3) if self.world_texture else None
            final = self.make_complete(mask=final_mask, voxel_size=voxel_size, rng=rng, world=world)
        else:
            complete = self.get_complete()
            final_sos = complete[0][nearest].reshape(matrix_size)
            final_density = complete[1][nearest].reshape(matrix_size)
            final = np.stack((final_sos, final_density), axis=0)

        return final

    def interpolate_up(self, matrix, input_voxel_size, target_voxel_size):
        x = np.linspace(0, matrix.shape[-3], matrix.shape[-3])
        y = np.linspace(0, matrix.shape[-2], matrix.shape[-2])
        z = np.linspace(0, matrix.shape[-1], matrix.shape[-1])

        transformed_x = x * input_voxel_size[0] / target_voxel_size[0]
        transformed_y = y * input_voxel_size[1] / target_voxel_size[1]
        transformed_z = z * input_voxel_size[2] / target_voxel_size[2]

        points = np.stack(
            np.meshgrid(
                np.arange(0, transformed_x[-1]),
                np.arange(0, transformed_y[-1]),
                np.arange(0, transformed_z[-1]),
                indexing="ij",
            ),
            axis=-1,
        )
        if points.shape[0] * points.shape[1] * points.shape[2] > 5e8:
            print(
                "desired phantom array size is very large (>500,000,000 voxels), consider reducing the simulation grid size or increasing the simulation voxel size"
            )
        if points.shape[0] * points.shape[1] * points.shape[2] > 2e9:
            assert False, (
                "desired phantom array size is too large (>2e9 voxels), consider reducing the simulation grid size or increasing the simulation voxel size"
            )

        if len(matrix.shape) > 3:
            sampled_matrix = []
            for i in range(matrix.shape[0]):
                interp = RegularGridInterpolator(
                    (transformed_x, transformed_y, transformed_z),
                    matrix[i],
                    method="nearest",
                )
                sampled_matrix.append(interp(points))
            sampled_matrix = np.stack(sampled_matrix, axis=0)
        else:
            interp = RegularGridInterpolator(
                (transformed_x, transformed_y, transformed_z), matrix, method="nearest"
            )
            sampled_matrix = interp(points)
        return sampled_matrix

    # crop phantom to the grid size of the simulation
    def __crop_phantom(self, sim_phantom, center):
        crop_size = [
            self.sim_properties.grid_size[0] - 2 * self.sim_properties.PML_size[0],
            self.sim_properties.grid_size[1] - 2 * self.sim_properties.PML_size[1],
            self.sim_properties.grid_size[2] - 2 * self.sim_properties.PML_size[2],
        ]

        axial_zero = int(center[-3])

        if crop_size[-3] > axial_zero:
            sim_phantom = np.stack(
                (
                    np.pad(
                        sim_phantom[0],
                        ((0, crop_size[-3]), (0, 0), (0, 0)),
                        "constant",
                        constant_values=(self.phantom.baseline[0],),
                    ),
                    np.pad(
                        sim_phantom[1],
                        ((0, crop_size[-3]), (0, 0), (0, 0)),
                        "constant",
                        constant_values=(self.phantom.baseline[1],),
                    ),
                ),
                axis=0,
            )

        if crop_size[-2] // 2 > int(center[-2]) or crop_size[
            -2
        ] // 2 > sim_phantom.shape[-2] - int(center[-2]):
            expand = max(
                crop_size[-2] // 2 - int(center[-2]),
                crop_size[-2] // 2 - (sim_phantom.shape[-2] - int(center[-2])),
            )
            sim_phantom = np.stack(
                (
                    np.pad(
                        sim_phantom[0],
                        ((0, 0), (expand, expand), (0, 0)),
                        "constant",
                        constant_values=(self.phantom.baseline[0],),
                    ),
                    np.pad(
                        sim_phantom[1],
                        ((0, 0), (expand, expand), (0, 0)),
                        "constant",
                        constant_values=(self.phantom.baseline[1],),
                    ),
                ),
                axis=0,
            )
            center = center + np.array([0, expand // 2, 0])

        if crop_size[-1] // 2 > int(center[-1]) or crop_size[
            -1
        ] // 2 > sim_phantom.shape[-1] - int(center[-1]):
            expand = max(
                crop_size[-1] // 2 - int(center[-1]),
                crop_size[-1] // 2 - (sim_phantom.shape[-1] - int(center[-1])),
            )
            sim_phantom = np.stack(
                (
                    np.pad(
                        sim_phantom[0],
                        ((0, 0), (0, 0), (expand, expand)),
                        "constant",
                        constant_values=(self.phantom.baseline[0],),
                    ),
                    np.pad(
                        sim_phantom[1],
                        ((0, 0), (0, 0), (expand, expand)),
                        "constant",
                        constant_values=(self.phantom.baseline[1],),
                    ),
                ),
                axis=0,
            )
            center = center + np.array([0, 0, expand // 2])

        return sim_phantom[
            :,
            axial_zero : crop_size[-3] + axial_zero,
            int((center[-2] - crop_size[-2] / 2)) : int(
                (center[-2] + crop_size[-2] / 2)
            ),
            int((center[-1] - crop_size[-1] / 2)) : int(
                (center[-1] + crop_size[-1] / 2)
            ),
        ]

    def render(
        self,
    ):
        print("not yet implemented")
