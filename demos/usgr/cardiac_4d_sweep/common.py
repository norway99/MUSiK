"""Shared configuration, paths, and geometry for the cardiac 4D sweep scripts.

The pose and sector helpers are trimmed copies of the ones in `synthetic/generate.py`.
That module imports from the usgr package, which is not installed alongside MUSiK.
Conventions match it: probe-local X is the beam axis, Y is lateral, Z is elevation, and
poses are 4x4 probe-to-world matrices in meters.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.interpolate import CubicSpline
from scipy.ndimage import center_of_mass, map_coordinates
from scipy.spatial.transform import Rotation, RotationSpline

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.append(str(REPO_ROOT))

DEFAULT_CONFIG = HERE / "config.yaml"


# ---------------------------------------------------------------------------
# Configuration and paths
# ---------------------------------------------------------------------------


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG) as handle:
        return yaml.safe_load(handle)


class Paths:
    """Every on-disk location, derived from `paths.output_root`."""

    def __init__(self, config: dict):
        self.root = Path(config["paths"]["output_root"])
        self.phantoms = self.root / "phantoms"
        self.trajectory = self.root / "trajectory"
        self.experiments = self.root / "experiments"
        self.recording = self.root / "recording"
        self.qa = self.root / "qa"

    def phantom(self, frame: int) -> Path:
        return self.phantoms / f"frame_{frame:02d}"

    def experiment(self, frame: int) -> Path:
        return self.experiments / f"phase_{frame:02d}"


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(data, handle, indent=2)


# ---------------------------------------------------------------------------
# Mask
# ---------------------------------------------------------------------------


def load_mask(config: dict) -> np.ndarray:
    """The raw 4D label mask as `(T, A0, A1, A2)` in the file's own spatial axis order.

    Axes are never transposed: reordering them can flip handedness and mirror the anatomy.
    """
    path = Path(config["paths"]["mask"])
    if path.suffix == ".npy":
        mask = np.load(path)
    elif path.suffix in (".tif", ".tiff"):
        import tifffile  # ImageJ hyperstack, (T, Z, Y, X)

        mask = tifffile.imread(path)
    else:
        import nibabel  # only needed for NIfTI input

        mask = np.moveaxis(np.asarray(nibabel.load(path).dataobj), -1, 0)
    expected = config["cardiac"]["num_frames"]
    if mask.ndim != 4 or mask.shape[0] != expected:
        raise ValueError(f"expected a (T={expected}, ...) 4D mask, got {mask.shape}")
    return mask.astype(np.uint8)


def load_cropped_mask(paths: Paths) -> np.ndarray:
    """The cropped and padded mask cached by `01_build_phantoms.py`, memory-mapped."""
    return np.load(paths.phantoms / "mask.npy", mmap_mode="r")


def heart_center_m(mask_frame: np.ndarray, spacing_m: np.ndarray, heart_labels: list[int]) -> np.ndarray:
    """Centroid of the heart labels in the world frame (grid centered at its middle)."""
    lookup = np.zeros(256, dtype=bool)
    lookup[heart_labels] = True
    centroid = np.array(center_of_mass(lookup[mask_frame]))
    return (centroid - np.array(mask_frame.shape) / 2.0) * spacing_m


def sample_labels(mask_frame: np.ndarray, spacing_m: np.ndarray, world: np.ndarray) -> np.ndarray:
    """Nearest-neighbour label lookup at world points (N, 3); outside the grid reads 0."""
    indices = world / spacing_m + np.array(mask_frame.shape) / 2.0
    return map_coordinates(mask_frame, indices.T, order=0, mode="constant", cval=0)


# ---------------------------------------------------------------------------
# Cardiac timing
# ---------------------------------------------------------------------------


def cardiac_phase(times_s: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Continuous phase in [0, 1) and the nearest CT frame index for each time."""
    cardiac = config["cardiac"]
    phase = np.mod(cardiac["phase_offset"] + times_s * cardiac["heart_rate_bpm"] / 60.0, 1.0)
    frame = np.round(phase * cardiac["num_frames"]).astype(int) % cardiac["num_frames"]
    return phase, frame


# ---------------------------------------------------------------------------
# Poses
# ---------------------------------------------------------------------------


def pose_matrix_to_12(pose: np.ndarray) -> list[float]:
    """Convert a 4x4 pose matrix to the row-major 12D pose representation."""
    return pose[:3, :3].reshape(-1).tolist() + pose[:3, 3].tolist()


def look_at_rotation(
    position: np.ndarray, target: np.ndarray, roll_rad: float, lateral_hint: np.ndarray | None = None
) -> np.ndarray:
    """Probe orientation whose local X axis (the beam axis) points at the target.

    With `lateral_hint`, the image's lateral axis (before roll) is that direction made
    orthogonal to the beam. Prescribed sweeps should pass one: the world-up fallback
    switches near vertical beams, which can flip the image mid-sweep.
    """
    forward = np.asarray(target, dtype=np.float64) - np.asarray(position, dtype=np.float64)
    norm = np.linalg.norm(forward)
    if norm < 1e-9:
        raise ValueError("Probe position must not coincide with the look-at target")
    forward = forward / norm

    if lateral_hint is not None:
        lateral = np.asarray(lateral_hint, dtype=np.float64)
        lateral = lateral - np.dot(lateral, forward) * forward
        if np.linalg.norm(lateral) < 1e-6:
            raise ValueError("lateral hint must not be parallel to the beam")
    else:
        world_up = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(forward, world_up))) > 0.95:
            world_up = np.array([0.0, 1.0, 0.0])
        lateral = np.cross(world_up, forward)
    lateral = lateral / np.linalg.norm(lateral)
    elevation = np.cross(forward, lateral)

    base = np.column_stack([forward, lateral, elevation])
    return Rotation.from_rotvec(roll_rad * forward).as_matrix() @ base


def interpolate_poses(knot_times: np.ndarray, knot_poses: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Cubic spline in position and rotation spline in orientation, onto `times`."""
    poses = np.tile(np.eye(4), (len(times), 1, 1))
    if len(knot_times) < 2:
        poses[:] = knot_poses[0]
        return poses
    poses[:, :3, 3] = CubicSpline(knot_times, knot_poses[:, :3, 3], axis=0, bc_type="natural")(times)
    rotations = RotationSpline(knot_times, Rotation.from_matrix(knot_poses[:, :3, :3]))
    poses[:, :3, :3] = rotations(times).as_matrix()
    return poses


def to_musik_transform(pose: np.ndarray):
    from musik.utils import geometry

    return geometry.Transform(rotation=pose[:3, :3], translation=pose[:3, 3], from_matrix=True)


# ---------------------------------------------------------------------------
# Sector
# ---------------------------------------------------------------------------


@dataclass
class Sector:
    """Fixed sector geometry and the per-pixel maps derived from it."""

    image_size: tuple[int, int]
    fov_degrees: float
    apex_pixels: tuple[float, float]
    radius_pixels: float
    radius_m: float
    microns_per_pixel: float
    local_points: np.ndarray  # (H, W, 3) probe-frame position in meters
    radius: np.ndarray  # (H, W) distance from the apex in meters
    angle: np.ndarray  # (H, W) beam angle in radians
    valid: np.ndarray  # (H, W) inside the sector


def create_sector(config: dict) -> Sector:
    """Build a centered sector filling the image width, imaging to `depth_mm`."""
    acquisition = config["acquisition"]
    height, width = acquisition["image_size"]
    fov_degrees, depth_m = acquisition["fov_degrees"], acquisition["depth_mm"] * 1e-3
    half_fov = math.radians(fov_degrees) / 2.0
    radius_pixels = width / (2.0 * math.sin(half_fov))
    apex_pixels = (round((height - radius_pixels) / 2.0), round(width / 2.0))
    microns_per_pixel = depth_m * 1e6 / radius_pixels

    rows = np.arange(height, dtype=np.float64)[:, None] + 0.5 - apex_pixels[0]
    cols = np.arange(width, dtype=np.float64)[None, :] + 0.5 - apex_pixels[1]
    rows, cols = np.broadcast_arrays(rows, cols)
    radius = np.hypot(rows, cols)
    angle = np.arctan2(cols, rows)
    local = np.stack([rows, cols, np.zeros_like(rows)], axis=-1) * (microns_per_pixel * 1e-6)
    return Sector(
        image_size=(height, width),
        fov_degrees=fov_degrees,
        apex_pixels=(float(apex_pixels[0]), float(apex_pixels[1])),
        radius_pixels=radius_pixels,
        radius_m=depth_m,
        microns_per_pixel=microns_per_pixel,
        local_points=local,
        radius=radius * microns_per_pixel * 1e-6,
        angle=angle,
        valid=(radius > 0) & (radius < radius_pixels) & (np.abs(angle) < half_fov),
    )


def sector_world_points(sector: Sector, pose: np.ndarray) -> np.ndarray:
    """World positions (M, 3) of the sector's valid pixels under one probe pose."""
    return sector.local_points[sector.valid] @ pose[:3, :3].T + pose[:3, 3]


def label_slice(sector: Sector, pose: np.ndarray, mask_frame: np.ndarray, spacing_m: np.ndarray) -> np.ndarray:
    """The label mask cut by the image plane, as an (H, W) image (0 outside the sector)."""
    mask_frame = np.asarray(mask_frame)
    image = np.zeros(sector.image_size, dtype=mask_frame.dtype)
    image[sector.valid] = sample_labels(mask_frame, spacing_m, sector_world_points(sector, pose))
    return image


def load_trajectory(paths: Paths) -> dict:
    with np.load(paths.trajectory / "trajectory.npz") as data:
        return {key: data[key] for key in data.files}
