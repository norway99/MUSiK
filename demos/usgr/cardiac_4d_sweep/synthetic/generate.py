"""Generate a synthetic ultrasound recording for reconstruction.

An analytic phantom of pulsating ellipsoids is rendered through a sector probe
following a smooth trajectory, and written as a v3.0 recording that
`ReconstructionDataset` loads exactly like an in-vivo one.

Three parts are deliberate:

- **Anatomy** is a pure function of world position and cardiac phase. Nothing
  view-dependent lives in it, so a reconstruction that reproduces frames by
  memorizing view direction is measurably wrong.
- **Appearance** is a multiplicative gain field over the sector in
  (depth, angle). It is view-dependent by construction and belongs to the
  observation model, never to the anatomy — but every artifact is *caused* by a
  particular ellipsoid. Lesions of three kinds (`enhancing`, `attenuating`,
  `specular`) each cast their own, placed and scaled by how the current frame
  cuts them, so the same view reproduces the same artifact and a view that misses
  the lesion sees none of it. Only a standing depth tilt and the additive noise
  are unattributable, so brightness structure a reconstruction cannot explain
  from the scene is a failure rather than an unmodellable term.
- **Pose error** uses tracker-origin coherent bias and colored/white fluctuations,
  independently of the correction model. Frames are rendered with true transducer
  poses; tracking errors pass through the actual tracker-to-transducer calibration.

Usage:
    python -m usgr.reconstruction.synthetic \
        --output-dir ./phantom_data --num-frames 1000

Every config field below is a CLI flag: `--phantom-num-ellipsoids 20`,
`--appearance-shadow-strength 0.9`, `--pose-error-bias-translation-rms-mm 2`.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import imageio_ffmpeg
import matplotlib
import numpy as np
from PIL import Image
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial.transform import Rotation, RotationSpline

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ...configs import DEFAULT_CALIBRATION  # noqa: E402
from ...configs.utils import add_config_args, apply_cli_overrides, load_yaml_config  # noqa: E402
from ...recording.data_access.load_sequence import (  # noqa: E402
    compute_tracker_from_transducer,
    compute_transducer_from_tracker,
    load_sequence,
)
from ...recording.data_access.metadata_schema import (  # noqa: E402
    CURRENT_FORMAT_VERSION,
    CalibrationReference,
    FrameMetadata,
    GeometryConfig,
    SequenceMetadataV3,
    SourceMetadata,
)
from ...utils.visualization_utils import (  # noqa: E402
    compute_pose_plot_bounds,
    plot_pose_pair_comparison_3d,
)

from .pose_error import PoseError, PoseErrorConfig, generate_pose_error, save_diagnostics, validate_trajectory
from .random_streams import child_seed, named_rng


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrajectoryConfig:
    """Probe path. Zero speeds hold the probe still, which is the static case."""

    frame_rate_hz: float = 10.0
    keyframe_interval_seconds: float = 4.0
    standoff_m: float = 0.10
    translation_speed_mps: float = 0.01
    rotation_speed_dps: float = 60.0
    look_at_radius_m: float = 0.05


@dataclass(frozen=True)
class PhantomConfig:
    """Pulsating ellipsoid anatomy. A `bpm_range` of `(0, 0)` freezes it.

    Lesions are ordinary ellipsoids of the anatomy that additionally cause an
    acoustic artifact wherever a frame cuts through them; they are added on top of
    `num_ellipsoids` background structures. Their size and echogenicity come from
    the module constants below, since what distinguishes a kind is the artifact it
    causes, not a value worth tuning per run.
    """

    num_ellipsoids: int = 10
    num_enhancing_lesions: int = 5
    num_attenuating_lesions: int = 5
    num_specular_lesions: int = 0
    bpm_range: tuple[float, float] = (50.0, 80.0)
    bpm_keyframe_interval_seconds: float = 8.0
    # Echogenicity is signed and accumulates where ellipsoids overlap, so the
    # background and the per-ellipsoid range are set to keep typical sums inside
    # [0, 1]: saturated pixels are anatomy that no reconstruction can recover.
    background: float = 0.42
    extent_m: float = 0.1
    axis_range_m: tuple[float, float] = (0.010, 0.040)
    center_amplitude_range_m: tuple[float, float] = (0.001, 0.005)
    axis_amplitude_range_frac: tuple[float, float] = (0.02, 0.40)
    echogenicity_range: tuple[float, float] = (-0.5, 0.5)
    echogenicity_amplitude_range: tuple[float, float] = (0.07, 0.1)
    # Hollow ellipsoids: a bright wall of fixed thickness around unchanged interior,
    # standing in for thin anatomy such as valve leaflets and septa.
    num_shells: int = 5
    shell_axis_range_m: tuple[float, float] = (0.010, 0.040)
    shell_thickness_range_m: tuple[float, float] = (0.0015, 0.003)
    shell_echogenicity_range: tuple[float, float] = (0.3, 0.5)
    # Texture-only ellipsoids: no echogenicity of their own (mean intensity is whatever
    # surrounds them), but speckle and additive noise are zeroed inside. They test
    # whether a reconstruction represents texture separately from intensity.
    num_smooth: int = 10
    smooth_axis_range_m: tuple[float, float] = (0.010, 0.040)


@dataclass(frozen=True)
class AppearanceConfig:
    """View-dependent image formation. Any term is disabled by setting it to zero.

    Apart from the depth tilt and the noise, every term here is caused by a lesion
    of the matching kind: these knobs set how strong that lesion's artifact is, and
    the lesion counts in `PhantomConfig` set how many cast one. There are no
    free-floating artifacts, so all brightness structure is either attributable to
    the scene or is the standing tilt.
    """

    # With speckle on (the default), weak additive noise; 0.1 without speckle.
    noise_std: float = 0.05
    depth_gain: float = 0.25
    enhancing_gain: float = 0.6
    shadow_strength: float = 0.7
    specular_gain: float = 1.0
    specular_darkening: float = 0.6
    # Multiplicative speckle (0 disables it). A complex Gaussian field on the (range, beam
    # angle) grid, smoothed by an anisotropic PSF; its Rayleigh envelope is raised to
    # contrast, then normalized to mean 1. Grain is set in range (mm) and in
    # beam angle (degrees), so lateral grain widens with depth as in a phased sector.
    # Between frames the complex field decorrelates with true probe motion and cardiac
    # phase: rho = exp(-(d/d_mm)**2 - (dphase/d_phase)**2), d being how far the imaging
    # plane moved at `speckle_reference_depth_m`. Only consecutive frames correlate, so
    # returning to a view later gives fresh speckle.
    speckle_contrast: float = 0.4
    speckle_axial_mm: float = 0.3
    speckle_lateral_deg: float = 1.2
    speckle_decorrelation_mm: float = 1.0
    speckle_decorrelation_phase: float = 0.5
    speckle_reference_depth_m: float = 0.1


@dataclass(frozen=True)
class DeformationConfig:
    """Smooth non-rigid warp of each frame, standing in for tissue deformation.

    Off at zero amplitude. Pixel p (probe-local) images the anatomy at
    ``true_pose · (p + u_t(p))``: the poses are unchanged, the image content moves.
    `u_t` is a Gaussian random field on a probe-local grid, smoothed with a Gaussian
    of std `correlation_length_mm` (so its spatial correlation falls as
    exp(-d**2 / (4 l**2))), and evolving as AR(1) with rho = exp(-dt /
    `correlation_seconds`); 0 s gives an independent warp per frame. The best-fit
    rigid part of each frame's warp is removed, so none of it is pose error, and the
    rest is scaled to exactly `rms_mm` (in-plane vector RMS over the sector) and
    `elevational_rms_mm` (out of plane). A depth scale (speed-of-sound error) is one
    of the shapes it can take. Speckle, noise and the straight-ray shadow onsets stay
    unwarped.
    """

    rms_mm: float = 0.0
    elevational_rms_mm: float = 0.0
    correlation_length_mm: float = 20.0
    correlation_seconds: float = 1.0
    remove_rigid: bool = True


@dataclass(frozen=True)
class PerturbedTestConfig:
    """Test frames (metadata v5.0) appended after the recording at a prescribed offset.

    Each test frame picks a random training frame and moves its imaging frame by exactly
    `rotation_deg` about a random probe-local axis through the frame's centre (the
    centroid of the sector's pixels), then `translation_mm` along a random probe-local
    direction: `x' = R (x - c) + c + d`. Pivoting at the image centre rather than the
    probe origin keeps large offsets inside the sampled field of view, so how far test
    views sit from the training data is controlled rather than left to the trajectory.

    The raw pose carries the base frame's tracking error (`raw_b @ inv(true_b)`, applied
    on the left), the phase is uniform, and timestamps continue after the last frame. The
    training frames, their tracking error and images are exactly those of the same seed
    without test frames.
    """

    count: int = 0
    rotation_deg: float = 30.0
    translation_mm: float = 25.0

    def validate(self) -> None:
        if self.count < 0 or self.rotation_deg < 0 or self.translation_mm < 0:
            raise ValueError("perturbed_test count and magnitudes must be non-negative")


# Spacing of appended test-frame timestamps after the last training frame.
TEST_FRAME_INTERVAL_SECONDS = 1.0


def image_center(sector: "Sector") -> np.ndarray:
    """Probe-local centroid of the sector's pixels, metres: the centre of the imaging frame."""
    return sector.local_points[sector.valid].astype(np.float64).mean(axis=0)


def perturbed_test_frames(
    config: PerturbedTestConfig,
    true_poses: np.ndarray,
    rng: np.random.Generator,
    pivot: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Base training-frame indices and `[K, 4, 4]` true test poses, rotated about `pivot`."""
    base = rng.choice(len(true_poses), size=config.count, replace=True)
    axes = random_unit_vectors(rng, config.count)
    directions = random_unit_vectors(rng, config.count)
    rotations = Rotation.from_rotvec(axes * math.radians(config.rotation_deg)).as_matrix()
    offsets = np.tile(np.eye(4), (config.count, 1, 1))
    offsets[:, :3, :3] = rotations
    # x' = R (x - c) + c + d, in probe-local coordinates.
    offsets[:, :3, 3] = pivot - rotations @ pivot + directions * config.translation_mm * 1e-3
    return base, true_poses[base] @ offsets


@dataclass(frozen=True)
class SyntheticConfig:
    """Every generation parameter."""

    output_dir: str = "./phantom_data"
    num_frames: int = 1000
    seed: int = 1
    image_size: tuple[int, int] = (512, 512)
    fov_degrees: float = 90.0
    depth_m: float = 0.22
    # Replace the generated probe path with the poses of a real recording. The
    # phantom is moved to where that trajectory looks.
    invivo_trajectory_dir: Optional[str] = None
    eval_grid_resolution: int = 48
    eval_grid_phases: int = 16
    skip_eval_grid: bool = False
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    phantom: PhantomConfig = field(default_factory=PhantomConfig)
    appearance: AppearanceConfig = field(default_factory=AppearanceConfig)
    pose_error: PoseErrorConfig = field(default_factory=PoseErrorConfig)
    deformation: DeformationConfig = field(default_factory=DeformationConfig)
    perturbed_test: PerturbedTestConfig = field(default_factory=PerturbedTestConfig)


# Component phase offsets that decorrelate an ellipsoid's translation, dilation
# and echogenicity, so the phantom is not one global pulsation.
CENTER_PHASES = np.array([0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0])
AXIS_PHASES = math.pi / 3.0 + np.array([math.pi / 6.0, 5.0 * math.pi / 6.0, 3.0 * math.pi / 2.0])
ECHOGENICITY_PHASE = 2.0 * math.pi / 3.0

# Depth over which a shadow reaches full strength behind its blocker, as a
# fraction of the sector radius. Short, so the shadow stays visually attached to
# the surface casting it.
SHADOW_PENUMBRA_DEPTH = 0.015
# Stand-in depth for beams that never reach a lesion, past the sector's far edge.
BEYOND_FAR_EDGE = 2.0

# Lesion kinds, in the order `create_phantom` stacks them, and the artifact each
# causes in a frame that cuts through it.
LESION_KINDS = ("enhancing", "attenuating", "specular")
# A small bright reflector rings at its own range: a bright band across every beam,
# just behind it.
ENHANCING_AXIS_RANGE_M = (0.004, 0.010)
ENHANCING_ECHOGENICITY_RANGE = (0.35, 0.50)
# A larger attenuating mass casts a shadow over the beams it blocks.
ATTENUATING_AXIS_RANGE_M = (0.018, 0.035)
ATTENUATING_ECHOGENICITY_RANGE = (-0.50, -0.30)
# A specular reflector returns echoes only from the surface facing the probe: a
# bright crescent, a dark body behind it, and a partial shadow beyond.
SPECULAR_AXIS_RANGE_M = (0.006, 0.016)
SPECULAR_ECHOGENICITY_RANGE = (0.10, 0.25)
SPECULAR_RIM_FRACTION = 0.35
SPECULAR_SHADOW_FRACTION = 0.5
# The artifact of a lesion the frame only clips is proportionally weaker, and a
# footprint this thin is dropped outright.
LESION_MIN_VISIBILITY = 0.02
LESION_BAND_HALF_WIDTH_FLOOR = 0.02
# Lesions sit closer to the middle of the phantom than background tissue does, so
# the sweep cuts them often enough to matter.
LESION_EXTENT_FRACTION = 0.6


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars and arrays."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


# ---------------------------------------------------------------------------
# SE(3)
# ---------------------------------------------------------------------------


def pose_matrix_to_12(pose: np.ndarray) -> list[float]:
    """Convert a 4x4 pose matrix to the row-major 12D pose representation."""
    return pose[:3, :3].reshape(-1).tolist() + pose[:3, 3].tolist()


def pose_12_to_matrix(pose_12: np.ndarray) -> np.ndarray:
    """Convert a 12D pose representation to a 4x4 pose matrix."""
    pose_12 = np.asarray(pose_12, dtype=np.float64)
    matrix = np.eye(4)
    matrix[:3, :3] = pose_12[:9].reshape(3, 3)
    matrix[:3, 3] = pose_12[9:12]
    return matrix


def keyframe_times(frame_times_seconds: np.ndarray, interval_seconds: float) -> np.ndarray:
    """Sparse knot times covering a frame sequence (one knot for a single frame)."""
    if interval_seconds <= 0:
        raise ValueError("keyframe intervals must be positive")
    times = np.asarray(frame_times_seconds, dtype=np.float64)
    duration = float(times[-1] - times[0]) if times.size > 1 else 0.0
    return times[0] + np.arange(0.0, duration + interval_seconds, interval_seconds)


def spline_to_frames(
    knot_times: np.ndarray, knot_values: np.ndarray, frame_times_seconds: np.ndarray
) -> np.ndarray:
    """Cubic-interpolate keyframe values onto frame times, constant if there is one knot."""
    if len(knot_times) < 2:
        return np.repeat(knot_values[:1], len(frame_times_seconds), axis=0)
    spline = CubicSpline(knot_times, knot_values, axis=0, bc_type="natural")
    return spline(frame_times_seconds)


# ---------------------------------------------------------------------------
# Probe trajectory
# ---------------------------------------------------------------------------


def random_unit_vectors(rng: np.random.Generator, count: int) -> np.ndarray:
    """Sample `count` directions uniformly on the unit sphere."""
    directions = rng.normal(size=(count, 3))
    norms = np.linalg.norm(directions, axis=-1, keepdims=True)
    directions = np.where(norms < 1e-9, np.array([1.0, 0.0, 0.0]), directions / norms)
    return directions


def look_at_rotation(position: np.ndarray, target: np.ndarray, roll_rad: float) -> np.ndarray:
    """Probe orientation whose local X axis (the beam axis) points at the target."""
    forward = np.asarray(target, dtype=np.float64) - np.asarray(position, dtype=np.float64)
    norm = np.linalg.norm(forward)
    if norm < 1e-9:
        raise ValueError("Probe position must not coincide with the look-at target")
    forward = forward / norm

    world_up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(forward, world_up))) > 0.95:
        world_up = np.array([0.0, 1.0, 0.0])
    lateral = np.cross(world_up, forward)
    lateral = lateral / np.linalg.norm(lateral)
    elevation = np.cross(forward, lateral)

    base = np.column_stack([forward, lateral, elevation])
    return Rotation.from_rotvec(roll_rad * forward).as_matrix() @ base


def generate_trajectory(
    frame_times_seconds: np.ndarray,
    config: TrajectoryConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Smooth `[N, 4, 4]` probe poses with speed-limited knot steps.

    Interpolation can overshoot these limits; measure_speeds reports realized peaks.

    Keyframe positions random-walk on a sphere of radius `standoff_m` and aim at a
    drifting target near the origin with a random roll. Each keyframe step is
    capped by the configured speeds, so zero speeds produce a stationary probe.
    """
    knot_times = keyframe_times(frame_times_seconds, config.keyframe_interval_seconds)
    steps = np.diff(knot_times, prepend=knot_times[0])

    positions = np.empty((len(knot_times), 3))
    rotations = np.empty((len(knot_times), 3, 3))
    directions = random_unit_vectors(rng, len(knot_times))
    # Targets uniform in a ball of `look_at_radius_m`, cube-root for uniform density.
    targets = (
        config.look_at_radius_m
        * rng.random(len(knot_times))[:, None] ** (1.0 / 3.0)
        * random_unit_vectors(rng, len(knot_times))
    )
    rolls = rng.uniform(-math.pi, math.pi, size=len(knot_times))

    positions[0] = config.standoff_m * directions[0]
    rotations[0] = look_at_rotation(positions[0], targets[0], rolls[0])
    for index in range(1, len(knot_times)):
        step = positions[index - 1] + config.translation_speed_mps * steps[index] * directions[index]
        norm = np.linalg.norm(step)
        positions[index] = (
            positions[index - 1] if norm < 1e-9 else step / norm * config.standoff_m
        )

        desired = look_at_rotation(positions[index], targets[index], rolls[index])
        # One cap over all three axes: aim and roll share the rotation budget.
        budget = math.radians(config.rotation_speed_dps) * steps[index]
        relative = Rotation.from_matrix(rotations[index - 1].T @ desired).as_rotvec()
        angle = float(np.linalg.norm(relative))
        if angle > budget:
            relative = relative * (budget / angle)
            desired = rotations[index - 1] @ Rotation.from_rotvec(relative).as_matrix()
        rotations[index] = desired

    if len(knot_times) < 2:
        frame_positions = np.repeat(positions[:1], len(frame_times_seconds), axis=0)
        frame_rotations = np.repeat(rotations[:1], len(frame_times_seconds), axis=0)
    else:
        frame_positions = spline_to_frames(knot_times, positions, frame_times_seconds)
        frame_rotations = RotationSpline(knot_times, Rotation.from_matrix(rotations))(
            frame_times_seconds
        ).as_matrix()

    poses = np.tile(np.eye(4), (len(frame_times_seconds), 1, 1))
    poses[:, :3, :3] = frame_rotations
    poses[:, :3, 3] = frame_positions
    return poses


def measure_speeds(poses: np.ndarray, frame_times_seconds: np.ndarray) -> tuple[float, float]:
    """Peak finite-difference translation (m/s) and rotation (deg/s) of a trajectory."""
    if len(poses) < 2:
        return 0.0, 0.0
    intervals = np.diff(frame_times_seconds)
    translation = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=-1) / intervals
    rotations = Rotation.from_matrix(poses[:, :3, :3])
    rotation = np.degrees((rotations[:-1].inv() * rotations[1:]).magnitude()) / intervals
    return float(translation.max()), float(rotation.max())


def load_invivo_trajectory(directory: str, max_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Frame times (s) and calibrated transducer poses from a real recording."""
    # Frames flagged as tracking dropouts carry no pose; drop them like any other
    # poseless frame. No temporal offset is applied: these poses define the truth.
    records = [
        record
        for record in load_sequence(directory, calibration=DEFAULT_CALIBRATION)
        if record.valid and record.transducer_pose is not None
    ]
    if max_frames > 0:
        records = records[:max_frames]
    if not records:
        raise ValueError(f"No valid frames found in {directory}")
    poses = np.stack([pose_12_to_matrix(record.transducer_pose) for record in records])
    timestamps = np.array([record.timestamp for record in records], dtype=np.int64)
    return (timestamps - timestamps[0]).astype(np.float64) / 1e9, poses


# ---------------------------------------------------------------------------
# Phantom
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Phantom:
    """Ellipsoids whose center, axes and echogenicity oscillate with cardiac phase.

    Fields are stacked over ellipsoids so the whole phantom evaluates at once.
    Lesions occupy the leading entries and are marked by `kinds`; they are anatomy
    like any other ellipsoid, and only the appearance model reads `kinds`.
    """

    centers: np.ndarray  # (K, 3) rest position
    center_amplitudes: np.ndarray  # (K, 3)
    axes: np.ndarray  # (K, 3) rest semi-axes
    axis_amplitudes: np.ndarray  # (K, 3) as a fraction of the rest axes
    echogenicities: np.ndarray  # (K,) signed, relative to the background
    echogenicity_amplitudes: np.ndarray  # (K,)
    rotations: np.ndarray  # (K, 3, 3) fixed orientation
    phase_offsets: np.ndarray  # (K,)
    kinds: np.ndarray  # (K,) "tissue", "shell", "smooth" or a member of LESION_KINDS
    background: float
    # (K,) wall thickness in metres; zero (or absent, for older recordings) is solid.
    shell_thicknesses: Optional[np.ndarray] = None

    @property
    def lesion_indices(self) -> np.ndarray:
        """Indices of the ellipsoids that cause an artifact."""
        return np.flatnonzero(np.isin(self.kinds, LESION_KINDS))

    def state(self, phase_rad: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Centers, semi-axes and echogenicities at one cardiac phase."""
        phase = phase_rad + self.phase_offsets[:, None]
        centers = self.centers + self.center_amplitudes * np.sin(phase + CENTER_PHASES)
        axes = self.axes * (1.0 + self.axis_amplitudes * np.sin(phase + AXIS_PHASES))
        echogenicities = self.echogenicities + self.echogenicity_amplitudes * np.sin(
            self.phase_offsets + phase_rad + ECHOGENICITY_PHASE
        )
        return centers, np.maximum(axes, 1e-4), echogenicities

    def evaluate(self, points: np.ndarray, phase_rad: float) -> np.ndarray:
        """Intensity in `[0, 1]` at `[M, 3]` world points, for one cardiac phase."""
        centers, axes, echogenicities = self.state(phase_rad)
        thicknesses = (np.zeros(len(centers)) if self.shell_thicknesses is None
                       else self.shell_thicknesses)
        intensity = np.full(len(points), self.background, dtype=np.float32)
        for center, axis, echogenicity, rotation, thickness in zip(
            centers, axes, echogenicities, self.rotations, thicknesses
        ):
            local = (points - center) @ rotation
            inside = np.sum((local / axis) ** 2, axis=-1) <= 1.0
            if thickness > 0:
                # The wall keeps its thickness as the ellipsoid pulses.
                inner = np.maximum(axis - thickness, 1e-4)
                inside &= np.sum((local / inner) ** 2, axis=-1) > 1.0
            intensity[inside] += echogenicity
        return np.clip(intensity, 0.0, 1.0)

    def texture_weight(self, points: np.ndarray, phase_rad: float) -> np.ndarray:
        """1 where speckle and noise apply, 0 inside texture-only ("smooth") ellipsoids."""
        weight = np.ones(len(points), dtype=np.float32)
        smooth = np.flatnonzero(self.kinds == "smooth")
        if smooth.size:
            centers, axes, _ = self.state(phase_rad)
            for index in smooth:
                local = (points - centers[index]) @ self.rotations[index]
                weight[np.sum((local / axes[index]) ** 2, axis=-1) <= 1.0] = 0.0
        return weight

    def bounding_box(self, padding_frac: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        """World box enclosing the phantom over the whole cycle, inflated by `padding_frac`.

        Reach uses the largest peak semi-axis plus the largest center excursion, so
        it stays valid under any fixed ellipsoid orientation.
        """
        reach = (
            np.max(self.axes * (1.0 + self.axis_amplitudes), axis=-1)
            + np.max(self.center_amplitudes, axis=-1)
        )[:, None]
        low = np.min(self.centers - reach, axis=0)
        high = np.max(self.centers + reach, axis=0)
        center, half = 0.5 * (low + high), 0.5 * (high - low) * (1.0 + padding_frac)
        return center - half, center + half


def create_phantom(config: PhantomConfig, rng: np.random.Generator, offset: np.ndarray) -> Phantom:
    """Independent group streams: removing lesions/shells preserves other anatomy."""
    seed = child_seed(rng)
    counts = ("num_enhancing_lesions", "num_attenuating_lesions", "num_specular_lesions",
              "num_ellipsoids", "num_shells", "num_smooth")
    if any(getattr(config, key) < 0 for key in counts):
        raise ValueError("ellipsoid and lesion counts must be non-negative")
    groups = []
    for key in counts:
        count = getattr(config, key)
        if count:
            group_config = replace(config, **{name: count if name == key else 0 for name in counts})
            groups.append(_create_phantom_group(group_config, named_rng(seed, key), offset))
    if not groups:
        raise ValueError("the phantom needs at least one ellipsoid")
    return Phantom(**{
        f.name: config.background if f.name == "background" else
        np.concatenate([getattr(group, f.name) for group in groups])
        for f in fields(Phantom)
    })


def _create_phantom_group(config: PhantomConfig, rng: np.random.Generator, offset: np.ndarray) -> Phantom:
    """Sample a phantom centered on `offset`, with phases spread around the cycle.

    Lesions are stacked first, in `LESION_KINDS` order, so their indices are stable
    and readable in the metadata.
    """
    groups = [
        ("enhancing", config.num_enhancing_lesions, ENHANCING_AXIS_RANGE_M, ENHANCING_ECHOGENICITY_RANGE),
        ("attenuating", config.num_attenuating_lesions, ATTENUATING_AXIS_RANGE_M, ATTENUATING_ECHOGENICITY_RANGE),
        ("specular", config.num_specular_lesions, SPECULAR_AXIS_RANGE_M, SPECULAR_ECHOGENICITY_RANGE),
        ("tissue", config.num_ellipsoids, config.axis_range_m, config.echogenicity_range),
        # Last, so adding shells leaves lesion and tissue indices unchanged.
        ("shell", config.num_shells, config.shell_axis_range_m, config.shell_echogenicity_range),
        ("smooth", config.num_smooth, config.smooth_axis_range_m, (0.0, 0.0)),
    ]
    if any(count < 0 for _, count, _, _ in groups):
        raise ValueError("ellipsoid and lesion counts must be non-negative")
    count = sum(group_count for _, group_count, _, _ in groups)
    if count <= 0:
        raise ValueError("the phantom needs at least one ellipsoid")

    axes = np.concatenate(
        [rng.uniform(*axis_range, size=(group_count, 3)) for _, group_count, axis_range, _ in groups]
    )
    echogenicities = np.concatenate(
        [rng.uniform(*echo_range, size=group_count) for _, group_count, _, echo_range in groups]
    )
    kinds = np.array([kind for kind, group_count, _, _ in groups for _ in range(group_count)])
    extents = np.where(np.isin(kinds, ("tissue", "shell", "smooth")), config.extent_m,
                       config.extent_m * LESION_EXTENT_FRACTION)
    centers = rng.uniform(-1.0, 1.0, size=(count, 3)) * extents[:, None] + offset

    # Stratified phases with jitter, so no two ellipsoids beat together and the
    # cycle is evenly covered.
    bin_width = 2.0 * math.pi / count
    phase_offsets = (np.arange(count) + 0.5) * bin_width + rng.uniform(
        -0.2 * bin_width, 0.2 * bin_width, size=count
    )
    rng.shuffle(phase_offsets)

    # Bound the pulsation so signed echogenicity stays inside [-0.98, 0.98].
    headroom = np.maximum(0.98 - np.abs(echogenicities), 0.01)
    amplitude_low = np.minimum(config.echogenicity_amplitude_range[0], headroom)
    amplitude_high = np.minimum(config.echogenicity_amplitude_range[1], headroom)
    center_amplitudes = rng.uniform(*config.center_amplitude_range_m, size=(count, 3))
    axis_amplitudes = rng.uniform(*config.axis_amplitude_range_frac, size=(count, 3))
    echogenicity_amplitudes = rng.uniform(amplitude_low, amplitude_high)
    # Texture-only ellipsoids never differ in intensity, not even as they pulse.
    echogenicity_amplitudes = np.where(kinds == "smooth", 0.0, echogenicity_amplitudes)
    rotations = Rotation.from_euler(
        "xyz", rng.uniform(-60.0, 60.0, size=(count, 3)), degrees=True
    ).as_matrix()
    # Thickness is only meaningful for the shell group.
    shell_thicknesses = np.zeros(count)
    if config.num_shells:
        shell_thicknesses[kinds == "shell"] = rng.uniform(
            *config.shell_thickness_range_m, size=config.num_shells
        )
    return Phantom(
        centers=centers,
        center_amplitudes=center_amplitudes,
        axes=axes,
        axis_amplitudes=axis_amplitudes,
        echogenicities=echogenicities,
        echogenicity_amplitudes=echogenicity_amplitudes,
        rotations=rotations,
        phase_offsets=phase_offsets,
        kinds=kinds,
        background=config.background,
        shell_thicknesses=shell_thicknesses,
    )


def generate_phase(
    frame_times_seconds: np.ndarray, config: PhantomConfig, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame cardiac frequency (Hz) and the wrapped phase integrated from it.

    Keyframe frequencies are drawn uniformly from `bpm_range` and splined. Spline
    overshoot is removed by contracting toward the range center rather than
    clipping, which would leave flat plateaus at the bounds.
    """
    low, high = (bpm / 60.0 for bpm in config.bpm_range)
    if not 0.0 <= low <= high:
        raise ValueError(f"bpm_range must satisfy 0 <= min <= max, got {config.bpm_range}")

    knot_times = keyframe_times(frame_times_seconds, config.bpm_keyframe_interval_seconds)
    knots = rng.uniform(low, high, size=(len(knot_times), 1))
    frequencies = spline_to_frames(knot_times, knots, frame_times_seconds)[:, 0]

    center, half_range = 0.5 * (low + high), 0.5 * (high - low)
    deviation = frequencies - center
    excess = float(np.abs(deviation).max())
    if excess > half_range:
        frequencies = center + deviation * (half_range / excess)

    cycles = cumulative_trapezoid(frequencies, frame_times_seconds, initial=0.0)
    phase = -math.pi + 2.0 * math.pi * cycles
    return frequencies.astype(np.float32), np.arctan2(np.sin(phase), np.cos(phase)).astype(np.float32)


# ---------------------------------------------------------------------------
# Sector geometry and appearance
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Sector:
    """Fixed sector geometry and the per-pixel maps derived from it."""

    image_size: tuple[int, int]
    fov_degrees: float
    apex_pixels: tuple[float, float]
    radius_pixels: float
    radius_m: float  # imaging depth: the physical distance a depth fraction of 1 is
    microns_per_pixel: float
    local_points: np.ndarray  # (H, W, 3) probe-frame position in meters
    depth: np.ndarray  # (H, W) radial distance from the apex, as a fraction of the radius
    angle: np.ndarray  # (H, W) beam angle in radians
    valid: np.ndarray  # (H, W) inside the sector
    beam_directions: np.ndarray  # (M, 3) unit vector along the beam, valid pixels only

    def scatter(self, values: np.ndarray) -> np.ndarray:
        """Place a per-valid-pixel array back into an image-shaped one."""
        image = np.zeros(self.image_size, dtype=values.dtype)
        image[self.valid] = values
        return image


def create_sector(image_size: tuple[int, int], fov_degrees: float, depth_m: float) -> Sector:
    """Build a centered sector filling the image width, imaging to `depth_m`."""
    height, width = image_size
    half_fov = math.radians(fov_degrees) / 2.0
    radius_pixels = width / (2.0 * math.sin(half_fov))
    apex_pixels = (round((height - radius_pixels) / 2.0), round(width / 2.0))
    microns_per_pixel = depth_m * 1e6 / radius_pixels

    rows = np.arange(height, dtype=np.float64)[:, None] + 0.5 - apex_pixels[0]
    cols = np.arange(width, dtype=np.float64)[None, :] + 0.5 - apex_pixels[1]
    rows, cols = np.broadcast_arrays(rows, cols)
    radius = np.hypot(rows, cols)
    angle = np.arctan2(cols, rows)

    # Local X is the beam axis and local Y is lateral, matching `ProbeProjector`.
    local = np.stack([rows, cols, np.zeros_like(rows)], axis=-1) * (microns_per_pixel * 1e-6)
    valid = (radius > 0) & (radius < radius_pixels) & (np.abs(angle) < half_fov)
    # Beams leave the apex, so the propagation direction at a pixel is its own
    # direction from the apex.
    beams = local[valid] / np.linalg.norm(local[valid], axis=-1, keepdims=True)
    return Sector(
        image_size=(height, width),
        fov_degrees=fov_degrees,
        apex_pixels=(float(apex_pixels[0]), float(apex_pixels[1])),
        radius_pixels=radius_pixels,
        radius_m=float(depth_m),
        microns_per_pixel=microns_per_pixel,
        local_points=local.astype(np.float32),
        depth=np.clip(radius / radius_pixels, 0.0, 1.0).astype(np.float32),
        angle=angle.astype(np.float32),
        valid=valid,
        beam_directions=beams.astype(np.float32),
    )


def world_points(sector: Sector, pose: np.ndarray, displacement: Optional[np.ndarray] = None) -> np.ndarray:
    """Map the sector's valid pixels into world space under one probe pose.

    `displacement` (valid pixels x 3, probe-local meters) warps where each pixel samples.
    """
    local = sector.local_points[sector.valid]
    if displacement is not None:
        local = local + displacement.astype(np.float32)
    return (local @ pose[:3, :3].T.astype(np.float32)) + pose[:3, 3].astype(np.float32)


def _smoothstep(values: np.ndarray, edge_low, edge_high) -> np.ndarray:
    """Hermite ramp from 0 below `edge_low` to 1 above `edge_high`.

    The edges may be per-pixel arrays, which is how a shadow onset follows the
    surface of the lesion casting it.
    """
    ramp = np.clip((values - edge_low) / np.maximum(edge_high - edge_low, 1e-6), 0.0, 1.0)
    return ramp * ramp * (3.0 - 2.0 * ramp)


def depth_gain_profile(sector: Sector, config: AppearanceConfig) -> np.ndarray:
    """The standing near/far brightness tilt an imperfect attenuation curve leaves.

    The only part of the appearance model not caused by a particular structure.
    """
    return np.clip(1.0 + config.depth_gain * (1.0 - 2.0 * sector.depth), 0.0, None)


@dataclass(frozen=True)
class LesionView:
    """A lesion as one frame cuts it, and the artifact that cut implies.

    A frame that clips a lesion near its edge sees a smaller cross-section and gets
    a proportionally weaker artifact, so the artifact appears and disappears with
    the geometry instead of switching on. `visibility` is the area fraction of the
    cut relative to the lesion's widest cross-section.
    """

    index: int
    kind: str
    visibility: float
    inside: np.ndarray  # (H, W) pixels within the lesion
    rim: np.ndarray  # (H, W) apex-facing surface shell, empty for other kinds
    # Per beam, from an exact ray-ellipsoid intersection: where the beam leaves the
    # lesion, and how far it travelled inside. Both are constant along a ray, so a
    # shadow drawn from them starts on the lesion's own curved back surface.
    exit_depth: np.ndarray  # (H, W) depth fraction, past the far edge where missed
    chord_fraction: np.ndarray  # (H, W) path length inside, over the widest chord
    depth_near: float
    depth_far: float
    depth_span: float
    angle_center: float
    angle_half_width: float


def project_lesions(
    sector: Sector,
    phantom: Phantom,
    pose: np.ndarray,
    phase_rad: float,
    points: np.ndarray,
) -> list[LesionView]:
    """Find where each lesion is cut by this frame's imaging plane.

    This is what makes an artifact anatomy-caused rather than free-floating: its
    depth, angle and strength are all read off the lesion's footprint in the frame,
    so returning to the same view reproduces the same artifact, and a view that
    misses the lesion sees nothing.
    """
    lesion_indices = phantom.lesion_indices
    if not len(lesion_indices):
        return []

    centers, axes, _ = phantom.state(phase_rad)
    rotation, translation = pose[:3, :3], pose[:3, 3]
    # The imaging plane is local z = 0, so its world normal is the pose's z axis.
    normal = rotation[:, 2]
    beams = sector.beam_directions @ rotation.T.astype(np.float32)

    views: list[LesionView] = []
    for index in lesion_indices:
        center, axis, lesion_rotation = centers[index], axes[index], phantom.rotations[index]
        # Support of the ellipsoid along the plane normal: how far it reaches out of
        # plane. The cut ellipse's area scales as 1 - (distance / support)^2.
        support = float(np.linalg.norm(axis * (normal @ lesion_rotation)))
        distance = abs(float((center - translation) @ normal))
        visibility = 1.0 - (distance / support) ** 2 if support > 0.0 else 0.0
        if visibility <= LESION_MIN_VISIBILITY:
            continue

        local = (points - center.astype(np.float32)) @ lesion_rotation.astype(np.float32)
        normalized = local / axis.astype(np.float32)
        inside = np.sum(normalized**2, axis=-1) <= 1.0
        if not inside.any():
            continue

        rim = np.zeros_like(inside)
        if phantom.kinds[index] == "specular":
            # Gradient of the ellipsoid function, pointing out of the surface. The
            # crescent is where that faces back up the beam.
            surface_normal = (normalized / axis.astype(np.float32)) @ lesion_rotation.T.astype(
                np.float32
            )
            facing = np.sum(surface_normal * beams, axis=-1) < 0.0
            rim = inside & facing & (np.sum(normalized**2, axis=-1) > 1.0 - SPECULAR_RIM_FRACTION)

        exit_depth, chord_fraction = _beam_intersection(
            sector, beams, translation, center, axis, lesion_rotation
        )
        depths, angles = sector.depth[sector.valid][inside], sector.angle[sector.valid][inside]
        views.append(
            LesionView(
                index=int(index),
                kind=str(phantom.kinds[index]),
                visibility=float(visibility),
                inside=sector.scatter(inside),
                rim=sector.scatter(rim),
                exit_depth=sector.scatter(exit_depth) + BEYOND_FAR_EDGE * ~sector.valid,
                chord_fraction=sector.scatter(chord_fraction),
                depth_near=float(depths.min()),
                depth_far=float(depths.max()),
                depth_span=float(depths.max() - depths.min()),
                angle_center=float(0.5 * (angles.min() + angles.max())),
                angle_half_width=float(0.5 * (angles.max() - angles.min())),
            )
        )
    return views


def _beam_intersection(
    sector: Sector,
    beams: np.ndarray,
    apex: np.ndarray,
    center: np.ndarray,
    axis: np.ndarray,
    rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Intersect every beam with one ellipsoid: exit depth and path length inside.

    Beams leave the apex, so a ray is `apex + t * beam`; in the ellipsoid's own
    scaled frame the surface is the unit sphere and `t` solves a quadratic. Doing
    this per beam rather than once per lesion is what keeps a shadow attached: the
    onset follows the curved back surface instead of sitting at the single deepest
    radius of the footprint.
    """
    scaled_beams = (beams @ rotation.astype(np.float32)) / axis.astype(np.float32)
    scaled_apex = ((apex - center) @ rotation).astype(np.float32) / axis.astype(np.float32)
    a = np.sum(scaled_beams**2, axis=-1)
    b = np.sum(scaled_beams * scaled_apex, axis=-1)
    c = float(np.sum(scaled_apex**2) - 1.0)

    discriminant = b**2 - a * c
    hit = discriminant > 0.0
    root = np.sqrt(np.maximum(discriminant, 0.0))
    far = (-b + root) / a
    near = (-b - root) / a

    # Only the part of the chord in front of the transducer attenuates.
    chord = np.where(hit, np.maximum(far - np.maximum(near, 0.0), 0.0), 0.0)
    widest_chord = 2.0 * float(np.max(axis))
    exit_depth = np.where(hit & (far > 0.0), far / sector.radius_m, BEYOND_FAR_EDGE)
    return exit_depth.astype(np.float32), (chord / widest_chord).astype(np.float32)


def gain_field(
    sector: Sector, config: AppearanceConfig, views: list[LesionView]
) -> np.ndarray:
    """Multiplicative gain over the sector: the depth tilt, then every lesion cut.

    Every term beyond the tilt is attributable to a particular ellipsoid, so any
    brightness structure a reconstruction cannot explain from the scene is either
    the tilt or noise.
    """
    gain = depth_gain_profile(sector, config)
    for view in views:
        if view.kind == "enhancing":
            # Rings at its own range: bright across every beam, centred on the
            # lesion rather than on its far edge, so the band runs through it.
            half_width = max(0.5 * view.depth_span, LESION_BAND_HALF_WIDTH_FLOOR)
            center = 0.5 * (view.depth_near + view.depth_far)
            profile = np.exp(-0.5 * ((sector.depth - center) / half_width) ** 2)
            gain = gain * (1.0 + config.enhancing_gain * view.visibility * profile)
        elif view.kind == "attenuating":
            gain = gain * _posterior_shadow(sector, view, config.shadow_strength)
        else:
            gain = np.where(view.rim, gain * (1.0 + config.specular_gain * view.visibility), gain)
            body = view.inside & ~view.rim
            gain = np.where(body, gain * (1.0 - config.specular_darkening * view.visibility), gain)
            gain = gain * _posterior_shadow(
                sector, view, config.shadow_strength * SPECULAR_SHADOW_FRACTION
            )
    return np.clip(gain, 0.0, None).astype(np.float32)


def _posterior_shadow(sector: Sector, view: LesionView, strength: float) -> np.ndarray:
    """Darkening behind a lesion, per beam, from where that beam leaves it.

    Beer-Lambert in the path length through the lesion, so `strength` is the signal
    removed behind the widest chord and beams that only graze it are barely
    darkened. That gives the penumbra at the shadow's edges for free, and needs no
    angular profile: the shadow covers exactly the beams the lesion blocks, and
    starts exactly where each of them leaves it.
    """
    if strength <= 0.0:
        return np.ones(sector.image_size, dtype=np.float32)
    transmission = (1.0 - strength) ** view.chord_fraction
    # The penumbra straddles the exit surface rather than starting at it: a ramp
    # that begins at the boundary leaves a lit sliver behind the lesion, which
    # reads as a detached shadow. Half strength at the surface closes it.
    beyond = _smoothstep(
        sector.depth,
        view.exit_depth - 0.5 * SHADOW_PENUMBRA_DEPTH,
        view.exit_depth + 0.5 * SHADOW_PENUMBRA_DEPTH,
    )
    return 1.0 - (1.0 - transmission) * beyond


class SpeckleField:
    """Frame-to-frame speckle on the sector's polar grid; see `AppearanceConfig`."""

    def __init__(self, sector: Sector, config: AppearanceConfig, rng: np.random.Generator):
        self.sector, self.config, self.rng = sector, config, rng
        half_fov = math.radians(sector.fov_degrees) / 2.0
        # Sample the grid at a third of each PSF width so the smoothing is resolved.
        self.range_step = config.speckle_axial_mm * 1e-3 / 3.0
        self.angle_step = math.radians(config.speckle_lateral_deg) / 3.0
        self.half_fov = half_fov
        self.shape = (int(math.ceil(sector.radius_m / self.range_step)) + 2,
                      int(math.ceil(2 * half_fov / self.angle_step)) + 2)
        self.sigma = (3.0, 3.0)
        # Unit variance per component after smoothing: scale by the kernel's L2 norm.
        impulse = np.zeros((41, 41))
        impulse[20, 20] = 1.0
        self.scale = 1.0 / np.sqrt(np.sum(gaussian_filter(impulse, self.sigma) ** 2))
        valid_depth = sector.depth[sector.valid] * sector.radius_m
        valid_angle = sector.angle[sector.valid]
        self.coords = np.stack([valid_depth / self.range_step, (valid_angle + half_fov) / self.angle_step])
        self.field = None
        self.previous = None

    def _innovation(self) -> np.ndarray:
        real, imag = (gaussian_filter(self.rng.normal(size=self.shape), self.sigma) * self.scale
                      for _ in range(2))
        return (real + 1j * imag) / np.sqrt(2.0)

    def correlation(self, pose: np.ndarray, phase_rad: float) -> float:
        """Correlation with the previous frame's field, from plane motion and phase change."""
        if self.previous is None:
            return 0.0
        prev_pose, prev_phase = self.previous
        depth = self.config.speckle_reference_depth_m
        local = np.array([[depth, 0.0, 0.0], [depth, depth / 2, 0.0], [depth, -depth / 2, 0.0]])
        moved = local @ pose[:3, :3].T + pose[:3, 3] - (local @ prev_pose[:3, :3].T + prev_pose[:3, 3])
        distance_mm = float(np.sqrt(np.mean(np.sum(moved ** 2, axis=1)))) * 1e3
        dphase = float(np.angle(np.exp(1j * (phase_rad - prev_phase))))
        return math.exp(-(distance_mm / self.config.speckle_decorrelation_mm) ** 2
                        - (dphase / self.config.speckle_decorrelation_phase) ** 2)

    def next(self, pose: np.ndarray, phase_rad: float) -> np.ndarray:
        """Mean-1 speckle envelope on the valid pixels for this frame."""
        rho = self.correlation(pose, phase_rad)
        innovation = self._innovation()
        self.field = innovation if self.field is None else rho * self.field + math.sqrt(1 - rho ** 2) * innovation
        self.previous = (pose, phase_rad)
        # For a unit-variance complex Gaussian Z, E[|Z|**c] = Gamma(1 + c/2).
        # Normalize AFTER applying contrast, before linear interpolation (which
        # preserves the mean). Powering the interpolated envelope would bias it again.
        contrast = self.config.speckle_contrast
        envelope = np.abs(self.field) ** contrast / math.gamma(1.0 + contrast / 2.0)
        values = map_coordinates(envelope, self.coords, order=1, mode="nearest")
        return values.astype(np.float32)


def interpolate_grid(grid: np.ndarray, origin_m: np.ndarray, spacing_m: float, points: np.ndarray) -> np.ndarray:
    """Bilinearly sample a (C, Gx, Gy) probe-local grid at in-plane points (M, >=2): (M, C)."""
    coords = ((np.asarray(points)[:, :2] - origin_m) / spacing_m).T
    return np.stack([map_coordinates(channel, coords, order=1, mode="nearest") for channel in grid], axis=-1)


class DeformationField:
    """Per-frame smooth warp of the probe-local sample positions; see `DeformationConfig`."""

    SIGMA_NODES = 3.0  # smoothing std in grid nodes, so the grid resolves the field

    def __init__(self, sector: Sector, config: DeformationConfig, rng: np.random.Generator):
        if config.rms_mm < 0.0 or config.elevational_rms_mm < 0.0:
            raise ValueError("deformation amplitudes must be non-negative")
        if config.correlation_length_mm <= 0.0 or config.correlation_seconds < 0.0:
            raise ValueError("deformation correlation length must be positive, time non-negative")
        self.config, self.rng = config, rng
        self.points = sector.local_points[sector.valid].astype(np.float64)
        self.spacing_m = config.correlation_length_mm * 1e-3 / self.SIGMA_NODES
        low = self.points[:, :2].min(axis=0) - self.spacing_m
        high = self.points[:, :2].max(axis=0) + self.spacing_m
        self.origin_m = low
        self.shape = tuple(int(math.ceil(extent / self.spacing_m)) + 1 for extent in high - low)
        # Noise is drawn on a padded grid so the smoothing has no edge effects.
        self.pad = int(math.ceil(3.0 * self.SIGMA_NODES))
        impulse = np.zeros((8 * self.pad + 1,) * 2)
        impulse[4 * self.pad, 4 * self.pad] = 1.0
        self.scale = 1.0 / np.sqrt(np.sum(gaussian_filter(impulse, self.SIGMA_NODES) ** 2))
        nodes = [low[axis] + self.spacing_m * np.arange(self.shape[axis]) for axis in range(2)]
        self.node_x, self.node_y = np.meshgrid(*nodes, indexing="ij")
        x, y = self.points[:, 0], self.points[:, 1]
        ones, zeros = np.ones_like(x), np.zeros_like(x)
        # Small rigid motion restricted to the plane z = 0: u = omega x p + v. In plane
        # (ux, uy) = (vx - wz y, vy + wz x); out of plane uz = vz + wx y - wy x.
        self.in_plane_basis = np.concatenate([
            np.stack([ones, zeros, -y], axis=1), np.stack([zeros, ones, x], axis=1)])
        self.elevational_basis = np.stack([ones, y, -x], axis=1)
        self.field: Optional[np.ndarray] = None
        self.previous_time: Optional[float] = None

    def _innovation(self) -> np.ndarray:
        padded = (self.shape[0] + 2 * self.pad, self.shape[1] + 2 * self.pad)
        crop = (slice(self.pad, self.pad + self.shape[0]), slice(self.pad, self.pad + self.shape[1]))
        return np.stack([gaussian_filter(self.rng.normal(size=padded), self.SIGMA_NODES)[crop] * self.scale
                         for _ in range(3)])

    def next(self, time_seconds: float) -> tuple[np.ndarray, np.ndarray, dict]:
        """This frame's per-valid-pixel displacement (M, 3) in meters, its grid in mm, and stats."""
        innovation = self._innovation()
        if self.field is None or self.config.correlation_seconds == 0.0:
            self.field = innovation
        else:
            rho = math.exp(-abs(time_seconds - self.previous_time) / self.config.correlation_seconds)
            self.field = rho * self.field + math.sqrt(1.0 - rho**2) * innovation
        self.previous_time = time_seconds

        grid = self.field.copy()
        values = interpolate_grid(grid, self.origin_m, self.spacing_m, self.points)
        rigid_in_plane = np.zeros(3)
        if self.config.remove_rigid:
            # Linear fields are exact under bilinear interpolation, so removing the fit
            # from the grid removes it from the pixels too.
            rigid_in_plane = np.linalg.lstsq(self.in_plane_basis, values[:, :2].T.reshape(-1), rcond=None)[0]
            rigid_elevational = np.linalg.lstsq(self.elevational_basis, values[:, 2], rcond=None)[0]
            vx, vy, wz = rigid_in_plane
            grid[0] -= vx - wz * self.node_y
            grid[1] -= vy + wz * self.node_x
            vz, wx, wy = rigid_elevational
            grid[2] -= vz + wx * self.node_y - wy * self.node_x
        values = interpolate_grid(grid, self.origin_m, self.spacing_m, self.points)
        in_plane_rms = float(np.sqrt(np.mean(np.sum(values[:, :2] ** 2, axis=1))))
        elevational_rms = float(np.sqrt(np.mean(values[:, 2] ** 2)))
        gains = np.array([self.config.rms_mm / max(in_plane_rms, 1e-12)] * 2
                         + [self.config.elevational_rms_mm / max(elevational_rms, 1e-12)]) * 1e-3
        grid_m = grid * gains[:, None, None]
        displacement = values * gains

        # Fold check on the in-plane map p -> p + u(p).
        dux_dx, dux_dy = np.gradient(grid_m[0], self.spacing_m)
        duy_dx, duy_dy = np.gradient(grid_m[1], self.spacing_m)
        min_jacobian = float(np.min((1.0 + dux_dx) * (1.0 + duy_dy) - dux_dy * duy_dx))
        if min_jacobian <= 0.0:
            raise ValueError(f"deformation folds (min Jacobian {min_jacobian:.3f}); "
                             "lower rms_mm or raise correlation_length_mm")
        rigid_mm = float(np.sqrt(np.mean(np.sum(
            (self.in_plane_basis @ rigid_in_plane).reshape(2, -1) ** 2, axis=0)))) * gains[0] * 1e3
        stats = {"rms_mm": float(np.sqrt(np.mean(np.sum(displacement[:, :2] ** 2, axis=1)))) * 1e3,
                 "rigid_removed_mm": rigid_mm, "min_jacobian": min_jacobian}
        return displacement.astype(np.float32), (grid_m * 1e3).astype(np.float32), stats


def render_frame(
    sector: Sector,
    phantom: Phantom,
    pose: np.ndarray,
    phase_rad: float,
    config: AppearanceConfig,
    rng: np.random.Generator,
    speckle: Optional[np.ndarray] = None,
    displacement: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, list[LesionView]]:
    """Render one B-mode frame: anatomy, optional speckle, then gain, then additive noise.

    `speckle` is a per-valid-pixel multiplier (from `SpeckleField.next`); without it the
    frame is speckle-free, which is what clean reference renders want. `displacement`
    (from `DeformationField.next`) warps where each pixel samples the anatomy, lesion
    footprints included.
    """
    points = world_points(sector, pose, displacement)
    image = np.zeros(sector.image_size, dtype=np.float32)
    image[sector.valid] = phantom.evaluate(points, phase_rad)
    texture = phantom.texture_weight(points, phase_rad)
    if speckle is not None:
        image[sector.valid] *= speckle ** texture

    views = project_lesions(sector, phantom, pose, phase_rad, points)
    image *= gain_field(sector, config, views)
    if config.noise_std > 0.0:
        image[sector.valid] += rng.normal(0.0, config.noise_std, size=int(sector.valid.sum())) * texture
    image[~sector.valid] = 0.0
    return np.clip(image, 0.0, 1.0), views


# ---------------------------------------------------------------------------
# Evaluation grid
# ---------------------------------------------------------------------------


def save_evaluation_grid(
    output_path: Path,
    phantom: Phantom,
    resolution: int,
    num_phases: int,
    padding_frac: float = 0.1,
) -> dict:
    """Save dense ground-truth anatomy over (xyz, phase) for reconstruction metrics.

    Anatomy is a pure function of cardiac phase, so the grid is parameterized by
    phase directly and is independent of the (drifting) instantaneous frequency.
    A learned phase only matches truth up to a constant offset, so consumers
    should search over constant offsets when scoring.
    """
    if resolution < 2:
        raise ValueError(f"eval_grid_resolution must be at least 2, got {resolution}")
    if num_phases < 1:
        raise ValueError(f"eval_grid_phases must be positive, got {num_phases}")

    low, high = phantom.bounding_box(padding_frac)
    axes = [np.linspace(low[dim], high[dim], resolution, dtype=np.float32) for dim in range(3)]
    grid = np.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
    coords_xyz = np.stack([grid[2], grid[1], grid[0]], axis=-1).reshape(-1, 3)

    phases = np.linspace(-math.pi, math.pi, num_phases, endpoint=False, dtype=np.float32)
    intensities = np.stack([phantom.evaluate(coords_xyz, float(phase)) for phase in phases])

    directory = output_path / "evaluation"
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        directory / "eval_grid.npz",
        coords_xyz=coords_xyz,
        phases_radians=phases,
        intensities=intensities,
    )
    metadata = {
        "description": "Dense (xyz, phase) grid of ground-truth phantom intensity.",
        "grid_file": "evaluation/eval_grid.npz",
        "coordinate_frame": "absolute_world",
        "num_points": int(len(coords_xyz)),
        "num_phases": int(num_phases),
        "resolution": int(resolution),
        "box_min_m": low.tolist(),
        "box_max_m": high.tolist(),
        "background": float(phantom.background),
        "intensity_range": [0.0, 1.0],
        "phase_offset_ambiguity": (
            "Learned phase matches truth only up to a constant offset; search over "
            "constant offsets when scoring predictions."
        ),
    }
    with (directory / "eval_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2, cls=NumpyEncoder)
    return metadata


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def save_trajectory_plot(
    output_path: Path,
    true_poses_12: np.ndarray,
    raw_poses_12: np.ndarray,
    frame_times_seconds: np.ndarray,
    frequencies_hz: np.ndarray,
    num_glyphs: int = 50,
) -> str:
    """Plot the true and raw pose paths beside the cardiac frequency trajectory."""
    directory = output_path / "visualizations"
    directory.mkdir(parents=True, exist_ok=True)

    true_poses_12 = np.asarray(true_poses_12, dtype=np.float32)
    raw_poses_12 = np.asarray(raw_poses_12, dtype=np.float32)
    glyphs = np.unique(
        np.round(np.linspace(0, len(true_poses_12) - 1, num=min(len(true_poses_12), num_glyphs)))
    ).astype(np.int64)

    figure = plt.figure(figsize=(15, 7))
    grid = figure.add_gridspec(1, 2, width_ratios=[1.5, 1.0])
    plot_pose_pair_comparison_3d(
        true_poses=true_poses_12[glyphs],
        corrected_poses=raw_poses_12[glyphs],
        true_trajectory_poses=true_poses_12,
        corrected_trajectory_poses=raw_poses_12,
        show_corrected_trajectory_line=False,
        bounds=compute_pose_plot_bounds(
            np.concatenate([true_poses_12, raw_poses_12], axis=0), padding=0.25
        ),
        ax=figure.add_subplot(grid[0], projection="3d"),
        title=f"Pose trajectory, true vs raw\nframes={len(true_poses_12)} glyphs={len(glyphs)}",
    )
    axis = figure.add_subplot(grid[1])
    axis.plot(frame_times_seconds, np.asarray(frequencies_hz) * 60.0, color="tab:purple", linewidth=2)
    axis.set_xlabel("time (s)")
    axis.set_ylabel("instantaneous frequency (bpm)")
    axis.set_title("Cardiac frequency trajectory")
    axis.grid(alpha=0.3)

    path = directory / "trajectory_overview.png"
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return str(path.relative_to(output_path))


def save_video(output_path: Path, images_dir: Path, frame_rate_hz: float) -> str | None:
    """Encode the rendered frames to MP4, or return None if ffmpeg is unavailable."""
    directory = output_path / "visualizations"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "dataset.mp4"
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        *("-y", "-hide_banner", "-loglevel", "error"),
        *("-framerate", f"{max(frame_rate_hz, 1e-3):.6g}"),
        *("-i", str(images_dir / "%05d.png")),
        *("-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"),
        *("-c:v", "libx264", "-pix_fmt", "yuv420p"),
        str(path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"Skipping MP4 generation: ffmpeg failed ({error})")
        return None
    return str(path.relative_to(output_path))


def write_metadata(
    output_path: Path,
    config: SyntheticConfig,
    sector: Sector,
    frame_times_seconds: np.ndarray,
    true_poses_12: np.ndarray,
    raw_poses_12: np.ndarray,
    frequencies_hz: np.ndarray,
    phases_rad: np.ndarray,
    num_train: int,
    raw_tracker_poses: np.ndarray,
    phantom: Phantom,
    pose_error: PoseError,
    extra: dict,
) -> None:
    """Write `sequence_metadata.json` (v5.0) with per-frame ground truth.

    `phases_rad` is the wrapped phase each image was rendered at, so evaluation scores
    against the phase that formed the image rather than re-integrating frequency.
    """
    frames = [
        FrameMetadata(
            image_path=f"images/{index:05d}.png",
            timestamp_ns=int(frame_times_seconds[index] * 1e9),
            microns_per_pixel=sector.microns_per_pixel,
            # Stored in raw tracker space so synthetic and in-vivo recordings load
            # through the same forward calibration.
            raw_tracker_pose=pose_matrix_to_12(raw_tracker_poses[index]),
            valid=True,
            # Appended perturbed test frames follow the training frames.
            split="test" if index >= num_train else "train",
            ground_truth_transducer_pose=list(true_poses_12[index]),
            ground_truth_frequency_hz=float(frequencies_hz[index]),
            ground_truth_phase_radians=float(phases_rad[index]),
        )
        for index in range(len(true_poses_12))
    ]
    mean_frequency = float(np.mean(frequencies_hz))
    metadata = SequenceMetadataV3(
        format_version=CURRENT_FORMAT_VERSION,
        sequence_type="synthetic",
        frames=frames,
        geometry={
            "default": GeometryConfig(
                image_size=list(sector.image_size),
                fov_degrees=sector.fov_degrees,
                sector_apex_pixels=list(sector.apex_pixels),
                sector_radius_pixels=sector.radius_pixels,
                probe_element="phased",
            )
        },
        calibration=CalibrationReference(
            name=DEFAULT_CALIBRATION.name, version=DEFAULT_CALIBRATION.version
        ),
        source=SourceMetadata(
            recording_time=datetime.now(timezone.utc).isoformat(),
            num_frames=len(frames),
            tracking_device="synthetic",
        ),
        time={"cardiac_period_seconds": 1.0 / mean_frequency if mean_frequency > 0.0 else 0.0},
        generation={
            "config": asdict(config),
            "true_poses": np.asarray(true_poses_12).tolist(),
            "raw_poses": np.asarray(raw_poses_12).tolist(),
            "true_frequency_hz": np.asarray(frequencies_hz).tolist(),
            "pose_error": pose_error.metadata(),
            "random_streams": {"version": "named_sha256_pcg64_v1", "seed": config.seed},
            "phantom": {
                "type": "pulsating_ellipsoids",
                **{
                    name: getattr(phantom, name).tolist()
                    for name in (
                        "centers",
                        "center_amplitudes",
                        "axes",
                        "axis_amplitudes",
                        "echogenicities",
                        "echogenicity_amplitudes",
                        "rotations",
                        "phase_offsets",
                        "kinds",
                        "shell_thicknesses",
                    )
                },
                "lesion_indices": phantom.lesion_indices.tolist(),
            },
            **extra,
        },
    )
    with (output_path / "sequence_metadata.json").open("w") as handle:
        json.dump(metadata.to_dict(), handle, indent=2, cls=NumpyEncoder)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate(config: SyntheticConfig) -> None:
    """Render a synthetic recording and write it to `config.output_dir`."""
    config.pose_error.validate()
    config.perturbed_test.validate()
    if config.invivo_trajectory_dir is None and (config.num_frames < 1 or not np.isfinite(config.trajectory.frame_rate_hz) or config.trajectory.frame_rate_hz <= 0):
        raise ValueError("generated trajectories need positive num_frames and frame_rate_hz")
    output_path = Path(config.output_dir)
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Names, not spawn order, define independent streams. Optional components cannot
    # shift the RNG state of another subsystem.
    trajectory_rng = named_rng(config.seed, "trajectory")
    phantom_rng = named_rng(config.seed, "phantom")
    phase_rng = named_rng(config.seed, "phase")
    noise_rng = named_rng(config.seed, "image_noise")
    speckle_rng = named_rng(config.seed, "speckle")
    deformation_rng = named_rng(config.seed, "deformation")

    sector = create_sector(config.image_size, config.fov_degrees, config.depth_m)
    print(
        f"Sector: {sector.image_size} px, fov={config.fov_degrees}deg, "
        f"depth={config.depth_m}m, {sector.microns_per_pixel:.1f} um/px, "
        f"{int(sector.valid.sum())} valid pixels"
    )

    if config.invivo_trajectory_dir:
        frame_times, true_poses = load_invivo_trajectory(
            config.invivo_trajectory_dir, config.num_frames
        )
        print(
            f"Loaded {len(true_poses)} poses from {config.invivo_trajectory_dir} "
            f"({frame_times[-1]:.1f}s)"
        )
    else:
        frame_times = np.arange(config.num_frames, dtype=np.float64) / config.trajectory.frame_rate_hz
        true_poses = generate_trajectory(frame_times, config.trajectory, trajectory_rng)
    validate_trajectory(true_poses, frame_times)
    num_train = len(true_poses)
    translation_speed, rotation_speed = measure_speeds(true_poses, frame_times)
    print(
        f"Trajectory: {len(true_poses)} poses, peak {translation_speed:.4f} m/s "
        f"and {rotation_speed:.2f} deg/s"
    )

    # Place the phantom where the probe looks: the beam axis is local X.
    mid = len(true_poses) // 2
    look_center = true_poses[:, :3, 3].mean(axis=0) + true_poses[mid, :3, 0] * (config.depth_m * 0.5)
    phantom = create_phantom(config.phantom, phantom_rng, look_center.astype(np.float32))
    frequencies_hz, phases = generate_phase(frame_times, config.phantom, phase_rng)
    print(
        f"Phantom: {config.phantom.num_ellipsoids} ellipsoids at "
        f"{np.round(look_center, 4).tolist()}, "
        f"{frequencies_hz.min() * 60.0:.1f}-{frequencies_hz.max() * 60.0:.1f} bpm"
    )

    true_tracker = np.stack([
        pose_12_to_matrix(compute_tracker_from_transducer(np.array(pose_matrix_to_12(pose)),
                                                        calibration=DEFAULT_CALIBRATION,
                                                        probe_element="phased"))
        for pose in true_poses
    ])
    pose_error = generate_pose_error(true_tracker, frame_times, config.pose_error, config.seed)
    raw_transducer = np.stack([
        compute_transducer_from_tracker(np.array(pose_matrix_to_12(pose)),
                                       calibration=DEFAULT_CALIBRATION, probe_element="phased")
        for pose in pose_error.raw_poses
    ])
    error_diagnostics = save_diagnostics(output_path, frame_times, pose_error)
    raw_tracker_poses = pose_error.raw_poses

    # Appended test frames: built from training frames, after everything above is fixed,
    # so the training recording is identical with or without them.
    test_config = config.perturbed_test
    test_metadata = {}
    if test_config.count:
        pivot = image_center(sector)
        base, test_poses = perturbed_test_frames(
            test_config, true_poses, named_rng(config.seed, "perturbed_test"), pivot
        )
        test_tracker = np.stack([
            pose_12_to_matrix(compute_tracker_from_transducer(np.array(pose_matrix_to_12(pose)),
                                                            calibration=DEFAULT_CALIBRATION,
                                                            probe_element="phased"))
            for pose in test_poses
        ])
        # The base frame's tracking error, as a world transform.
        errors = raw_tracker_poses[base] @ np.linalg.inv(true_tracker[base])
        test_raw_tracker = errors @ test_tracker
        test_phases = named_rng(config.seed, "perturbed_test/phase").uniform(
            -math.pi, math.pi, size=test_config.count
        ).astype(phases.dtype)
        test_times = frame_times[-1] + TEST_FRAME_INTERVAL_SECONDS * np.arange(1, test_config.count + 1)
        frame_times = np.concatenate([frame_times, test_times])
        true_poses = np.concatenate([true_poses, test_poses])
        raw_tracker_poses = np.concatenate([raw_tracker_poses, test_raw_tracker])
        raw_transducer = np.concatenate([raw_transducer, np.stack([
            compute_transducer_from_tracker(np.array(pose_matrix_to_12(pose)),
                                           calibration=DEFAULT_CALIBRATION, probe_element="phased")
            for pose in test_raw_tracker
        ])])
        phases = np.concatenate([phases, test_phases])
        frequencies_hz = np.concatenate([frequencies_hz, frequencies_hz[base]])
        test_metadata = {"perturbed_test": {
            "config": asdict(test_config),
            "pivot_probe_local_m": pivot.tolist(),
            "base_frames": base.tolist(),
            "raw_pose": "base frame's tracking error raw_b @ inv(true_b), applied on the left",
        }}
        print(
            f"Perturbed test frames: {test_config.count} at {test_config.rotation_deg} deg / "
            f"{test_config.translation_mm} mm from training frames {base.tolist()}"
        )
    error_stats = error_diagnostics["statistics"]
    print(f"Tracker error RMS: {error_stats['translation']['rms']:.3f} mm / "
          f"{error_stats['rotation']['rms']:.3f} degrees")

    kinds, counts = np.unique(phantom.kinds[phantom.lesion_indices], return_counts=True)
    print(f"Artifact-casting lesions: {dict(zip(kinds.tolist(), counts.tolist())) or 'none'}")

    print(f"Rendering {len(true_poses)} frames...")
    lesion_frames: list[list[dict]] = []
    # Speckle follows the true probe motion and cardiac phase, as a physical pattern would.
    speckle = SpeckleField(sector, config.appearance, speckle_rng) if config.appearance.speckle_contrast > 0 else None
    warp_config = config.deformation
    deformation = (DeformationField(sector, warp_config, deformation_rng)
                   if warp_config.rms_mm > 0 or warp_config.elevational_rms_mm > 0 else None)
    warp_grids, warp_stats = [], []
    for index, pose in enumerate(true_poses):
        displacement = None
        if deformation is not None:
            displacement, grid_mm, stats = deformation.next(float(frame_times[index]))
            warp_grids.append(grid_mm.astype(np.float16))
            warp_stats.append(stats)
        image, views = render_frame(
            sector=sector,
            phantom=phantom,
            pose=pose,
            phase_rad=float(phases[index]),
            config=config.appearance,
            rng=noise_rng,
            speckle=None if speckle is None else speckle.next(pose, float(phases[index])),
            displacement=displacement,
        )
        lesion_frames.append(
            [
                # The footprint too, so a diagnostic can ask whether error
                # concentrates where an artifact was without re-deriving it.
                {
                    "index": view.index,
                    "kind": view.kind,
                    "visibility": round(view.visibility, 4),
                    "depth_range": [round(view.depth_near, 4), round(view.depth_far, 4)],
                    "angle_center_deg": round(math.degrees(view.angle_center), 2),
                    "angle_half_width_deg": round(math.degrees(view.angle_half_width), 2),
                }
                for view in views
            ]
        )
        Image.fromarray((image * 255).astype(np.uint8), mode="L").save(
            images_dir / f"{index:05d}.png"
        )
        if (index + 1) % 100 == 0 or index == len(true_poses) - 1:
            print(f"  {index + 1}/{len(true_poses)}")
    visible = [len(frame) for frame in lesion_frames]
    if phantom.lesion_indices.size:
        print(
            f"Lesion artifacts: {np.mean(visible):.1f} visible per frame, "
            f"{np.mean(np.asarray(visible) > 0) * 100:.0f}% of frames affected"
        )

    deformation_metadata = {}
    if deformation is not None:
        # Per-frame fields in mm on the probe-local grid (float16); `interpolate_grid`
        # with this origin and spacing gives any pixel's displacement.
        np.savez_compressed(
            output_path / "deformation.npz",
            displacement_mm=np.stack(warp_grids),
            origin_m=deformation.origin_m,
            spacing_m=deformation.spacing_m,
        )
        print(
            "Deformation: "
            f"{np.mean([s['rms_mm'] for s in warp_stats]):.2f} mm RMS in plane, rigid part removed "
            f"{np.mean([s['rigid_removed_mm'] for s in warp_stats]):.2f} mm, min Jacobian "
            f"{min(s['min_jacobian'] for s in warp_stats):.3f}"
        )
        deformation_metadata = {"deformation": {
            "composition": "pixel p (probe-local) images true_pose @ (p + u_t(p))",
            "file": "deformation.npz",
            "grid_axes": "displacement_mm[frame, (axial, lateral, elevational), x_node, y_node]",
            **{key: [round(s[key], 5) for s in warp_stats]
               for key in ("rms_mm", "rigid_removed_mm", "min_jacobian")},
        }}

    true_poses_12 = np.array([pose_matrix_to_12(pose) for pose in true_poses])
    raw_poses_12 = raw_transducer

    frame_rate_hz = (
        (len(frame_times) - 1) / frame_times[-1] if len(frame_times) > 1 and frame_times[-1] > 0 else 1.0
    )
    visualizations = {
        "pose_error_plot": error_diagnostics["plot"],
        "trajectory_plot": save_trajectory_plot(
            output_path, true_poses_12, raw_poses_12, frame_times, frequencies_hz
        ),
        "dataset_video": save_video(output_path, images_dir, frame_rate_hz),
    }
    evaluation_grid = None
    if not config.skip_eval_grid:
        evaluation_grid = save_evaluation_grid(
            output_path, phantom, config.eval_grid_resolution, config.eval_grid_phases
        )
        print(
            f"Evaluation grid: {evaluation_grid['num_points']} points x "
            f"{evaluation_grid['num_phases']} phases"
        )

    write_metadata(
        output_path=output_path,
        config=config,
        sector=sector,
        frame_times_seconds=frame_times,
        true_poses_12=true_poses_12,
        raw_poses_12=raw_poses_12,
        frequencies_hz=frequencies_hz,
        phases_rad=phases,
        num_train=num_train,
        raw_tracker_poses=raw_tracker_poses,
        phantom=phantom,
        pose_error=pose_error,
        extra={
            "pose_error_diagnostics": error_diagnostics["statistics"],
            "lesion_frames": lesion_frames,
            "trajectory_source": config.invivo_trajectory_dir or "generated",
            "measured_translation_speed_mps": translation_speed,
            "measured_rotation_speed_dps": rotation_speed,
            "evaluation_grid": evaluation_grid,
            "visualizations": visualizations,
            **deformation_metadata,
            **test_metadata,
        },
    )
    print(f"\nWrote {len(true_poses)} frames and sequence_metadata.json to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=str, help="YAML file of the same fields")
    add_config_args(parser, SyntheticConfig)
    parser.add_argument("--no-pose-error", dest="pose_error__enabled", action="store_false",
                        default=None, help="Disable all injected tracking error, including lag")
    args = parser.parse_args()
    config = (
        load_yaml_config(SyntheticConfig, args.config) if args.config else SyntheticConfig()
    )
    generate(apply_cli_overrides(config, args))


if __name__ == "__main__":
    main()
