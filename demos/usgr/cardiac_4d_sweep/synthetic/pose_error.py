"""Jump-free tracker corruption, independent of the reconstruction correction model.

Translations are displacements of the tracker origin, NOT SE(3) exponential
translation coordinates. Rotation vectors use radians internally; configuration
amplitudes specify ensemble RMS vector norms (not coordinate standard deviations).
See tracking_error.md alongside this module for assumptions and empirical targets.
"""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .random_streams import named_rng


@dataclass(frozen=True)
class PoseErrorConfig:
    enabled: bool = True
    # A convention, not an experimentally identified source of error.
    frame: str = "world"  # world or body; rotations are about the tracker origin
    bias_translation_rms_mm: float = 2.0
    bias_rotation_rms_deg: float = 0.5
    # Smooth bias evolves slowly at rest, and as the tracker traverses its path.
    # These three scales are modeling assumptions, not fitted correlation times.
    bias_time_seconds: float = 120.0
    bias_distance_m: float = 0.15  # 0 disables this contribution
    bias_angle_deg: float = 45.0  # 0 disables this contribution
    colored_translation_rms_mm: float = 0.15
    colored_rotation_rms_deg: float = 0.018
    translation_correlation_seconds: float = 0.08
    rotation_correlation_seconds: float = 0.20
    white_translation_rms_mm: float = 0.05
    white_rotation_rms_deg: float = 0.005
    lag_seconds: float = 0.0  # optional timing stress test, unmeasured by the fixture

    def validate(self):
        if self.frame not in ("world", "body"):
            raise ValueError("pose_error.frame must be 'world' or 'body'")
        for key, value in asdict(self).items():
            if key in ("enabled", "frame"):
                continue
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"pose_error.{key} must be finite and non-negative")
        if self.bias_time_seconds <= 0:
            raise ValueError("pose_error.bias_time_seconds must be positive")


def validate_trajectory(poses, times):
    poses, times = np.asarray(poses, dtype=float), np.asarray(times, dtype=float)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) == 0:
        raise ValueError("poses must have nonempty shape (N, 4, 4)")
    if times.shape != (len(poses),) or not np.isfinite(times).all():
        raise ValueError("timestamps must be finite and match poses")
    if np.any(np.diff(times) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    r = poses[:, :3, :3]
    if (
        not np.isfinite(poses).all()
        or not np.allclose(poses[:, 3], [0, 0, 0, 1])
        or not np.allclose(r @ r.swapaxes(1, 2), np.eye(3), atol=1e-5)
        or not np.allclose(np.linalg.det(r), 1, atol=1e-5)
    ):
        raise ValueError("poses must be finite rigid transforms")
    return poses, times


def correlated_noise(times, correlation_seconds, rng):
    """Stationary unit-variance OU samples; exact transition at each actual dt."""
    white = rng.normal(size=(len(times), 3))
    if correlation_seconds == 0:
        return white
    result = white.copy()
    for i, dt in enumerate(np.diff(times), 1):
        rho = np.exp(-dt / correlation_seconds)
        result[i] = (
            rho * result[i - 1]
            + np.sqrt(-np.expm1(-2 * dt / correlation_seconds)) * white[i]
        )
    return result


def smooth_bias(clock, rng):
    """Stationary Matérn-3/2 process with a continuous first derivative.

    In the effective clock, covariance is (1 + sqrt(3)*d) exp(-sqrt(3)*d).
    Exact two-state transitions avoid spline knots shared with the estimator.
    """
    innovations = rng.normal(size=(len(clock), 2, 3))
    state = innovations[0].copy()
    result = np.empty((len(clock), 3))
    result[0] = state[0]
    for i, dt in enumerate(np.diff(clock), 1):
        h = np.sqrt(3) * dt
        a = np.exp(-h) * np.array([[1 + h, h], [-h, 1 - h]])
        covariance = np.eye(2) - a @ a.T
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        root = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0)))
        state = a @ state + root @ innovations[i]
        result[i] = state[0]
    return result


def lagged_poses(poses, times, lag_seconds):
    if lag_seconds == 0 or len(times) == 1:
        return poses.copy()
    query = np.clip(times - lag_seconds, times[0], times[-1])
    result = poses.copy()
    result[:, :3, :3] = Slerp(times, Rotation.from_matrix(poses[:, :3, :3]))(
        query
    ).as_matrix()
    for axis in range(3):
        result[:, axis, 3] = np.interp(query, times, poses[:, axis, 3])
    return result


@dataclass(frozen=True)
class PoseError:
    true_poses: np.ndarray  # raw tracker coordinate convention
    raw_poses: np.ndarray
    components: (
        dict  # each: [rotation vector radians (3), origin displacement metres (3)]
    )
    bias_clock: np.ndarray
    config: PoseErrorConfig

    def metadata(self):
        return {
            "model": "tracker_correlated_bias_v1",
            "config": asdict(self.config),
            "coordinate_space": "raw_tracker",
            "component_frame": self.config.frame,
            "component_convention": "[rotation_vector_rad(3), origin_displacement_m(3)]; not an se3 twist",
            "composition": (
                "R_raw = Exp(sum(rotation)) @ R_source; t_raw = t_source + sum(displacement)"
                if self.config.frame == "world"
                else "R_raw = R_source @ Exp(sum(rotation)); t_raw = t_source + R_source @ sum(displacement)"
            ),
            "source": "true_tracker_pose(t - effective_lag_seconds), clamped to recording endpoints",
            "effective_lag_seconds": self.config.lag_seconds
            if self.config.enabled
            else 0.0,
            "amplitude_convention": "ensemble RMS vector norm; isotropic coordinate std = RMS / sqrt(3)",
            "true_tracker_poses": self.true_poses.tolist(),
            "raw_tracker_poses": self.raw_poses.tolist(),
            "bias_clock": self.bias_clock.tolist(),
            "components": {
                name: value.tolist() for name, value in self.components.items()
            },
        }


def generate_pose_error(true_poses, frame_times_seconds, config, seed):
    """Apply tracker-origin displacement and rotation, with no jumps or patient motion."""
    config.validate()
    poses, times = validate_trajectory(true_poses, frame_times_seconds)
    elapsed = np.diff(times) / config.bias_time_seconds
    if config.bias_distance_m:
        elapsed += (
            np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)
            / config.bias_distance_m
        )
    if config.bias_angle_deg:
        relative = poses[1:, :3, :3] @ poses[:-1, :3, :3].swapaxes(1, 2)
        angles = (
            Rotation.from_matrix(relative).magnitude() if len(relative) else np.zeros(0)
        )
        elapsed += angles / np.radians(config.bias_angle_deg)
    clock = np.r_[0.0, np.cumsum(elapsed)]
    components = {}
    for name in ("bias", "colored", "white"):
        parts = []
        for channel in ("rotation", "translation"):
            rng = named_rng(seed, f"pose_error/{name}/{channel}")
            if name == "bias":
                values = smooth_bias(clock, rng)
            elif name == "colored":
                values = correlated_noise(
                    times, getattr(config, f"{channel}_correlation_seconds"), rng
                )
            else:
                values = rng.normal(size=(len(times), 3))
            amplitude = getattr(
                config,
                f"{name}_{channel}_rms_{'deg' if channel == 'rotation' else 'mm'}",
            )
            scale = np.radians(amplitude) if channel == "rotation" else amplitude / 1000
            parts.append(values * (scale / np.sqrt(3)) * config.enabled)
        components[name] = np.concatenate(parts, axis=1)
    total = sum(components.values())
    source = lagged_poses(poses, times, config.lag_seconds if config.enabled else 0)
    raw = source.copy()
    delta = Rotation.from_rotvec(total[:, :3]).as_matrix()
    if config.frame == "world":
        raw[:, :3, :3] = delta @ source[:, :3, :3]
        raw[:, :3, 3] += total[:, 3:]
    else:
        raw[:, :3, :3] = source[:, :3, :3] @ delta
        raw[:, :3, 3] += np.einsum("nij,nj->ni", source[:, :3, :3], total[:, 3:])
    return PoseError(poses, raw, components, clock, config)


def save_diagnostics(directory, times, error):
    """Realized error, autocorrelation and short-lag changes in tracker world axes."""
    import matplotlib.pyplot as plt

    position = (error.raw_poses[:, :3, 3] - error.true_poses[:, :3, 3]) * 1000
    rotation = np.degrees(
        Rotation.from_matrix(
            error.raw_poses[:, :3, :3] @ error.true_poses[:, :3, :3].swapaxes(1, 2)
        ).as_rotvec()
    )
    fig, axes = plt.subplots(4, 2, figsize=(12, 12), constrained_layout=True)
    stats = {}
    for col, (values, unit, label) in enumerate(
        ((position, "mm", "Translation"), (rotation, "deg", "Rotation"))
    ):
        norms = np.linalg.norm(values, axis=1)
        stats[label.lower()] = {
            "rms": float(np.sqrt(np.mean(norms**2))),
            "p95": float(np.percentile(norms, 95)),
            "unit": unit,
        }
        for j, coordinate in enumerate("xyz"):
            axes[0, col].plot(times, values[:, j], label=coordinate, lw=0.8)
        axes[0, col].set(
            title=f"Realized tracker-origin {label.lower()} error",
            ylabel=unit,
            xlabel="Time (s)",
        )
        axes[0, col].legend()
        component_stats = {}
        for name, component in error.components.items():
            part = component[:, 3:] * 1000 if col == 0 else np.degrees(component[:, :3])
            magnitude = np.linalg.norm(part, axis=1)
            axes[1, col].plot(times, magnitude, label=name, lw=0.8)
            component_stats[name] = float(np.sqrt(np.mean(magnitude**2)))
        axes[1, col].set(xlabel="Time (s)", ylabel=f"Component norm ({unit})")
        axes[1, col].legend()
        stats[label.lower()]["component_rms"] = component_stats
        lags, changes, correlations, pair_counts = [], [], [], []
        tolerance = np.median(np.diff(times)) / 2 if len(times) > 1 else 0
        for lag in (0.0625, 0.125, 0.25, 0.5, 1, 2, 3, 5):
            target = times + lag
            right = np.minimum(np.searchsorted(times, target), len(times) - 1)
            left = np.maximum(right - 1, 0)
            match = np.where(
                abs(times[left] - target) < abs(times[right] - target), left, right
            )
            use = (abs(times[match] - target) <= tolerance) & (
                match > np.arange(len(times))
            )
            if use.sum() < 2:
                continue
            x, y = values[use], values[match[use]]
            change = y - x
            if col == 1:
                # Exact relative change between world error rotations, not a rotvec difference.
                change = np.degrees(
                    (
                        Rotation.from_rotvec(np.radians(y))
                        * Rotation.from_rotvec(np.radians(x)).inv()
                    ).as_rotvec()
                )
            lags.append(float(np.mean(times[match[use]] - times[use])))
            pair_counts.append(int(use.sum()))
            changes.append(float(np.sqrt(np.mean(np.sum(change**2, axis=1)))))
            centered_x, centered_y = x - x.mean(0), y - y.mean(0)
            denom = np.sqrt(
                np.sum(centered_x**2, axis=0) * np.sum(centered_y**2, axis=0)
            )
            correlations.append(
                np.divide(
                    np.sum(centered_x * centered_y, axis=0),
                    denom,
                    out=np.zeros(3),
                    where=denom > 1e-20,
                )
            )
        axes[2, col].plot(lags, changes, "o-")
        axes[2, col].set(xlabel="Actual lag (s)", ylabel=f"RMS change ({unit})")
        if correlations:
            axes[3, col].plot(lags, correlations, "o-")
        axes[3, col].set(
            xlabel="Actual lag (s)", ylabel="Component correlation", ylim=(-1.05, 1.05)
        )
        stats[label.lower()]["lag_seconds"] = lags
        stats[label.lower()]["lag_change_rms"] = changes
        stats[label.lower()]["lag_pair_counts"] = pair_counts
        stats[label.lower()]["lag_component_correlation"] = np.asarray(
            correlations
        ).tolist()
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Injected tracking error (bias + correlated + white; includes optional lag)"
    )
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"pose_error_diagnostics.{suffix}", dpi=140)
    plt.close(fig)
    return {"plot": "pose_error_diagnostics.png", "statistics": stats}
