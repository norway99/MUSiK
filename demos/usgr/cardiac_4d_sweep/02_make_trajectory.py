"""Generate the probe trajectory and check it against the label mask before simulating.

    python 02_make_trajectory.py [--config config.yaml] [--preview-every 25]

Writes `trajectory/trajectory.npz` (times, poses, cardiac phase and frame, segment per
image) and QA outputs: a 3D path plot, speed traces, images-per-phase counts, a heart
coverage fraction, and a montage of label slices cut by the planned image planes.
Needs `01_build_phantoms.py` to have written `phantom_info.json` first.
"""

import argparse
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from common import (  # noqa: E402
    Paths,
    cardiac_phase,
    create_sector,
    interpolate_poses,
    label_slice,
    load_config,
    load_cropped_mask,
    look_at_rotation,
    save_json,
    sector_world_points,
)


def build_trajectory(config: dict, heart_center_m: np.ndarray) -> dict:
    """Concatenate every segment into one image sequence sampled at the frame rate."""
    frame_period = 1.0 / config["acquisition"]["frame_rate_hz"]
    times, poses, segments = [], [], []
    start = 0.0
    for index, segment in enumerate(config["trajectory"]["segments"]):
        start += segment.get("gap_s", 0.0)
        keyframes = segment["keyframes"]
        knot_times = np.array([k["t"] for k in keyframes], dtype=np.float64)
        knot_poses = np.tile(np.eye(4), (len(keyframes), 1, 1))
        for knot, keyframe in zip(knot_poses, keyframes):
            position = heart_center_m + np.array(keyframe["position"]) * 1e-3
            target = heart_center_m + np.array(keyframe["target"]) * 1e-3
            lateral = keyframe.get("lateral", segment.get("lateral"))
            knot[:3, :3] = look_at_rotation(position, target, math.radians(keyframe["roll_deg"]), lateral)
            knot[:3, 3] = position

        local_times = np.arange(0.0, knot_times[-1] + 1e-9, frame_period)
        times.append(start + local_times)
        poses.append(interpolate_poses(knot_times, knot_poses, local_times))
        segments.append(np.full(len(local_times), index))
        start += local_times[-1] + frame_period

    times = np.concatenate(times)
    phase, frame = cardiac_phase(times, config)
    return {
        "times_s": times,
        "poses": np.concatenate(poses),
        "segment": np.concatenate(segments),
        "cardiac_phase": phase,
        "cardiac_frame": frame,
    }


def speeds(trajectory: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per-step translation (mm/s) and rotation (deg/s) speeds within segments."""
    dt = np.diff(trajectory["times_s"])
    poses = trajectory["poses"]
    translation = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=-1) * 1e3 / dt
    relative = np.einsum("nji,njk->nik", poses[:-1, :3, :3], poses[1:, :3, :3])
    rotation = np.degrees(np.linalg.norm(Rotation.from_matrix(relative).as_rotvec(), axis=-1)) / dt
    same_segment = np.diff(trajectory["segment"]) == 0
    return np.where(same_segment, translation, np.nan), np.where(same_segment, rotation, np.nan)


def coverage(trajectory: dict, sector, mask: np.ndarray, spacing_m: np.ndarray, heart_labels: list[int]) -> float:
    """Fraction of heart voxels (cycle union) hit by at least one image plane, ignoring elevation thickness."""
    heart = np.isin(mask, heart_labels).any(axis=0)
    hit = np.zeros_like(heart)
    shape = np.array(heart.shape)
    for pose in trajectory["poses"]:
        indices = np.round(sector_world_points(sector, pose) / spacing_m + shape / 2.0).astype(int)
        inside = np.all((indices >= 0) & (indices < shape), axis=1)
        hit[tuple(indices[inside].T)] = True
    return float((hit & heart).sum() / max(heart.sum(), 1))


def save_plots(trajectory: dict, translation, rotation, output_dir) -> None:
    positions = trajectory["poses"][:, :3, 3] * 1e3
    beams = trajectory["poses"][:, :3, 0]
    figure = plt.figure(figsize=(14, 4))
    axis = figure.add_subplot(1, 3, 1, projection="3d")
    for segment in np.unique(trajectory["segment"]):
        keep = trajectory["segment"] == segment
        axis.plot(*positions[keep].T, label=f"segment {segment}")
    step = max(len(positions) // 40, 1)
    axis.quiver(*positions[::step].T, *beams[::step].T, length=20, color="gray", linewidth=0.5)
    axis.set_title("probe path (mm) and beam axis")
    axis.legend(fontsize=7)

    axis = figure.add_subplot(1, 3, 2)
    axis.plot(trajectory["times_s"][1:], translation, label="translation (mm/s)")
    axis.plot(trajectory["times_s"][1:], rotation, label="rotation (deg/s)")
    axis.set_xlabel("time (s)")
    axis.legend(fontsize=7)

    axis = figure.add_subplot(1, 3, 3)
    counts = np.bincount(trajectory["cardiac_frame"])
    axis.bar(np.arange(len(counts)), counts)
    axis.set_title("images per cardiac frame")
    figure.tight_layout()
    figure.savefig(output_dir / "trajectory_overview.png", dpi=150)
    plt.close(figure)


def save_preview(trajectory: dict, sector, mask: np.ndarray, spacing_m: np.ndarray, every: int, output_dir) -> None:
    """Montage of label slices along the sweep: the cheapest check that the planes cut the heart."""
    picks = np.arange(0, len(trajectory["times_s"]), every)
    columns = 8
    rows = math.ceil(len(picks) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(2 * columns, 2 * rows), squeeze=False)
    vmax = int(mask.max())
    for axis in axes.flat:
        axis.axis("off")
    for axis, index in zip(axes.flat, picks):
        frame = trajectory["cardiac_frame"][index]
        labels = label_slice(sector, trajectory["poses"][index], mask[frame], spacing_m)
        # Mask outside the sector so it doesn't read as background label 0.
        axis.imshow(np.ma.masked_where(~sector.valid, labels), cmap="tab10", vmin=0, vmax=vmax, interpolation="nearest")
        axis.set_title(f"#{index} s{trajectory['segment'][index]} f{frame}", fontsize=7)
    figure.tight_layout()
    figure.savefig(output_dir / "label_slice_preview.png", dpi=120)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--preview-every", type=int, default=25)
    parser.add_argument("--skip-qa", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = Paths(config)
    paths.trajectory.mkdir(parents=True, exist_ok=True)
    with (paths.phantoms / "phantom_info.json").open() as handle:
        info = json.load(handle)

    trajectory = build_trajectory(config, np.array(info["heart_center_m"]))
    np.savez(paths.trajectory / "trajectory.npz", **trajectory)

    translation, rotation = speeds(trajectory)
    counts = np.bincount(trajectory["cardiac_frame"], minlength=config["cardiac"]["num_frames"])
    summary = {
        "num_images": len(trajectory["times_s"]),
        "duration_s": float(trajectory["times_s"][-1]),
        "num_simulations": len(trajectory["times_s"]) * config["transducer"]["ray_num"],
        "images_per_cardiac_frame": counts.tolist(),
        "max_translation_mm_s": float(np.nanmax(translation)),
        "max_rotation_deg_s": float(np.nanmax(rotation)),
    }

    if not args.skip_qa:
        mask = load_cropped_mask(paths)
        spacing_m = np.array(info["mask_spacing_m"])
        sector = create_sector(config)
        summary["heart_coverage_fraction"] = coverage(trajectory, sector, mask, spacing_m, config["mask"]["heart_labels"])
        save_plots(trajectory, translation, rotation, paths.trajectory)
        save_preview(trajectory, sector, mask, spacing_m, args.preview_every, paths.trajectory)

    save_json(summary, paths.trajectory / "summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
