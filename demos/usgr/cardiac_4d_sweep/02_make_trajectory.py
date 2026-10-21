"""Generate the probe trajectory and check it against the label mask before simulating.

    python 02_make_trajectory.py [--config config.yaml] [--preview-every 25]

The trajectory is one continuous path through the keyframes of every section, in order.
Keyframes carry no times: each interval gets the time its rotation needs at
`max_rotation_dps` (or its translation at `max_translation_mmps`), and all intervals are
then scaled together to `duration_s`. So every part of the path moves at the same fraction
of the speed limits. Position is interpolated monotonically (no overshoot, so a fixed
contact point stays fixed); orientation by a rotation spline.

Writes `trajectory/trajectory.npz` (times, poses, cardiac phase and frame, section per
image) and QA outputs: a 3D path plot, speed traces, images-per-phase counts, coverage of
each chamber by the image planes, probe clearance from the heart, and a montage of label
slices cut by the planned image planes. Needs `01_build_phantoms.py` to have run first.
"""

import argparse
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.interpolate import PchipInterpolator  # noqa: E402
from scipy.ndimage import distance_transform_edt  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.spatial.transform import Rotation, Slerp  # noqa: E402

from common import (  # noqa: E402
    Paths,
    cardiac_phase,
    create_sector,
    label_slice,
    load_config,
    load_cropped_mask,
    look_at_rotation,
    save_json,
    sector_world_points,
)


def read_keyframes(config: dict, heart_center_m: np.ndarray) -> dict:
    """Keyframe look-at parameters in world units: position, target, roll (rad) and lateral
    hint per keyframe (NaN target for position-only waypoints), plus each one's section."""
    rows = []
    for index, section in enumerate(config["trajectory"]["sections"]):
        for keyframe in section["keyframes"]:
            oriented = "target" in keyframe
            lateral = keyframe.get("lateral", section.get("lateral"))
            if oriented and lateral is None:
                raise ValueError(f"section {section['name']}: oriented keyframes need a `lateral` direction")
            rows.append({
                "position": heart_center_m + np.array(keyframe["position"]) * 1e-3,
                "target": heart_center_m + np.array(keyframe["target"]) * 1e-3 if oriented else np.full(3, np.nan),
                "roll": math.radians(keyframe.get("roll_deg", 0.0)),
                "lateral": np.array(lateral, dtype=float) / np.linalg.norm(lateral) if oriented else np.full(3, np.nan),
                "section": index,
            })
    keyframes = {key: np.array([row[key] for row in rows]) for key in rows[0]}
    keyframes["oriented"] = ~np.isnan(keyframes["target"][:, 0])
    if not (keyframes["oriented"][0] and keyframes["oriented"][-1]):
        raise ValueError("the first and last keyframes need a target")
    return keyframes


def keyframe_poses(keyframes: dict) -> np.ndarray:
    """Poses (K, 4, 4) of the keyframes, used to time the intervals.

    A position-only waypoint's orientation is slerped between the oriented keyframes
    around it, in proportion to path length.
    """
    oriented = keyframes["oriented"]
    poses = np.tile(np.eye(4), (len(oriented), 1, 1))
    poses[:, :3, 3] = keyframes["position"]
    for k in np.flatnonzero(oriented):
        poses[k, :3, :3] = look_at_rotation(keyframes["position"][k], keyframes["target"][k], keyframes["roll"][k], keyframes["lateral"][k])

    known = np.flatnonzero(oriented)
    for a, b in zip(known[:-1], known[1:]):
        if b - a < 2:
            continue
        steps = np.linalg.norm(np.diff(poses[a : b + 1, :3, 3], axis=0), axis=1)
        fraction = np.cumsum(steps)[:-1] / steps.sum()
        slerp = Slerp([0.0, 1.0], Rotation.from_matrix(poses[[a, b], :3, :3]))
        poses[a + 1 : b, :3, :3] = slerp(fraction).as_matrix()
    return poses


def interpolate_look_at(keyframes: dict, knot_times: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Frame poses from monotone (PCHIP) interpolation of the look-at parameters.

    Position runs through every keyframe; target, roll and lateral hint run through the
    oriented ones. The beam therefore always points at a smoothly moving target: a fan
    tilts exactly along its targets, and a roll stays a pure rotation about the beam.
    """
    oriented = keyframes["oriented"]
    position = PchipInterpolator(knot_times, keyframes["position"], axis=0)(times)
    target = PchipInterpolator(knot_times[oriented], keyframes["target"][oriented], axis=0)(times)
    roll = PchipInterpolator(knot_times[oriented], keyframes["roll"][oriented])(times)
    lateral = PchipInterpolator(knot_times[oriented], keyframes["lateral"][oriented], axis=0)(times)
    poses = np.tile(np.eye(4), (len(times), 1, 1))
    poses[:, :3, 3] = position
    for i in range(len(times)):
        poses[i, :3, :3] = look_at_rotation(position[i], target[i], roll[i], lateral[i])
    return poses


def interval_owners(knot_sections: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Section owning each keyframe interval, and whether it is a link.

    An interval within a section belongs to it; an interval between sections belongs to
    whichever side is a link (or to the later section if neither is).
    """
    link = np.array([section.get("link", False) for section in config["trajectory"]["sections"]])
    earlier, later = knot_sections[:-1], knot_sections[1:]
    owner = np.where(link[later], later, np.where(link[earlier], earlier, later))
    return owner, link[owner]


def fill_duration(durations: np.ndarray, is_link: np.ndarray, total: float) -> np.ndarray:
    """Scale the non-link intervals so all intervals add up to `total`."""
    remaining = total - durations[is_link].sum()
    if remaining <= 0:
        raise ValueError(f"duration_s is too short: the links alone need {durations[is_link].sum():.1f} s")
    durations = durations.copy()
    durations[~is_link] *= remaining / durations[~is_link].sum()
    return durations


def keyframe_times(keyframes: dict, is_link: np.ndarray, config: dict, iterations: int = 10) -> np.ndarray:
    """Knot times from the speed limits.

    The first estimate gives each interval the time its keyframe-to-keyframe rotation needs
    at `max_rotation_dps` (or its translation at `max_translation_mmps`). The interpolated
    path can move faster than that estimate, so each link is then stretched until its
    measured peak speed is `link_speed_fraction` of the limits. The other intervals share
    the rest of `duration_s`, so the sweeps get the images.
    """
    limits = config["trajectory"]
    poses = keyframe_poses(keyframes)
    rotations = Rotation.from_matrix(poses[:, :3, :3])
    angles = np.degrees((rotations[:-1].inv() * rotations[1:]).magnitude())
    distances = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1) * 1e3
    needed = np.maximum(angles / limits["max_rotation_dps"], distances / limits["max_translation_mmps"])
    if np.any(needed <= 0):
        raise ValueError("consecutive keyframes must differ in position or orientation")
    durations = fill_duration(needed / limits["link_speed_fraction"], is_link, limits["duration_s"])

    for _ in range(iterations):
        knot_times = np.concatenate([[0.0], np.cumsum(durations)])
        fine = np.linspace(0.0, knot_times[-1], int(knot_times[-1] * 50) + 1)
        translation, rotation = speeds({"times_s": fine, "poses": interpolate_look_at(keyframes, knot_times, fine)})
        interval = np.clip(np.searchsorted(knot_times, fine[1:], side="right") - 1, 0, len(durations) - 1)
        peak = np.zeros(len(durations))
        np.maximum.at(peak, interval, np.maximum(rotation / limits["max_rotation_dps"], translation / limits["max_translation_mmps"]))
        if np.any(peak[~is_link] > 1.0):
            raise ValueError("a sweep exceeds the speed limits; increase duration_s")
        stretch = is_link & (peak > limits["link_speed_fraction"] * 1.01)
        if not stretch.any():
            return knot_times
        durations[stretch] *= peak[stretch] / limits["link_speed_fraction"]
        durations = fill_duration(durations, is_link, limits["duration_s"])
    raise ValueError("link timing did not converge")


def build_trajectory(config: dict, heart_center_m: np.ndarray) -> dict:
    """The continuous path through every section, sampled at the frame rate."""
    keyframes = read_keyframes(config, heart_center_m)
    owner, is_link = interval_owners(keyframes["section"], config)
    knot_times = keyframe_times(keyframes, is_link, config)
    times = np.arange(0.0, knot_times[-1] + 1e-9, 1.0 / config["acquisition"]["frame_rate_hz"])
    interval = np.clip(np.searchsorted(knot_times, times, side="right") - 1, 0, len(owner) - 1)

    phase, frame = cardiac_phase(times, config)
    return {
        "times_s": times,
        "poses": interpolate_look_at(keyframes, knot_times, times),
        "section": owner[interval],
        "cardiac_phase": phase,
        "cardiac_frame": frame,
    }


def speeds(trajectory: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per-step translation (mm/s) and rotation (deg/s) speeds."""
    dt = np.diff(trajectory["times_s"])
    poses = trajectory["poses"]
    translation = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=-1) * 1e3 / dt
    relative = np.einsum("nji,njk->nik", poses[:-1, :3, :3], poses[1:, :3, :3])
    rotation = np.degrees(np.linalg.norm(Rotation.from_matrix(relative).as_rotvec(), axis=-1)) / dt
    return translation, rotation


def coverage(trajectory: dict, sector, mask: np.ndarray, spacing_m: np.ndarray, chambers: dict, within_mm: float) -> tuple[dict, np.ndarray]:
    """Per chamber (union over the cycle): fraction of voxels within `within_mm` of an image
    plane, and the 95th-percentile and largest distance to one. Elevation thickness is ignored.
    Also returns, per image, whether its plane cuts any chamber in its own cardiac frame."""
    shape = np.array(mask.shape[1:])
    hit = np.zeros(shape, dtype=bool)
    lookup = np.zeros(256, dtype=bool)
    lookup[[label for labels in chambers.values() for label in labels]] = True
    sees_heart = np.zeros(len(trajectory["poses"]), dtype=bool)
    for i, (pose, frame) in enumerate(zip(trajectory["poses"], trajectory["cardiac_frame"])):
        indices = np.round(sector_world_points(sector, pose) / spacing_m + shape / 2.0).astype(int)
        indices = indices[np.all((indices >= 0) & (indices < shape), axis=1)]
        hit[tuple(indices.T)] = True
        sees_heart[i] = lookup[np.asarray(mask[frame])[tuple(indices.T)]].any()
    distance = distance_transform_edt(~hit, sampling=spacing_m * 1e3)
    result = {}
    for name, labels in chambers.items():
        voxels = np.zeros(shape, dtype=bool)
        for frame in mask:
            voxels |= np.isin(frame, labels)
        d = distance[voxels]
        result[name] = {
            f"within_{within_mm:g}mm": round(float((d <= within_mm).mean()), 4),
            "p95_gap_mm": round(float(np.percentile(d, 95)), 2),
            "max_gap_mm": round(float(d.max()), 2),
        }
    return result, sees_heart


def probe_clearance_mm(trajectory: dict, mask: np.ndarray, spacing_m: np.ndarray, heart_labels: list[int]) -> float:
    """Smallest distance from the probe to the heart (any frame) along the whole path."""
    heart = np.zeros(mask.shape[1:], dtype=bool)
    for frame in mask:
        heart |= np.isin(frame, heart_labels)
    points = (np.argwhere(heart)[::5] - np.array(heart.shape) / 2.0) * spacing_m
    distance, _ = cKDTree(points).query(trajectory["poses"][:, :3, 3])
    return float(distance.min() * 1e3)


def save_plots(trajectory: dict, translation, rotation, names: list[str], output_dir) -> None:
    positions = trajectory["poses"][:, :3, 3] * 1e3
    beams = trajectory["poses"][:, :3, 0]
    figure = plt.figure(figsize=(14, 4))
    axis = figure.add_subplot(1, 3, 1, projection="3d")
    for section in np.unique(trajectory["section"]):
        keep = trajectory["section"] == section
        axis.plot(*positions[keep].T, label=names[section])
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
        axis.set_title(f"#{index} s{trajectory['section'][index]} f{frame}", fontsize=7)
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

    names = [section["name"] for section in config["trajectory"]["sections"]]
    translation, rotation = speeds(trajectory)
    counts = np.bincount(trajectory["cardiac_frame"], minlength=config["cardiac"]["num_frames"])
    summary = {
        "num_images": len(trajectory["times_s"]),
        "duration_s": float(trajectory["times_s"][-1]),
        "num_simulations": len(trajectory["times_s"]) * config["transducer"]["ray_num"],
        "sections": {
            name: {
                "images": int((trajectory["section"] == index).sum()),
                "max_rotation_deg_s": round(float(rotation[trajectory["section"][1:] == index].max()), 2),
            }
            for index, name in enumerate(names)
        },
        "images_per_cardiac_frame": counts.tolist(),
        "max_translation_mm_s": round(float(translation.max()), 2),
        "max_rotation_deg_s": round(float(rotation.max()), 2),
    }

    if not args.skip_qa:
        mask = load_cropped_mask(paths)
        spacing_m = np.array(info["mask_spacing_m"])
        sector = create_sector(config)
        summary["probe_clearance_mm"] = round(probe_clearance_mm(trajectory, mask, spacing_m, config["mask"]["heart_labels"]), 1)
        summary["coverage"], sees_heart = coverage(trajectory, sector, mask, spacing_m, config["mask"]["chambers"], within_mm=2.0)
        summary["images_without_heart"] = {
            name: int((~sees_heart & (trajectory["section"] == index)).sum()) for index, name in enumerate(names)
        }
        save_plots(trajectory, translation, rotation, names, paths.trajectory)
        save_preview(trajectory, sector, mask, spacing_m, args.preview_every, paths.trajectory)

    save_json(summary, paths.trajectory / "summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
