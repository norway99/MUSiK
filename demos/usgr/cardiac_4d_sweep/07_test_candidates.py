"""Candidate novel-view test frames: perturbed training poses, scored for novelty.

    python 07_test_candidates.py [--config config.yaml]

Each candidate follows `synthetic/generate.py:perturbed_test_frames`: pick a random training
frame and rotate its imaging frame about a random probe-local axis through the image centre
(the centroid of the sector's pixels), then translate it by exactly `translation_mm` along a
random probe-local direction, x' = R (x - c) + c + d. `rotation_deg` is either a fixed angle
or a [min, max] range sampled uniformly per candidate. The cardiac phase is uniform.

Novelty follows `synthetic/nvs.py:novelty`, with phase unconstrained: the nearest training
frame by mean distance between corresponding image points under the two poses (a coarse
grid of sector points, standing in for token centres), also trying each training frame
rolled 180 degrees about its depth axis (the same image mirrored). Support is the fraction
of the candidate's points within 1 / 2 / 5 mm of any training frame's points.

Writes `test_candidates/candidates.npz`, `candidates.json` and `candidates.png` (each
candidate's tissue mask next to its nearest training frame's, at their own phases).
"""

import argparse
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from common import Paths, create_sector, label_slice, load_config, load_cropped_mask, load_trajectory, save_json  # noqa: E402

# 180-degree roll about the probe-local depth axis (x): the same image mirrored left-right.
FLIP = np.diag([1.0, -1.0, -1.0])
SUPPORT_MM = (1.0, 2.0, 5.0)
POINT_STRIDE = 16  # pixels between scoring points, roughly a token grid
HEART_LABELS = [1, 2, 3, 4, 5, 8, 9, 10]


def random_unit_vectors(rng: np.random.Generator, count: int) -> np.ndarray:
    directions = rng.normal(size=(count, 3))
    return directions / np.linalg.norm(directions, axis=-1, keepdims=True)


def perturbed_poses(train_poses: np.ndarray, pivot: np.ndarray, count: int, rotation_deg,
                    translation_mm: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Base training indices, rotation angles (deg) and [K, 4, 4] poses, as in
    generate.py:perturbed_test_frames; `rotation_deg` is a fixed angle or a [min, max] range."""
    base = rng.choice(len(train_poses), size=count, replace=True)
    axes = random_unit_vectors(rng, count)
    directions = random_unit_vectors(rng, count)
    low, high = (rotation_deg, rotation_deg) if np.isscalar(rotation_deg) else rotation_deg
    angles = rng.uniform(low, high, size=count)
    rotations = Rotation.from_rotvec(axes * np.radians(angles)[:, None]).as_matrix()
    offsets = np.tile(np.eye(4), (count, 1, 1))
    offsets[:, :3, :3] = rotations
    offsets[:, :3, 3] = pivot - rotations @ pivot + directions * translation_mm * 1e-3
    return base, angles, train_poses[base] @ offsets


def plane_points(local: np.ndarray, poses: np.ndarray) -> np.ndarray:
    """World positions [N, P, 3] of probe-local points [P, 3] under poses [N, 4, 4]."""
    return np.einsum("nij,pj->npi", poses[:, :3, :3], local) + poses[:, None, :3, 3]


def rotation_deg(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(math.acos(np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2, -1, 1)))


def novelty(test_poses: np.ndarray, train_poses: np.ndarray, local: np.ndarray) -> list[dict]:
    flipped_poses = train_poses.copy()
    flipped_poses[:, :3, :3] = train_poses[:, :3, :3] @ FLIP
    train_points = plane_points(local, train_poses)
    flipped_points = plane_points(local, flipped_poses)
    tree = cKDTree(train_points.reshape(-1, 3))
    rows = []
    for pose, points in zip(test_poses, plane_points(local, test_poses)):
        direct = np.linalg.norm(train_points - points, axis=-1).mean(-1) * 1e3
        rolled = np.linalg.norm(flipped_points - points, axis=-1).mean(-1) * 1e3
        distance = np.minimum(direct, rolled)
        nearest = int(distance.argmin())
        flipped = bool(rolled[nearest] < direct[nearest])
        matched = flipped_poses[nearest] if flipped else train_poses[nearest]
        support = tree.query(points)[0] * 1e3
        rows.append({
            "nearest_train": nearest,
            "nearest_flipped": flipped,
            "nearest_distance_mm": round(float(distance[nearest]), 2),
            "nearest_angle_deg": round(rotation_deg(pose, matched), 1),
            **{f"support_{r:g}mm": round(float((support <= r).mean()), 3) for r in SUPPORT_MM},
            "support_median_mm": round(float(np.median(support)), 2),
        })
    return rows


def save_panels(candidates: dict, rows: list[dict], trajectory: dict, sector, mask, spacing_m, path,
                titles: list[str] | None = None, pairs_per_row: int = 5, slots: list[int] | None = None) -> list[float]:
    """Each candidate's tissue mask beside its nearest training frame's; returns heart fractions.

    `titles` replaces the default first title line of each candidate; `slots` places each
    candidate at a grid position (row * pairs_per_row + column) instead of in sequence.
    """
    count = len(rows)
    slots = list(range(count)) if slots is None else list(slots)
    grid_rows = max(slots) // pairs_per_row + 1
    # A narrow spacer column between pairs keeps each candidate visibly grouped with its neighbor.
    figure = plt.figure(figsize=(2.0 * 2 * pairs_per_row + 0.3 * (pairs_per_row - 1), 2.3 * grid_rows))
    layout = figure.add_gridspec(grid_rows, 3 * pairs_per_row - 1, width_ratios=([1, 1, 0.15] * pairs_per_row)[:-1])
    pairs = [
        (figure.add_subplot(layout[r, 3 * c]), figure.add_subplot(layout[r, 3 * c + 1]))
        for r in range(grid_rows)
        for c in range(pairs_per_row)
    ]
    vmax = int(np.asarray(mask[0]).max())
    shown = lambda labels: np.ma.masked_where(~sector.valid, labels)
    heart_fractions = []
    for axis in (axis for pair in pairs for axis in pair):
        axis.axis("off")
    pairs = [pairs[slot] for slot in slots]
    for k, row in enumerate(rows):
        test_labels = label_slice(sector, candidates["poses"][k], np.asarray(mask[candidates["cardiac_frame"][k]]), spacing_m)
        n = row["nearest_train"]
        train_labels = label_slice(sector, trajectory["poses"][n], np.asarray(mask[trajectory["cardiac_frame"][n]]), spacing_m)
        if row["nearest_flipped"]:
            train_labels = train_labels[:, ::-1]  # the rolled pose shows the training image mirrored
        heart = float(np.isin(test_labels[sector.valid], HEART_LABELS).mean())
        heart_fractions.append(heart)
        left, right = pairs[k]
        left.imshow(shown(test_labels), cmap="tab10", vmin=0, vmax=vmax, interpolation="nearest")
        first = titles[k] if titles else f"cand {k} (f{candidates['cardiac_frame'][k]}, rot {candidates['rotation_deg'][k]:.0f}°)"
        left.set_title(f"{first}, heart {100 * heart:.0f}%\n"
                       f"{row['nearest_distance_mm']:.1f} mm / {row['nearest_angle_deg']:.0f}° to nearest", fontsize=7)
        right.imshow(shown(train_labels), cmap="tab10", vmin=0, vmax=vmax, interpolation="nearest")
        right.set_title(f"nearest train #{n} (f{trajectory['cardiac_frame'][n]})"
                        f"{', mirrored' if row['nearest_flipped'] else ''}\nsupport 2mm {100 * row['support_2mm']:.0f}%", fontsize=7)
    figure.tight_layout()
    # Outline each candidate / nearest-training pair, titles included.
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for left, right in pairs[:count]:
        box = Bbox.union([axis.get_tightbbox(renderer).transformed(figure.transFigure.inverted()) for axis in (left, right)])
        figure.add_artist(FancyBboxPatch(
            (box.x0, box.y0), box.width, box.height, boxstyle="round,pad=0.003",
            transform=figure.transFigure, fill=False, edgecolor="0.3", linewidth=1.2,
        ))
    figure.savefig(path, dpi=90)
    plt.close(figure)
    return heart_fractions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    test = config["test_frames"]
    paths = Paths(config)
    out = paths.root / "test_candidates"
    out.mkdir(parents=True, exist_ok=True)
    trajectory = load_trajectory(paths)
    sector = create_sector(config)
    with (paths.phantoms / "phantom_info.json").open() as handle:
        spacing_m = np.array(json.load(handle)["mask_spacing_m"])

    pivot = sector.local_points[sector.valid].astype(np.float64).mean(axis=0)  # image centre
    rng = np.random.default_rng(test["seed"])
    base, angles, poses = perturbed_poses(trajectory["poses"], pivot, test["candidates"], test["rotation_deg"], test["translation_mm"], rng)
    phase = rng.uniform(0.0, 1.0, size=len(base))  # fraction of a cycle, like trajectory["cardiac_phase"]
    frame = np.round(phase * config["cardiac"]["num_frames"]).astype(int) % config["cardiac"]["num_frames"]
    candidates = {"poses": poses, "base_train": base, "rotation_deg": angles, "cardiac_phase": phase, "cardiac_frame": frame}

    stride = sector.valid[::POINT_STRIDE, ::POINT_STRIDE]
    local = sector.local_points[::POINT_STRIDE, ::POINT_STRIDE][stride].astype(np.float64)
    rows = novelty(poses, trajectory["poses"], local)
    hearts = save_panels(candidates, rows, trajectory, sector, load_cropped_mask(paths), spacing_m, out / "candidates.png")

    for k, row in enumerate(rows):
        row.update({"candidate": k, "base_train": int(base[k]), "rotation_deg": round(float(angles[k]), 1),
                    "cardiac_frame": int(frame[k]), "heart_fraction": round(hearts[k], 3)})
    np.savez(out / "candidates.npz", **candidates)
    save_json({"pivot_probe_local_m": pivot.tolist(), "settings": test, "scoring_points": int(len(local)), "candidates": rows}, out / "candidates.json")
    print(f"{'cand':>4s} {'base':>5s} {'rot':>4s} {'near':>5s} {'dist mm':>8s} {'angle':>6s} {'sup1':>5s} {'sup2':>5s} {'sup5':>5s} {'heart':>6s}")
    for row in rows:
        print(f"{row['candidate']:4d} {row['base_train']:5d} {row['rotation_deg']:4.0f} {row['nearest_train']:5d}{'*' if row['nearest_flipped'] else ' '}"
              f"{row['nearest_distance_mm']:7.1f} {row['nearest_angle_deg']:6.0f} {row['support_1mm']:5.2f} {row['support_2mm']:5.2f} "
              f"{row['support_5mm']:5.2f} {100 * row['heart_fraction']:5.0f}%")
    print("* nearest match is the mirrored (rolled) training pose")


if __name__ == "__main__":
    main()
