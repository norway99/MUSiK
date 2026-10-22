"""Candidate test frames at canonical TTE views: a best-guess pose per view, plus perturbed samples.

    python 08_canonical_views.py [--config config.yaml]

Best guesses come from the sweep's three windows and landmarks found in the label mask
(end-diastolic frame): the mitral and aortic valves (where the LV meets the LA / aorta)
and the ascending aorta's centre and axis.

- PSAX aortic valve / mitral valve / mid-ventricular: the parasternal probe in short-axis
  orientation, aimed at the aortic valve, the mitral valve, and a point `psax_mid_fraction`
  of the way from the mitral valve to the apex.
- PLAX: the mid-level short axis rolled 90 degrees, so the plane holds the LV long axis.
- A4C: the apical sweep's starting view. A5C: A4C pitched `a5c_tilt_deg` toward the aortic
  valve. A3C: A4C rolled to the angle whose plane passes closest to the aortic valve.
  A2C: A4C rolled to the angle (45-135 degrees) showing the least right heart.
- Subcostal LV: the subcostal fan's central view rolled 90 degrees, aimed at the LV.
- Right parasternal ascending aorta: the PLAX probe moved `right_parasternal_shift_mm`
  across the sternum (and superior), aimed at the ascending aorta along its axis.

Sample 0 of each view is the best guess; the rest add a random rotation of up to
`rotation_max_deg` about the image centre and a random shift of up to `translation_max_mm`.
Each sample has a uniform random cardiac phase. Novelty and panels as in 07.

Writes `canonical_views/views.npz`, `views.json`, `all_views.png` (one row per view) and
`view_<name>.png`.
"""

import argparse
import importlib
import json
import math

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial.transform import Rotation

from common import Paths, create_sector, label_slice, load_config, load_cropped_mask, load_trajectory, look_at_rotation, save_json

candidates_module = importlib.import_module("07_test_candidates")

RIGHT_HEART = [4, 5, 8, 10]
SHORT_NAMES = {
    "psax_aortic_valve": "PSAX-AV", "psax_mitral_valve": "PSAX-MV", "psax_mid_ventricular": "PSAX-mid", "plax": "PLAX",
    "apical_4_chamber": "A4C", "apical_5_chamber": "A5C", "apical_3_chamber": "A3C", "apical_2_chamber": "A2C",
    "subcostal_lv": "SC-LV", "right_parasternal_ascending_aorta": "RPS-AAo",
}


def landmarks(mask_frame: np.ndarray, spacing_m: np.ndarray, heart_center_m: np.ndarray) -> dict:
    """Valve and ascending-aorta landmarks in metres, world frame."""
    shape = np.array(mask_frame.shape)
    world = lambda indices: (np.asarray(indices, float) - shape / 2) * spacing_m

    def interface(a, b):
        return world(np.argwhere((mask_frame == a) & binary_dilation(mask_frame == b, iterations=2)).mean(0))

    aortic = interface(3, 6)
    aorta = world(np.argwhere(mask_frame == 6))
    # The aorta near the valve and not posterior to it: the ascending segment, not the arch's descent.
    ascending = aorta[(np.linalg.norm(aorta - aortic, axis=1) < 0.045) & (aorta[:, 1] < aortic[1] + 0.010)]
    centre = ascending.mean(0)
    axis = np.linalg.eigh(np.cov((ascending - centre).T))[1][:, -1]
    axis *= -np.sign(axis[0])  # point superior (-Z)
    return {"mitral": interface(3, 2), "aortic": aortic, "ascending_centre": centre, "ascending_axis": axis,
            "heart_center": heart_center_m}


def pose(position, target, roll_deg, lateral) -> np.ndarray:
    result = np.eye(4)
    result[:3, 3] = position
    result[:3, :3] = look_at_rotation(position, target, math.radians(roll_deg), lateral)
    return result


def pitched(base: np.ndarray, degrees: float) -> np.ndarray:
    """Tilt the image plane: rotate the probe about its lateral axis."""
    result = base.copy()
    result[:3, :3] = base[:3, :3] @ Rotation.from_rotvec([0.0, math.radians(degrees), 0.0]).as_matrix()
    return result


def plane_distance(p: np.ndarray, point: np.ndarray) -> float:
    """Distance from a world point to the image plane of pose p (normal = probe elevation axis)."""
    return abs(float(np.dot(point - p[:3, 3], p[:3, 2])))


def best_guesses(config: dict, marks: dict, sector, mask_frame, spacing_m) -> tuple[dict, dict]:
    """Best-guess pose per view, and the angles chosen along the way."""
    hc = marks["heart_center"]
    sweep = {s["name"]: s for s in config["trajectory"]["sections"]}
    mm = lambda v: hc + np.asarray(v, float) * 1e-3
    views = config["canonical_views"]

    para = sweep["parasternal_short_axis"]
    para_position, para_lateral = mm(para["keyframes"][0]["position"]), para["lateral"]
    apex = mm([10, -27, 55])
    mid = marks["mitral"] + views["psax_mid_fraction"] * (apex - marks["mitral"])

    apical = sweep["apical_rotation"]["keyframes"][0]
    a_position, a_target, a_lateral = mm(apical["position"]), mm(apical["target"]), sweep["apical_rotation"]["lateral"]
    a4c = pose(a_position, a_target, 0, a_lateral)
    rolls = np.arange(0, 180, 2.5)
    a3c_roll = float(min((r for r in rolls if 20 <= r <= 160),
                         key=lambda r: plane_distance(pose(a_position, a_target, r, a_lateral), marks["aortic"])))

    def right_heart_fraction(r):
        labels = label_slice(sector, pose(a_position, a_target, r, a_lateral), mask_frame, spacing_m)
        return float(np.isin(labels[sector.valid], RIGHT_HEART).mean())

    a2c_roll = float(min((r for r in rolls if 45 <= r <= 135), key=right_heart_fraction))
    tilts = [views["a5c_tilt_deg"], -views["a5c_tilt_deg"]]
    a5c_tilt = min(tilts, key=lambda t: plane_distance(pitched(a4c, t), marks["aortic"]))

    sub = sweep["subcostal_fan"]["keyframes"][1]
    sub_position = mm(sub["position"])
    lv_centre = mm([0, 10, 23])

    shift = views["right_parasternal_shift_mm"]
    right_position = para_position + np.array([-0.4 * shift, 0.0, -shift]) * 1e-3  # superior and to the patient's right

    guesses = {
        "psax_aortic_valve": pose(para_position, marks["aortic"], 90, para_lateral),
        "psax_mitral_valve": pose(para_position, marks["mitral"], 90, para_lateral),
        "psax_mid_ventricular": pose(para_position, mid, 90, para_lateral),
        "plax": pose(para_position, mid, 0, para_lateral),
        "apical_4_chamber": a4c,
        "apical_5_chamber": pitched(a4c, a5c_tilt),
        "apical_3_chamber": pose(a_position, a_target, a3c_roll, a_lateral),
        "apical_2_chamber": pose(a_position, a_target, a2c_roll, a_lateral),
        "subcostal_lv": pose(sub_position, lv_centre, 90, sweep["subcostal_fan"]["lateral"]),
        "right_parasternal_ascending_aorta": pose(right_position, marks["ascending_centre"], 0, marks["ascending_axis"]),
    }
    choices = {"a3c_roll_deg": a3c_roll, "a2c_roll_deg": a2c_roll, "a5c_tilt_deg": a5c_tilt,
               "right_parasternal_probe_mm": ((right_position - hc) * 1e3).round(1).tolist()}
    return guesses, choices


def perturb(base: np.ndarray, pivot: np.ndarray, rotation_deg: float, translation_mm: float, rng) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    offset = np.eye(4)
    offset[:3, :3] = Rotation.from_rotvec(axis * math.radians(rotation_deg)).as_matrix()
    offset[:3, 3] = pivot - offset[:3, :3] @ pivot + direction * translation_mm * 1e-3
    return base @ offset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    views = config["canonical_views"]
    paths = Paths(config)
    out = paths.root / "canonical_views"
    out.mkdir(parents=True, exist_ok=True)
    trajectory = load_trajectory(paths)
    sector = create_sector(config)
    info = json.load(open(paths.phantoms / "phantom_info.json"))
    spacing_m = np.array(info["mask_spacing_m"])
    mask = load_cropped_mask(paths)
    end_diastole = np.asarray(mask[0])

    marks = landmarks(end_diastole, spacing_m, np.array(info["heart_center_m"]))
    guesses, choices = best_guesses(config, marks, sector, end_diastole, spacing_m)
    pivot = sector.local_points[sector.valid].astype(np.float64).mean(axis=0)

    rng = np.random.default_rng(views["seed"])
    poses, view_index, sample, rotation, translation = [], [], [], [], []
    for v, (name, base) in enumerate(guesses.items()):
        for k in range(views["samples_per_view"]):
            angle = 0.0 if k == 0 else rng.uniform(0, views["rotation_max_deg"])
            shift = 0.0 if k == 0 else rng.uniform(0, views["translation_max_mm"])
            poses.append(base if k == 0 else perturb(base, pivot, angle, shift, rng))
            view_index.append(v)
            sample.append(k)
            rotation.append(angle)
            translation.append(shift)
    poses = np.array(poses)
    phase = rng.uniform(0.0, 1.0, size=len(poses))
    frame = np.round(phase * config["cardiac"]["num_frames"]).astype(int) % config["cardiac"]["num_frames"]
    names = list(guesses)
    candidates = {"poses": poses, "view": np.array(view_index), "sample": np.array(sample), "rotation_deg": np.array(rotation),
                  "translation_mm": np.array(translation), "cardiac_phase": phase, "cardiac_frame": frame}

    stride = candidates_module.POINT_STRIDE
    local = sector.local_points[::stride, ::stride][sector.valid[::stride, ::stride]].astype(np.float64)
    rows = candidates_module.novelty(poses, trajectory["poses"], local)
    titles = [f"{SHORT_NAMES[names[v]]} #{k} (f{frame[i]}, {rotation[i]:.0f}°/{translation[i]:.0f}mm)" for i, (v, k) in enumerate(zip(view_index, sample))]
    hearts = candidates_module.save_panels(candidates, rows, trajectory, sector, mask, spacing_m, out / "all_views.png",
                                           titles=titles, pairs_per_row=views["samples_per_view"])
    for v, name in enumerate(names):
        keep = np.flatnonzero(np.array(view_index) == v)
        subset = {key: value[keep] for key, value in candidates.items()}
        candidates_module.save_panels(subset, [rows[i] for i in keep], trajectory, sector, mask, spacing_m, out / f"view_{name}.png",
                                      titles=[titles[i] for i in keep], pairs_per_row=5)

    for i, row in enumerate(rows):
        row.update({"view": names[view_index[i]], "sample": sample[i], "rotation_deg": round(rotation[i], 1),
                    "translation_mm": round(translation[i], 1), "cardiac_frame": int(frame[i]), "heart_fraction": round(hearts[i], 3)})
    np.savez(out / "views.npz", view_names=np.array(names), **candidates)
    save_json({
        "settings": views,
        "choices": choices,
        "landmarks_mm_from_heart_center": {k: ((v - marks["heart_center"]) * 1e3).round(1).tolist() for k, v in marks.items()
                                           if k not in ("heart_center", "ascending_axis")},
        "ascending_axis": marks["ascending_axis"].round(3).tolist(),
        "best_guess_poses": {name: base.tolist() for name, base in guesses.items()},
        "candidates": rows,
    }, out / "views.json")
    print("choices:", json.dumps(choices))
    for v, name in enumerate(names):
        keep = [r for r in rows if r["view"] == name]
        print(f"{name:34s} heart {min(r['heart_fraction'] for r in keep) * 100:3.0f}-{max(r['heart_fraction'] for r in keep) * 100:3.0f}%  "
              f"nearest {min(r['nearest_distance_mm'] for r in keep):5.1f}-{max(r['nearest_distance_mm'] for r in keep):5.1f} mm")


if __name__ == "__main__":
    main()
