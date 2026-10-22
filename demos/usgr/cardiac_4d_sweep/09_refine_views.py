"""Refine canonical-view candidates from a previous round's picks (`refine_views` in the config).

    python 09_refine_views.py [--config config.yaml]

Each view either accepts a sample from the previous round as final (`accept: k`) or starts
from one (`from: k`, optionally of another view via `from_view`) and resamples around it:

- deterministic adjustments, applied first:
  - `yaw: septum_vertical`: in-plane rotation (about the image centre) that makes the
    interventricular septum (LV myocardium bordering the RV) vertical in the image.
  - `yaw: center_valves`: in-plane rotation that puts the valve level (midway between the
    mitral and aortic valves) on the image's centre line.
  - `pitch: toward_aortic_valve | toward_tricuspid_valve`: tilt of the plane about the
    probe's lateral axis that passes it closest to that valve.
  - `slide: center_heart`: slide the probe along the image's lateral axis so the heart
    (its pixels in that frame's label image) is centred left-right.
- `from` may list several samples; the start is then their average pose.
- random sampling around the adjusted pose (sample 0 is the adjusted pose itself):
  `pitch_range_deg` / `roll_range_deg` / `yaw_range_deg` (rotations about the image centre,
  which stays fixed: tilt of the plane, rotation about the beam, in-plane), `lateral_shift_mm` (probe slide along the image's
  lateral axis), `advance_mm: [lo, hi]` (push along the beam, into the chest; sample 0 takes
  the midpoint), then `rotation_max_deg` / `translation_max_mm` about the image centre.
- or a deterministic progression: `steps: [[pitch, roll], ...]` (degrees about the image
  centre, added to the start pose; all samples keep the start frame's phase).
- or a space-filling cluster: `grid: {count, yaw_deg, pitch_deg, roll_deg, lateral_mm,
  elevation_mm}` (scrambled Sobol over +/- each span; rotations about the image centre,
  then probe translation; all at the start frame's phase).

Writes `<round>/views.npz`, `views.json`, `all_views.png` and `view_<name>.png`.
"""

import argparse
import importlib
import json
import math

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial.transform import Rotation
from scipy.stats import qmc

from common import Paths, create_sector, label_slice, load_config, load_cropped_mask, load_trajectory, save_json

candidates_module = importlib.import_module("07_test_candidates")
views_module = importlib.import_module("08_canonical_views")

SHORT_NAMES = {**views_module.SHORT_NAMES, "parasternal_rv_inflow": "PS-RVIF"}


def rotated(base: np.ndarray, pivot: np.ndarray, axis: int, degrees: float) -> np.ndarray:
    """Rotate about the probe-local axis (0 beam / roll, 1 lateral / pitch, 2 elevation / yaw)
    through `pivot`, which stays fixed in space."""
    vector = np.zeros(3)
    vector[axis] = math.radians(degrees)
    offset = np.eye(4)
    offset[:3, :3] = Rotation.from_rotvec(vector).as_matrix()
    offset[:3, 3] = pivot - offset[:3, :3] @ pivot
    return base @ offset


def in_plane(base: np.ndarray, pivot: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate the image within its plane (about the probe-local elevation axis through `pivot`)."""
    return rotated(base, pivot, 2, degrees)


def slid(base: np.ndarray, millimetres: float, axis: int = 1) -> np.ndarray:
    """Move the probe along a probe-local axis: 1 = image lateral (slide), 0 = beam (advance)."""
    offset = np.eye(4)
    offset[axis, 3] = millimetres * 1e-3
    return base @ offset


def average_pose(poses: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[:3, 3] = poses[:, :3, 3].mean(0)
    result[:3, :3] = Rotation.from_matrix(poses[:, :3, :3]).mean().as_matrix()
    return result


def septum_angle(labels: np.ndarray) -> float | None:
    """Angle (deg) of the septum's principal axis from the image vertical, or None if absent."""
    septum = (labels == 1) & binary_dilation(labels == 5, iterations=3)
    points = np.argwhere(septum).astype(float)
    if len(points) < 50:
        return None
    rows, cols = (points - points.mean(0)).T
    direction = np.linalg.eigh(np.cov(np.stack([rows, cols])))[1][:, -1]
    return math.degrees(math.atan2(direction[1], direction[0]) if direction[0] >= 0 else math.atan2(-direction[1], -direction[0]))


def adjust(base, spec, frame, sector, pivot, marks, mask, spacing_m) -> tuple[np.ndarray, dict]:
    """Apply the view's deterministic yaw / pitch adjustment; returns the pose and what was chosen."""
    chosen = {}
    yaw = spec.get("yaw")
    if yaw == "septum_vertical":
        frame_mask = np.asarray(mask[frame])

        def tilt(degrees):
            angle = septum_angle(label_slice(sector, in_plane(base, pivot, degrees), frame_mask, spacing_m))
            return abs(angle) if angle is not None else np.inf

        best = min(np.arange(-45, 45.1, 1.5), key=tilt)
        chosen = {"yaw_deg": float(best), "septum_angle_before_deg": round(float(tilt(0)), 1), "septum_angle_after_deg": round(float(tilt(best)), 1)}
        base = in_plane(base, pivot, best)
    elif yaw == "plax_frame":
        # Valve level on the centre line (probe slide), LV apex near the sector edge (in-plane
        # rotation about the valve level, so it stays centred); twice, since each step nudges the other.
        valves = (marks["mitral"] + marks["aortic"]) / 2
        local = lambda p, point: p[:3, :3].T @ (point - p[:3, 3])
        edge = sector.fov_degrees / 2 - 5
        total_yaw = total_slide = 0.0
        for _ in range(2):
            offset = float(local(base, valves)[1]) * 1e3
            base, total_slide = slid(base, offset), total_slide + offset
            centre = local(base, valves)
            centre[2] = 0.0

            def apex_miss(degrees):
                apex = local(in_plane(base, centre, degrees), marks["apex"])
                return abs(abs(math.degrees(math.atan2(apex[1], apex[0]))) - edge)

            best = min(np.arange(-90, 90.1, 1.0), key=lambda d: (round(apex_miss(d), 0), abs(d)))
            base, total_yaw = in_plane(base, centre, best), total_yaw + best
        apex = local(base, marks["apex"])
        chosen = {"yaw_deg": round(total_yaw, 1), "slide_mm": round(total_slide, 1),
                  "valve_lateral_mm_after": round(float(local(base, valves)[1]) * 1e3, 1),
                  "apex_angle_deg_after": round(math.degrees(math.atan2(apex[1], apex[0])), 1)}
    elif yaw == "center_valves":
        valves = (marks["mitral"] + marks["aortic"]) / 2

        def lateral(degrees):
            p = in_plane(base, pivot, degrees)
            return abs(float((p[:3, :3].T @ (valves - p[:3, 3]))[1]))

        best = min(np.arange(-45, 45.1, 1.0), key=lateral)
        chosen = {"yaw_deg": float(best), "valve_lateral_mm_before": round(lateral(0) * 1e3, 1), "valve_lateral_mm_after": round(lateral(best) * 1e3, 1)}
        base = in_plane(base, pivot, best)
    if spec.get("slide") == "center_heart":
        labels = label_slice(sector, base, np.asarray(mask[frame]), spacing_m)
        heart = sector.valid & np.isin(labels, views_module.RIGHT_HEART + [1, 2, 3, 9])
        offset = float(sector.local_points[heart][:, 1].mean()) * 1e3
        base = slid(base, offset)
        chosen["slide_mm"] = round(offset, 1)
    pitch = spec.get("pitch")
    if pitch:
        landmark = {"toward_aortic_valve": marks["aortic"], "toward_tricuspid_valve": marks["tricuspid"]}[pitch]
        best = min(np.arange(-40, 40.1, 1.0), key=lambda d: views_module.plane_distance(views_module.pitched(base, d), landmark))
        chosen.update({"pitch_deg": float(best),
                       "plane_distance_mm_before": round(views_module.plane_distance(base, landmark) * 1e3, 1),
                       "plane_distance_mm_after": round(views_module.plane_distance(views_module.pitched(base, best), landmark) * 1e3, 1)})
        base = views_module.pitched(base, best)
    return base, chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    refine = config["refine_views"]
    paths = Paths(config)
    source = np.load(paths.root / refine["source"] / "views.npz")
    source_names = list(source["view_names"])
    out = paths.root / refine["output"]
    out.mkdir(parents=True, exist_ok=True)
    trajectory = load_trajectory(paths)
    sector = create_sector(config)
    info = json.load(open(paths.phantoms / "phantom_info.json"))
    spacing_m = np.array(info["mask_spacing_m"])
    mask = load_cropped_mask(paths)
    end_diastole = np.asarray(mask[0])
    marks = views_module.landmarks(end_diastole, spacing_m, np.array(info["heart_center_m"]))
    shape = np.array(end_diastole.shape)
    world = lambda idx: (np.asarray(idx, float) - shape / 2) * spacing_m
    marks["tricuspid"] = world(np.argwhere((end_diastole == 5) & binary_dilation(end_diastole == 4, iterations=2)).mean(0))
    marks["apex"] = np.array(info["heart_center_m"]) + np.array([10, -27, 55]) * 1e-3  # LV apex, from the long-axis analysis
    pivot = sector.local_points[sector.valid].astype(np.float64).mean(axis=0)
    rng = np.random.default_rng(refine["seed"])
    num_frames = config["cardiac"]["num_frames"]

    def source_index(view, k):
        return int(np.flatnonzero((source["view"] == source_names.index(view)) & (source["sample"] == k))[0])

    poses, view_index, sample, phase, origin, descriptions = [], [], [], [], [], []
    names = list(refine["views"])
    adjustments = {}
    for v, name in enumerate(names):
        spec = refine["views"][name]
        if "accept" in spec:
            i = source_index(name, spec["accept"])
            poses.append(source["poses"][i]); view_index.append(v); sample.append(0)
            phase.append(float(source["cardiac_phase"][i])); origin.append(f"{name} #{spec['accept']}")
            descriptions.append("kept")  # provenance is in views.json ("from")
            continue
        starts = spec["from"] if isinstance(spec["from"], list) else [spec["from"]]
        indices = [source_index(spec.get("from_view", name), k) for k in starts]
        i = indices[0]
        base_frame = int(source["cardiac_frame"][i])
        start = average_pose(source["poses"][indices]) if len(indices) > 1 else source["poses"][i]
        base, chosen = adjust(start, spec, base_frame, sector, pivot, marks, mask, spacing_m)
        adjustments[name] = {"from": f"{spec.get('from_view', name)} #{spec['from']}", **chosen}
        if "grid" in spec:
            # Quasi-random (scrambled Sobol) coverage of rotations about the image centre and probe
            # translations along the lateral / elevation axes; all at the start frame's phase.
            grid = spec["grid"]
            spans = np.array([grid["yaw_deg"], grid["pitch_deg"], grid["roll_deg"], grid["lateral_mm"], grid["elevation_mm"]], float)
            points = qmc.Sobol(d=5, scramble=True, seed=refine["seed"]).random(grid["count"])
            offsets = [np.zeros(5)] + list((2 * points - 1) * spans)
            for k, (d_yaw, d_pitch, d_roll, d_lateral, d_elevation) in enumerate(offsets):
                q = rotated(rotated(in_plane(base, pivot, d_yaw), pivot, 1, d_pitch), pivot, 0, d_roll)
                q = slid(slid(q, d_lateral, axis=1), d_elevation, axis=2)
                poses.append(q); view_index.append(v); sample.append(k)
                phase.append(float(source["cardiac_phase"][i])); origin.append(adjustments[name]["from"])
                descriptions.append(f"y{d_yaw:+.0f} p{d_pitch:+.0f} r{d_roll:+.0f}°, lat{d_lateral:+.0f} el{d_elevation:+.0f}mm" if k else "start")
            continue
        if "steps" in spec:
            # Deterministic progression: sample k adds the k-th (pitch, roll) step about the
            # image centre, all at the start frame's phase so only the pose changes.
            for k, (step_pitch, step_roll) in enumerate([(0.0, 0.0)] + [tuple(step) for step in spec["steps"]]):
                poses.append(rotated(rotated(base, pivot, 1, step_pitch), pivot, 0, step_roll))
                view_index.append(v); sample.append(k)
                phase.append(float(source["cardiac_phase"][i])); origin.append(adjustments[name]["from"])
                descriptions.append(f"+pitch {step_pitch:.1f}°, +roll {step_roll:.1f}°" if k else "start")
            continue
        for k in range(refine["samples"]):
            p = base
            yaw = shift = rot = trans = advance = pitch = roll = 0.0
            if "advance_mm" in spec:
                advance = float(np.mean(spec["advance_mm"])) if k == 0 else rng.uniform(*spec["advance_mm"])
                p = slid(p, advance, axis=0)
            if k:
                if "pitch_range_deg" in spec:
                    pitch = rng.uniform(*spec["pitch_range_deg"])
                    p = rotated(p, pivot, 1, pitch)
                if "roll_range_deg" in spec:
                    roll = rng.uniform(*spec["roll_range_deg"])
                    p = rotated(p, pivot, 0, roll)
                if "yaw_range_deg" in spec:
                    yaw = rng.uniform(*spec["yaw_range_deg"])
                    p = in_plane(p, pivot, yaw)
                if "lateral_shift_mm" in spec:
                    shift = rng.uniform(*spec["lateral_shift_mm"])
                    p = slid(p, shift)
                rot = rng.uniform(0, spec.get("rotation_max_deg", 0))
                trans = rng.uniform(0, spec.get("translation_max_mm", 0))
                p = views_module.perturb(p, pivot, rot, trans, rng)
            poses.append(p); view_index.append(v); sample.append(k)
            # Sample 0 keeps the source frame's phase; the rest draw a new one.
            phase.append(float(source["cardiac_phase"][i]) if k == 0 else float(rng.uniform(0, 1)))
            origin.append(adjustments[name]["from"])
            descriptions.append(", ".join(x for x in (f"pitch {pitch:+.0f}°" if pitch else "", f"roll {roll:+.0f}°" if roll else "",
                                                       f"yaw {yaw:+.0f}°" if yaw else "", f"slide {shift:+.0f}mm" if shift else "",
                                                       f"fwd {advance:.0f}mm" if advance else "",
                                                       f"{rot:.0f}°/{trans:.0f}mm" if (rot or trans) else "") if x) or "centre")

    poses = np.array(poses)
    phase = np.array(phase)
    frame = np.round(phase * num_frames).astype(int) % num_frames
    candidates = {"poses": poses, "view": np.array(view_index), "sample": np.array(sample), "cardiac_phase": phase, "cardiac_frame": frame}
    stride = candidates_module.POINT_STRIDE
    local = sector.local_points[::stride, ::stride][sector.valid[::stride, ::stride]].astype(np.float64)
    rows = candidates_module.novelty(poses, trajectory["poses"], local)
    titles = [f"{SHORT_NAMES[names[v]]} #{k} (f{frame[i]}, {descriptions[i]})" for i, (v, k) in enumerate(zip(view_index, sample))]
    # Each view starts a new grid row (and takes as many rows as it needs), so views never share a row.
    per_row = refine["samples"]
    counts = np.bincount(view_index, minlength=len(names))
    first_row = np.concatenate([[0], np.cumsum(-(-counts // per_row))])[:-1]
    hearts = candidates_module.save_panels(candidates, rows, trajectory, sector, mask, spacing_m, out / "all_views.png",
                                           titles=titles, pairs_per_row=per_row,
                                           slots=[first_row[v] * per_row + k for v, k in zip(view_index, sample)])
    for v, name in enumerate(names):
        keep = np.flatnonzero(np.array(view_index) == v)
        subset = {key: value[keep] for key, value in candidates.items()}
        candidates_module.save_panels(subset, [rows[i] for i in keep], trajectory, sector, mask, spacing_m, out / f"view_{name}.png",
                                      titles=[titles[i] for i in keep], pairs_per_row=5)
    for i, row in enumerate(rows):
        row.update({"view": names[view_index[i]], "sample": sample[i], "from": origin[i], "perturbation": descriptions[i],
                    "cardiac_frame": int(frame[i]), "heart_fraction": round(hearts[i], 3)})
    np.savez(out / "views.npz", view_names=np.array(names), **candidates)
    save_json({"settings": refine, "adjustments": adjustments, "candidates": rows}, out / "views.json")
    for name, chosen in adjustments.items():
        print(f"{name:34s} {json.dumps(chosen)}")
    for v, name in enumerate(names):
        keep = [r for r in rows if r["view"] == name]
        print(f"{SHORT_NAMES[name]:9s} {len(keep):2d} samples  nearest {min(r['nearest_distance_mm'] for r in keep):5.1f}-{max(r['nearest_distance_mm'] for r in keep):5.1f} mm  "
              f"heart {min(r['heart_fraction'] for r in keep) * 100:3.0f}-{max(r['heart_fraction'] for r in keep) * 100:3.0f}%")


if __name__ == "__main__":
    main()
