"""Reconstruct every simulated image and write the recording (PNGs + sequence_metadata.json).

    python 05_reconstruct.py [--config config.yaml] [--recompute-polar]

Stage 1, per phase experiment: beamform each scan line with the transducer's own
`make_scan_line` + `preprocess` (TGC, band-pass, envelope) and resample it onto a fixed
depth grid. The result is `polar.npy` with shape (num_images, ray_num, num_depth_samples)
in global image order. Images with missing simulations are marked invalid.

Stage 2: log-compress against one global reference, scan-convert each image into the
512x512 sector, and write `recording/images/%05d.png` plus v3 metadata mirroring
`synthetic/generate.py:write_metadata`.
"""

import argparse
import json
from datetime import datetime, timezone

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

from common import Paths, create_sector, load_config, load_trajectory, pose_matrix_to_12, save_json

from musik import experiment


def ray_angles(config: dict) -> np.ndarray:
    """Steering angle of each scan line, matching `Transducer.make_ray_transforms` for 2D."""
    half = np.radians(config["acquisition"]["fov_degrees"]) / 2.0
    return np.linspace(-half, half, config["transducer"]["ray_num"])


def beamform_phase(frame: int, entries: list[dict], depths_m: np.ndarray, config: dict, paths: Paths):
    """Polar envelopes (len(entries), ray_num, depth) for one phase, plus per-image validity."""
    loaded = experiment.Experiment.load(str(paths.experiment(frame)))
    # A dry run rebuilds the receive-side state that `make_scan_line` needs; the transducers
    # are identical, so one ray per transducer is enough.
    loaded.run(dry=True, dry_fast=True)
    available = set(loaded.results.indices())
    c0 = loaded.phantom.baseline[0]
    ray_num = config["transducer"]["ray_num"]

    polar = np.zeros((len(entries), ray_num, len(depths_m)), dtype=np.float32)
    valid = np.ones(len(entries), dtype=bool)
    for row, entry in enumerate(entries):
        probe = loaded.transducer_set.transducers[entry["transducer"]]
        indices = range(entry["first_simulation"], entry["first_simulation"] + ray_num)
        if not available.issuperset(indices):
            valid[row] = False
            continue
        for ray, index in enumerate(indices):
            times, signals = loaded.results[index][:2]
            line = probe.preprocess(
                probe.make_scan_line(signals, True), times, loaded.sim_properties,
                attenuation_factor=config["reconstruction"]["tgc"],
            )
            # Time zero is the start of the transmit pulse; shift by half its length, as
            # `Transducer.gain_compensation` does.
            pulse_offset = probe.get_pulse().shape[-1] * (times[-1] / times.shape[0]) / 2.0
            ranges = c0 * (times / 2.0 - pulse_offset)
            polar[row, ray] = np.interp(depths_m, ranges, line, left=0.0, right=0.0)
    return polar, valid


def scan_convert(polar: np.ndarray, sector, angles: np.ndarray, depths_m: np.ndarray) -> np.ndarray:
    """Bilinear (angle, depth) lookup for every valid sector pixel; zero outside."""
    angle_index = (sector.angle[sector.valid] - angles[0]) / (angles[1] - angles[0])
    depth_index = sector.radius[sector.valid] / (depths_m[1] - depths_m[0])
    image = np.zeros(sector.image_size, dtype=np.float32)
    image[sector.valid] = map_coordinates(polar, [angle_index, depth_index], order=1, mode="constant", cval=0.0)
    return image


def write_metadata(paths: Paths, config: dict, sector, trajectory: dict, valid: np.ndarray, images: list, extra: dict) -> None:
    """`sequence_metadata.json` in the v3.0 layout of `synthetic/generate.py:write_metadata`."""
    heart_hz = config["cardiac"]["heart_rate_bpm"] / 60.0
    trajectory = {key: value[images] for key, value in trajectory.items()}
    valid = valid[images]
    frames = [
        {
            "image_path": f"images/{image:05d}.png",
            "timestamp_ns": int(trajectory["times_s"][index] * 1e9),
            "microns_per_pixel": sector.microns_per_pixel,
            # TODO: raw tracker pose needs usgr's calibration (compute_tracker_from_transducer).
            "raw_tracker_pose": None,
            "valid": bool(valid[index]),
            "ground_truth_transducer_pose": pose_matrix_to_12(trajectory["poses"][index]),
            "ground_truth_frequency_hz": heart_hz,
        }
        for index, image in enumerate(images)
    ]
    metadata = {
        "format_version": "3.0",
        "sequence_type": "simulated",
        "frames": frames,
        "geometry": {
            "default": {
                "image_size": list(sector.image_size),
                "fov_degrees": sector.fov_degrees,
                "sector_apex_pixels": list(sector.apex_pixels),
                "sector_radius_pixels": sector.radius_pixels,
                "probe_element": "phased",
            }
        },
        "calibration": None,  # TODO: same calibration reference as synthetic recordings
        "source": {
            "recording_time": datetime.now(timezone.utc).isoformat(),
            "num_frames": len(frames),
            "tracking_device": "simulated",
        },
        "time": {"cardiac_period_seconds": 1.0 / heart_hz},
        "generation": {
            "config": config,
            "trajectory_index": list(images),
            "true_poses": [pose_matrix_to_12(p) for p in trajectory["poses"]],
            "cardiac_phase": trajectory["cardiac_phase"].tolist(),
            "cardiac_frame": trajectory["cardiac_frame"].tolist(),
            "section": trajectory["section"].tolist(),
            **extra,
        },
    }
    save_json(metadata, paths.recording / "sequence_metadata.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--recompute-polar", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = Paths(config)
    recon = config["reconstruction"]
    trajectory = load_trajectory(paths)
    with (paths.experiments / "image_index.json").open() as handle:
        image_index = json.load(handle)["images"]
    num_images = len(trajectory["times_s"])
    depths_m = np.linspace(0.0, config["acquisition"]["depth_mm"] * 1e-3, recon["num_depth_samples"])
    angles = ray_angles(config)

    polar_path, valid_path = paths.recording / "polar.npy", paths.recording / "valid.npy"
    paths.recording.mkdir(parents=True, exist_ok=True)
    if args.recompute_polar or not polar_path.exists():
        polar = np.zeros((num_images, len(angles), len(depths_m)), dtype=np.float32)
        valid = np.zeros(num_images, dtype=bool)
        for frame in range(config["cardiac"]["num_frames"]):
            entries = [e for e in image_index if e["phase"] == frame]
            if not entries:
                continue
            rows = [e["image"] for e in entries]
            polar[rows], valid[rows] = beamform_phase(frame, entries, depths_m, config, paths)
            print(f"phase {frame:02d}: beamformed {len(entries)} images ({int(valid[rows].sum())} complete)")
        np.save(polar_path, polar)
        np.save(valid_path, valid)
    polar, valid = np.load(polar_path), np.load(valid_path)

    # One reference for the whole sweep keeps brightness comparable across images and phases.
    reference = np.percentile(polar[valid], recon["reference_percentile"])
    floor_db = -recon["dynamic_range_db"]
    sector = create_sector(config)
    images_dir = paths.recording / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    # Only images that were set up (all of them, unless 03 was given --images).
    images = sorted(entry["image"] for entry in image_index)
    for index in images:
        decibels = 20.0 * np.log10(scan_convert(polar[index], sector, angles, depths_m) / reference + 1e-12)
        pixels = np.clip((decibels - floor_db) / -floor_db, 0.0, 1.0) * 255.0
        pixels[~sector.valid] = 0.0
        Image.fromarray(pixels.astype(np.uint8)).save(images_dir / f"{index:05d}.png")

    write_metadata(paths, config, sector, trajectory, valid, images, {"log_reference": float(reference)})
    print(f"wrote {len(images)} images ({int(valid[images].sum())} valid) to {paths.recording}")


if __name__ == "__main__":
    main()
