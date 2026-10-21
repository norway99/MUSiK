"""Create one MUSiK experiment per cardiac frame from the planned trajectory.

    python 03_setup_experiments.py [--config config.yaml] [--frames 0 1 2]

Images are binned by their nearest CT frame. The experiment for frame p holds that
frame's phantom and one transducer per image in the bin, posed at that image's probe
pose. MUSiK numbers simulations by transducer and then by ray, so image k of phase p
owns simulation indices [k * ray_num, (k + 1) * ray_num). `image_index.json` records
that mapping for reconstruction.
"""

import argparse

import numpy as np

from common import Paths, load_config, load_trajectory, save_json, to_musik_transform

from musik import experiment, phantom, sensor, simulation, transducer, transducer_set


def make_transducer(config: dict, c0: float) -> transducer.Focused:
    t = config["transducer"]
    probe = transducer.Focused(
        max_frequency=t["max_frequency"],
        elements=t["elements"],
        width=t["width_mm"] * 1e-3,
        height=t["height_mm"] * 1e-3,
        sensor_sampling_scheme=t["sensor_sampling_scheme"],
        sweep=np.radians(config["acquisition"]["fov_degrees"]),
        ray_num=t["ray_num"],
        imaging_ndims=2,
        focus_azimuth=t["focus_azimuth_mm"] * 1e-3,
        focus_elevation=t["focus_elevation_mm"] * 1e-3,
        cycles=t["cycles"],
    )
    probe.make_sensor_coords(c0)
    return probe


def make_sim_properties(config: dict) -> simulation.SimProperties:
    s = config["simulation"]
    return simulation.SimProperties(
        grid_size=tuple(np.array(s["grid_size_mm"]) * 1e-3),
        PML_size=tuple(s["PML_size"]),
        PML_alpha=s["PML_alpha"],
        bona=s["bona"],
        alpha_coeff=s["alpha_coeff"],
        alpha_power=s["alpha_power"],
    )


def build_experiment(frame: int, poses: np.ndarray, config: dict, paths: Paths) -> experiment.Experiment:
    frame_phantom = phantom.Phantom.load(paths.phantom(frame))
    c0 = frame_phantom.baseline[0]
    # One transducer object per pose: MUSiK stores sensor coordinates per transducer.
    transducers = [make_transducer(config, c0) for _ in poses]
    probes = transducer_set.TransducerSet(transducers, poses=[to_musik_transform(p) for p in poses], seed=config["phantom"]["seed"])
    return experiment.Experiment(
        simulation_path=str(paths.experiment(frame)),
        sim_properties=make_sim_properties(config),
        phantom=frame_phantom,
        transducer_set=probes,
        sensor=sensor.Sensor(transducer_set=probes, aperture_type="transmit_as_receive"),
        nodes=config["simulation"]["nodes"],
        workers=config["simulation"]["workers"],
        additional_keys=[],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--frames", type=int, nargs="*", default=None)
    parser.add_argument("--images", type=int, nargs="*", default=None, help="only these trajectory images (pilots)")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = Paths(config)
    trajectory = load_trajectory(paths)
    ray_num = config["transducer"]["ray_num"]
    selected = np.zeros(len(trajectory["times_s"]), dtype=bool)
    selected[args.images if args.images else slice(None)] = True

    index = {"ray_num": ray_num, "images": []}
    frames = args.frames if args.frames else range(config["cardiac"]["num_frames"])
    for frame in frames:
        images = np.flatnonzero((trajectory["cardiac_frame"] == frame) & selected)
        if len(images) == 0:
            print(f"phase {frame:02d}: no images, skipping")
            continue
        built = build_experiment(frame, trajectory["poses"][images], config, paths)
        built.save()
        print(f"phase {frame:02d}: {len(images)} images, {len(built)} simulations -> {built.simulation_path}")
        index["images"] += [
            {"image": int(image), "phase": frame, "transducer": k, "first_simulation": k * ray_num}
            for k, image in enumerate(images)
        ]

    # Only complete when every frame was set up in one call; partial runs overwrite it.
    index["images"].sort(key=lambda entry: entry["image"])
    save_json(index, paths.experiments / "image_index.json")


if __name__ == "__main__":
    main()
