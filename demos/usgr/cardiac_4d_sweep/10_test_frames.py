"""Novel-view test frames at the chosen canonical views, simulated as their own recording.

    python 10_test_frames.py select        # picks (config `test_set`) -> test_frames/
    python 10_test_frames.py setup         # one experiment per CT frame -> test_experiments/
    python 10_test_frames.py run           # simulate (settings from `simulation`)
    python 10_test_frames.py reconstruct   # -> recording_test/ (images, metadata, QA panel)

The training experiments and recording are untouched. Reconstruction uses the training
recording's log-compression reference, so test and training brightness are comparable.
Test timestamps continue after the last training frame at `frame_interval_s`, as in
`synthetic/generate.py`; each frame keeps the cardiac phase it was chosen at.
"""

import argparse
import importlib
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from common import Paths, create_sector, label_slice, load_config, load_cropped_mask, load_trajectory, save_json  # noqa: E402

from musik import experiment  # noqa: E402

setup_module = importlib.import_module("03_setup_experiments")
reconstruct_module = importlib.import_module("05_reconstruct")


def test_paths(config: dict) -> Paths:
    """Paths whose experiments and recording point at the test-frame locations."""
    paths = Paths(config)
    paths.experiments = paths.root / "test_experiments"
    paths.recording = paths.root / "recording_test"
    return paths


def select(config: dict) -> None:
    paths = Paths(config)
    test = config["test_set"]
    source = np.load(paths.root / test["source"] / "views.npz")
    rows = json.load(open(paths.root / test["source"] / "views.json"))["candidates"]
    names = list(source["view_names"])
    keep = []
    for name, k in test["picks"].items():
        i = int(np.flatnonzero((source["view"] == names.index(name)) & (source["sample"] == k))[0])
        keep.append(i)
    out = paths.root / "test_frames"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "test_frames.npz", view_names=np.array(list(test["picks"])), poses=source["poses"][keep],
             cardiac_phase=source["cardiac_phase"][keep], cardiac_frame=source["cardiac_frame"][keep])
    save_json({"source": test["source"], "frames": [
        {"test_index": t, "view": name, "source_sample": k, "from": rows[i]["from"], "cardiac_frame": int(source["cardiac_frame"][i]),
         "nearest_train": rows[i]["nearest_train"], "nearest_distance_mm": rows[i]["nearest_distance_mm"],
         "nearest_angle_deg": rows[i]["nearest_angle_deg"], "support_2mm": rows[i]["support_2mm"]}
        for t, ((name, k), i) in enumerate(zip(test["picks"].items(), keep))
    ]}, out / "test_frames.json")
    print(f"selected {len(keep)} test frames -> {out}")


def setup(config: dict) -> None:
    paths = test_paths(config)
    frames = np.load(paths.root / "test_frames" / "test_frames.npz")
    ray_num = config["transducer"]["ray_num"]
    index = {"ray_num": ray_num, "images": []}
    for frame in np.unique(frames["cardiac_frame"]):
        images = np.flatnonzero(frames["cardiac_frame"] == frame)
        built = setup_module.build_experiment(int(frame), frames["poses"][images], config, paths)
        built.save()
        print(f"phase {frame:02d}: {len(images)} test frames, {len(built)} simulations -> {built.simulation_path}")
        index["images"] += [{"image": int(image), "phase": int(frame), "transducer": k, "first_simulation": k * ray_num}
                            for k, image in enumerate(images)]
    index["images"].sort(key=lambda entry: entry["image"])
    save_json(index, paths.experiments / "image_index.json")


def run(config: dict) -> None:
    paths = test_paths(config)
    sim = config["simulation"]
    for path in sorted(paths.experiments.glob("phase_*")):
        loaded = experiment.Experiment.load(str(path))
        print(f"{path.name}: {len(loaded.indices_to_run())} of {len(loaded)} simulations remaining")
        loaded.run(prep_workers=sim["prep_workers"], sim_workers=sim["sim_workers"], gpus=sim["gpus"])


def reconstruct(config: dict) -> None:
    paths = test_paths(config)
    training = json.load(open(Paths(config).recording / "sequence_metadata.json"))
    reference = training["generation"]["log_reference"]
    frames = np.load(paths.root / "test_frames" / "test_frames.npz")
    names = list(frames["view_names"])
    count = len(names)
    image_index = json.load(open(paths.experiments / "image_index.json"))["images"]
    recon = config["reconstruction"]
    depths_m = np.linspace(0.0, config["acquisition"]["depth_mm"] * 1e-3, recon["num_depth_samples"])
    angles = reconstruct_module.ray_angles(config)

    polar = np.zeros((count, len(angles), len(depths_m)), dtype=np.float32)
    valid = np.zeros(count, dtype=bool)
    for frame in sorted({entry["phase"] for entry in image_index}):
        entries = [entry for entry in image_index if entry["phase"] == frame]
        rows = [entry["image"] for entry in entries]
        polar[rows], valid[rows] = reconstruct_module.beamform_phase(frame, entries, depths_m, config, paths)
    paths.recording.mkdir(parents=True, exist_ok=True)
    np.save(paths.recording / "polar.npy", polar)

    sector = create_sector(config)
    floor_db = -recon["dynamic_range_db"]
    (paths.recording / "images").mkdir(exist_ok=True)
    for t in range(count):
        decibels = 20.0 * np.log10(reconstruct_module.scan_convert(polar[t], sector, angles, depths_m) / reference + 1e-12)
        pixels = np.clip((decibels - floor_db) / -floor_db, 0.0, 1.0) * 255.0
        pixels[~sector.valid] = 0.0
        Image.fromarray(pixels.astype(np.uint8)).save(paths.recording / "images" / f"{t:05d}.png")

    last_training_time = load_trajectory(Paths(config))["times_s"][-1]
    test_trajectory = {
        "times_s": last_training_time + config["test_set"]["frame_interval_s"] * np.arange(1, count + 1),
        "poses": frames["poses"], "cardiac_phase": frames["cardiac_phase"], "cardiac_frame": frames["cardiac_frame"],
        "section": np.arange(count),  # one "section" per view; names in generation.test_views
    }
    selection = json.load(open(paths.root / "test_frames" / "test_frames.json"))
    reconstruct_module.write_metadata(paths, config, sector, test_trajectory, valid, list(range(count)), {
        "split": "test", "log_reference": float(reference), "log_reference_source": "training recording",
        "test_views": names, "selection": selection,
    })

    # QA: each reconstructed test frame against the labels its pose and phase should show.
    mask = load_cropped_mask(paths)
    spacing_m = np.array(json.load(open(paths.phantoms / "phantom_info.json"))["mask_spacing_m"])
    figure, axes = plt.subplots(2, count, figsize=(2.6 * count, 5.6))
    for t, name in enumerate(names):
        labels = label_slice(sector, frames["poses"][t], np.asarray(mask[frames["cardiac_frame"][t]]), spacing_m)
        image = np.asarray(Image.open(paths.recording / "images" / f"{t:05d}.png"))
        axes[0, t].imshow(image, cmap="gray")
        axes[0, t].contour(labels, levels=np.unique(labels)[1:] - 0.5, colors="r", linewidths=0.4)
        axes[0, t].set_title(f"{t}: {name}", fontsize=7)
        axes[1, t].imshow(np.ma.masked_where(~sector.valid, labels), cmap="tab10", vmin=0, vmax=int(np.asarray(mask[0]).max()), interpolation="nearest")
        for axis in axes[:, t]:
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(paths.recording / "test_qa.png", dpi=90)
    print(f"wrote {count} test images ({int(valid.sum())} valid) to {paths.recording}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["select", "setup", "run", "reconstruct"])
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    {"select": select, "setup": setup, "run": run, "reconstruct": reconstruct}[args.step](config)


if __name__ == "__main__":
    main()
