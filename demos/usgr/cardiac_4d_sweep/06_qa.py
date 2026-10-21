"""Compare reconstructed images against the label slices they should show.

    python 06_qa.py [--config config.yaml] [--every 50]

For a sample of images, plots the reconstruction next to the label slice cut by the
same pose and cardiac frame, with label boundaries drawn over the B-mode. A flip,
mirror, or depth offset in the pose or scan-conversion conventions shows up as
boundaries that don't line up with the anatomy.
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from common import Paths, create_sector, label_slice, load_config, load_cropped_mask, load_trajectory  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--every", type=int, default=50)
    args = parser.parse_args()

    config = load_config(args.config)
    paths = Paths(config)
    paths.qa.mkdir(parents=True, exist_ok=True)
    trajectory = load_trajectory(paths)
    mask = load_cropped_mask(paths)
    with (paths.phantoms / "phantom_info.json").open() as handle:
        spacing_m = np.array(json.load(handle)["mask_spacing_m"])
    sector = create_sector(config)

    with (paths.experiments / "image_index.json").open() as handle:
        simulated = sorted(entry["image"] for entry in json.load(handle)["images"])
    picks = simulated[:: args.every]
    figure, axes = plt.subplots(len(picks), 2, figsize=(6, 3 * len(picks)), squeeze=False)
    for (left, right), index in zip(axes, picks):
        frame = trajectory["cardiac_frame"][index]
        labels = label_slice(sector, trajectory["poses"][index], mask[frame], spacing_m)
        bmode = np.asarray(Image.open(paths.recording / "images" / f"{index:05d}.png"))
        left.imshow(bmode, cmap="gray")
        left.contour(labels, levels=np.unique(labels)[1:] - 0.5, colors="r", linewidths=0.4)
        right.imshow(labels, cmap="tab10", interpolation="nearest")
        left.set_title(f"image {index} (phase {frame})", fontsize=8)
        for axis in (left, right):
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(paths.qa / "recon_vs_labels.png", dpi=120)
    print(f"wrote {paths.qa / 'recon_vs_labels.png'}")


if __name__ == "__main__":
    main()
