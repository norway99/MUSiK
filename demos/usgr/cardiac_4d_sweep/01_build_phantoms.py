"""Build one MUSiK phantom per cardiac frame from the 4D label mask.

    python 01_build_phantoms.py [--config config.yaml] [--frames 0 1 2] [--mask-only]

The raw mask is cropped to the union of all labels over the cycle plus
`phantom.margin_mm` on every side (padding with background where that runs past the
scan). The probe has to sit inside the grid: MUSiK clamps lookups at the edges, which
replicates baked speckle into streaks. The cropped mask is cached as `phantoms/mask.npy`
for the later scripts. Each frame is then resampled (nearest neighbour) to
`phantom.voxel_size_mm`, labels are mapped to tissues, and the phantom is saved to
`phantoms/frame_XX`. All frames share a seed. With `bake_speckle`, the property volume
is generated once here, so every scan line in that phase samples the same scatterers.
"""

import argparse

import numpy as np

from common import Paths, heart_center_m, load_config, load_cropped_mask, load_mask, save_json

from musik import phantom, tissue


def crop_with_margin(mask: np.ndarray, spacing_m: np.ndarray, margin_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Crop (T, ...) to the labelled region of any frame plus a margin; returns the crop and its origin index."""
    occupied = np.argwhere(mask.any(axis=0))
    margin = np.ceil(margin_m / spacing_m).astype(int)
    lo = occupied.min(axis=0) - margin
    hi = occupied.max(axis=0) + 1 + margin
    cropped = np.zeros((mask.shape[0], *(hi - lo)), dtype=mask.dtype)
    src_lo, src_hi = np.maximum(lo, 0), np.minimum(hi, mask.shape[1:])
    dst = tuple(slice(a, b) for a, b in zip(src_lo - lo, src_hi - lo))
    src = tuple(slice(a, b) for a, b in zip(src_lo, src_hi))
    cropped[(slice(None), *dst)] = mask[(slice(None), *src)]
    return cropped, lo


def make_tissues(config: dict) -> dict:
    """Tissue objects keyed by name; label 0 is MUSiK's water, so labels start at 1."""
    return {
        name: tissue.Tissue(name=name, label=index + 1, **properties)
        for index, (name, properties) in enumerate(config["tissues"].items())
    }


def resample_nearest(labels: np.ndarray, spacing_m: np.ndarray, voxel_m: float) -> np.ndarray:
    """Nearest-neighbour resample that keeps both grids centered, as MUSiK places them in the world."""
    out_shape = np.round(np.array(labels.shape) * spacing_m / voxel_m).astype(int)
    index = [
        np.clip(np.round((np.arange(n) - n / 2) * voxel_m / s + m / 2).astype(int), 0, m - 1)
        for n, m, s in zip(out_shape, labels.shape, spacing_m)
    ]
    return labels[np.ix_(*index)]


def build_phantom(mask_frame: np.ndarray, config: dict, tissues: dict) -> phantom.Phantom:
    spacing_m = np.array(config["mask"]["spacing_mm"]) * 1e-3
    voxel_m = config["phantom"]["voxel_size_mm"] * 1e-3
    labels = resample_nearest(mask_frame, spacing_m, voxel_m)

    default = tissues[config["default_tissue"]]
    result = phantom.Phantom(
        voxel_dims=(voxel_m, voxel_m, voxel_m),
        matrix_dims=labels.shape,
        baseline=(default.c, default.rho),
        seed=config["phantom"]["seed"],
    )
    for value, name in config["mask"]["labels"].items():
        result.add_tissue(tissues[name], mask=labels == int(value))
    # Anything left unlabeled (including mask values not in the label map) becomes the default.
    result.set_default_tissue(default)

    if config["phantom"]["bake_speckle"]:
        result.get_complete()
        # With `from_mask` off, `interpolate_phantom` samples the baked volume instead of
        # regenerating tissue for each simulation.
        result.from_mask = False
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--frames", type=int, nargs="*", default=None)
    parser.add_argument("--mask-only", action="store_true", help="only crop and cache the mask")
    parser.add_argument("--reuse-mask", action="store_true", help="skip reading the raw mask and use the cached crop")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = Paths(config)
    paths.phantoms.mkdir(parents=True, exist_ok=True)
    spacing_m = np.array(config["mask"]["spacing_mm"]) * 1e-3
    tissues = make_tissues(config)

    if args.reuse_mask:
        # The cached crop already fixes the margin; only the tissue build reruns.
        mask = load_cropped_mask(paths)
    else:
        mask, crop_origin = crop_with_margin(load_mask(config), spacing_m, config["phantom"]["margin_mm"] * 1e-3)
        np.save(paths.phantoms / "mask.npy", mask)
        print(f"cropped mask {mask.shape[1:]} ({np.round(np.array(mask.shape[1:]) * spacing_m * 1e3).tolist()} mm)")
        write_info(mask, crop_origin, spacing_m, config, tissues, paths)
    if args.mask_only:
        return

    frames = args.frames if args.frames else range(mask.shape[0])
    for frame in frames:
        built = build_phantom(np.asarray(mask[frame]), config, tissues)
        built.save(paths.phantom(frame))
        print(f"frame {frame:02d}: {built.matrix_dims.tolist()} voxels -> {paths.phantom(frame)}")


def write_info(mask: np.ndarray, crop_origin: np.ndarray, spacing_m: np.ndarray, config: dict, tissues: dict, paths: Paths) -> None:
    """Values every later script needs without reloading the full mask."""
    centers = np.array([heart_center_m(mask[f], spacing_m, config["mask"]["heart_labels"]) for f in range(mask.shape[0])])
    save_json(
        {
            "mask_shape": list(mask.shape),
            "mask_spacing_m": spacing_m.tolist(),
            "crop_origin_index": crop_origin.tolist(),
            "label_counts": {int(v): int(c) for v, c in zip(*np.unique(mask[0], return_counts=True))},
            "heart_center_per_frame_m": centers.tolist(),
            # Trajectory targets are relative to the cycle-averaged center.
            "heart_center_m": centers.mean(axis=0).tolist(),
            "tissue_labels": {name: t.label for name, t in tissues.items()},
        },
        paths.phantoms / "phantom_info.json",
    )


if __name__ == "__main__":
    main()
