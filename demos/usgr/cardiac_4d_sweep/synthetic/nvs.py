"""E3 novel-view readouts: novelty and support, simple baselines, and comparison panels.

**Novelty** of a test frame, on one reference trajectory (truth where the recording has
it, otherwise the run's fitted training poses and registered test pose):
- nearest training frame by ordered patch distance — mean distance between the same
  token centres under the two poses (mm) — with phase unconstrained, and among
  phase-compatible frames (|phase gap| < `phase_tolerance`). Each training frame is also
  considered rolled 180° about the probe's depth axis (`flip_poses`): that view is the
  same image mirrored left-right, so it is not novel. `*_flipped` says which matched;
- plane angle, elapsed time and phase gap to that frame;
- support: the fraction of the test sector within 1 / 2 / 5 mm of any training frame's
  token centres, which can be dense even when no single frame is close.

**Baselines**, built only from what a method without truth has (the run's fitted
training trajectory and the registered test query), on the renderer's output grid:
- `copy_nearest`: the recorded image of the spatially nearest training frame, mirrored
  left-right when the nearer match is its flipped pose;
- `copy_phase`: the nearest phase-compatible training frame, likewise;
- `compound`: phase-windowed voxel means of recorded intensity over the fitted
  trajectory (`TrainedRecording.intensity_map`), sampled at the query's output pixels.
  Unsupported pixels take the mean of the supported ones; `compound_coverage` records
  how many were supported.

All methods are scored on the same mask as the model's render, with no test-dependent
intensity fitting.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from ..information import DEFAULT_PHASE_WINDOW  # noqa: E402
from ..rendering.intensity import pixel_offsets, pixel_points  # noqa: E402
from .scoring import image_points, wrap  # noqa: E402

PHASE_TOLERANCE = 0.5  # rad, for phase-compatible neighbours
SUPPORT_MM = (1.0, 2.0, 5.0)
# 180-degree roll about the probe-local depth axis (x: rows from the apex). The sector is
# centred on that axis (centered scan conversion), so the rolled view is the same image
# mirrored left-right: R -> R @ diag(1, -1, -1), the flip `guidance` uses for mirrored
# references.
_FLIP = torch.diag(torch.tensor([1.0, -1.0, -1.0], dtype=torch.float64))


def flip_poses(poses: torch.Tensor) -> torch.Tensor:
    """`[N, 12]` poses rolled 180 degrees about their depth axis (translation unchanged)."""
    rotation = poses[:, :9].double().reshape(-1, 3, 3) @ _FLIP.to(poses.device)
    return torch.cat([rotation.reshape(-1, 9), poses[:, 9:].double()], dim=-1).to(poses.dtype)


def _rotation_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Angle between the rotations of `[12]` pose `a` and `[N, 12]` poses `b`."""
    ra = a[:9].reshape(3, 3).double()
    rb = b[:, :9].reshape(-1, 3, 3).double()
    trace = torch.einsum("ij,nij->n", ra, rb)
    return torch.rad2deg(torch.arccos(((trace - 1) / 2).clamp(-1, 1)))


def novelty(
    projector,
    query_poses: torch.Tensor,
    query_phases: torch.Tensor | None,
    query_times: torch.Tensor,
    query_scales: torch.Tensor,
    train_poses: torch.Tensor,
    train_phases: torch.Tensor | None,
    train_times: torch.Tensor,
    train_scales: torch.Tensor,
    phase_tolerance: float = PHASE_TOLERANCE,
) -> tuple[list[dict], np.ndarray]:
    """Per-query novelty rows (training indices are positions in `train_*`), and the
    per-token distance (mm) from each query's token centres to the nearest training one.

    Distances take the nearer of each training frame and its flipped pose."""
    train_points = image_points(projector, train_poses, train_scales).float()      # [N, P, 3]
    flipped_poses = flip_poses(train_poses)
    flipped_points = image_points(projector, flipped_poses, train_scales).float()  # [N, P, 3]
    query_points = image_points(projector, query_poses, query_scales).float()      # [Q, P, 3]
    # A flipped frame samples the same physical plane, so support needs only one of them.
    tree = cKDTree(train_points.reshape(-1, 3).cpu().numpy())
    rows, support_distance = [], []
    for q in range(query_poses.shape[0]):
        direct = (train_points - query_points[q]).norm(dim=-1).mean(-1) * 1e3      # [N]
        rolled = (flipped_points - query_points[q]).norm(dim=-1).mean(-1) * 1e3    # [N]
        flipped = rolled < direct
        distance = torch.minimum(direct, rolled)
        nearest = int(distance.argmin())
        matched = flipped_poses if flipped[nearest] else train_poses
        row = {
            "nearest_train": nearest,
            "nearest_flipped": bool(flipped[nearest]),
            "nearest_distance_mm": float(distance[nearest]),
            "nearest_angle_deg": float(_rotation_deg(query_poses[q], matched[nearest : nearest + 1])),
            "nearest_elapsed_s": float(query_times[q] - train_times[nearest]),
        }
        if query_phases is not None and train_phases is not None:
            gap = wrap(train_phases.double() - float(query_phases[q])).abs()
            row["nearest_phase_gap_rad"] = float(gap[nearest])
            compatible = gap < phase_tolerance
            if compatible.any():
                masked = torch.where(compatible.to(distance.device), distance, torch.inf)
                index = int(masked.argmin())
                row["phase_compatible_train"] = index
                row["phase_compatible_flipped"] = bool(flipped[index])
                row["phase_compatible_distance_mm"] = float(distance[index])
            else:
                row["phase_compatible_train"] = None
                row["phase_compatible_flipped"] = None
                row["phase_compatible_distance_mm"] = None
        nearest_mm = tree.query(query_points[q].cpu().numpy())[0] * 1e3
        for radius in SUPPORT_MM:
            row[f"support_{radius:g}mm"] = float((nearest_mm <= radius).mean())
        row["support_median_mm"] = float(np.median(nearest_mm))
        rows.append(row)
        support_distance.append(nearest_mm)
    return rows, np.stack(support_distance)


class OutputGrid:
    """Maps `[T, K]` per-token output pixels (`frame_pixels` order) to `[H, W]` images."""

    def __init__(self, model):
        projector = model.projector
        self.factor = int(model.pixels_per_token)
        grid_h, grid_w = projector.grid_size
        self.shape = (grid_h * self.factor, grid_w * self.factor)
        tokens = projector.valid_token_index.cpu()
        sub = torch.arange(self.factor)
        rows = tokens // grid_w
        cols = tokens % grid_w
        self.rows = (rows[:, None] * self.factor + sub.repeat_interleave(self.factor)[None]).reshape(-1)
        self.cols = (cols[:, None] * self.factor + sub.repeat(self.factor)[None]).reshape(-1)
        self.token_rows, self.token_cols = rows, cols

    def image(self, values: torch.Tensor, fill: float = float("nan")) -> torch.Tensor:
        canvas = torch.full(self.shape, fill, dtype=torch.float32)
        canvas[self.rows, self.cols] = values.reshape(-1).float().cpu()
        return canvas

    def token_image(self, values: np.ndarray) -> np.ndarray:
        """Per-valid-token values on the token grid, NaN elsewhere."""
        grid = np.full((int(self.token_rows.max()) + 1, int(self.token_cols.max()) + 1), np.nan)
        grid[self.token_rows.numpy(), self.token_cols.numpy()] = values
        return grid


@torch.no_grad()
def baselines(recording, query_poses, query_phases, query_scales, neighbours: list[dict]) -> dict:
    """Copy and compounding predictions `[Q, 1, H, W]` on the output grid, plus coverage."""
    model, device = recording.model, recording.device
    grid = OutputGrid(model)
    train = recording.train_indices

    def copy(key):
        images = []
        for row in neighbours:
            index = row.get(key)
            if index is None:
                images.append(torch.full((1, *grid.shape), float("nan")))
                continue
            image = recording.dataset.get_record(train[index])["image"][None].to(device)
            values = model.encoder(image).values[0].float().cpu()
            # A flipped match is the training frame's view mirrored left-right.
            if row.get(key.replace("_train", "_flipped")):
                values = torch.flip(values, dims=[-1])
            images.append(values)
        return torch.stack(images)

    offsets = pixel_offsets(model.projector, grid.factor)
    compounded, coverage = [], []
    for q in range(query_poses.shape[0]):
        phase = None if query_phases is None else float(query_phases[q])
        volume = recording.intensity_map(
            pose_source="corrected", phase=phase, phase_window=DEFAULT_PHASE_WINDOW
        )
        points = pixel_points(model.projector, offsets, query_poses[q : q + 1].to(device), query_scales[q : q + 1])
        mean = volume.sample(points[0])["mean"].float()
        supported = torch.isfinite(mean)
        coverage.append(float(supported.float().mean()))
        fill = float(mean[supported].mean()) if supported.any() else 0.0
        # Zero outside the valid tokens: masked out, but NaN would poison masked sums.
        filled = torch.where(supported, mean, torch.full_like(mean, fill))
        compounded.append(grid.image(filled, fill=0.0)[None])
    return {
        "copy_nearest": copy("nearest_train"),
        "copy_phase": copy("phase_compatible_train"),
        "compound": torch.stack(compounded),
        "compound_coverage": coverage,
    }


def save_panels(
    directory: Path,
    frame_ids: list[int],
    target: torch.Tensor,
    mask: torch.Tensor,
    predictions: dict[str, torch.Tensor],
    psnr: dict[str, list[float]],
    support_distance: np.ndarray,
    grid: OutputGrid,
) -> list[str]:
    """One PNG per test frame: target, every prediction, model error and support."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    first = next(iter(predictions))
    for q, frame in enumerate(frame_ids):
        valid = mask[q, 0].cpu().numpy()
        shown = lambda image: np.where(valid, image.cpu().numpy(), np.nan)
        columns = [(f"test frame {frame}\ntarget", shown(target[q, 0]))]
        columns += [(f"{name}\n{psnr[name][q]:.2f} dB", shown(image[q, 0])) for name, image in predictions.items()]
        columns.append((f"|error| {first}", shown((predictions[first][q, 0] - target[q, 0]).abs())))
        figure, axes = plt.subplots(1, len(columns) + 1, figsize=(2.6 * (len(columns) + 1), 2.9))
        for axis, (title, image) in zip(axes, columns):
            axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0 if "error" not in title else 0.5)
            axis.set_title(title, fontsize=7)
            axis.axis("off")
        support = axes[-1].imshow(grid.token_image(support_distance[q]), cmap="magma", vmin=0.0, vmax=5.0)
        axes[-1].set_title("distance to training\nplanes (mm)", fontsize=7)
        axes[-1].axis("off")
        figure.colorbar(support, ax=axes[-1], fraction=0.046)
        figure.tight_layout()
        path = directory / f"frame_{frame:05d}.png"
        figure.savefig(path, dpi=110)
        plt.close(figure)
        paths.append(str(path))
    return paths
