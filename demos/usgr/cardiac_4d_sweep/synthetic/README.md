**Synthetic recordings for reconstruction**

This package generates controlled ultrasound recordings with known anatomy,
cardiac phase, true poses, and injected tracking error. Outputs use the normal
v3 recording format and load through `ReconstructionDataset`.

```bash
python -m usgr.reconstruction.synthetic \
  --seed 1 --output-dir ./phantom_data --num-frames 1000
```

Use `--config path/to/config.yaml` for configuration files. Dataclass fields map
to hyphenated CLI arguments; CLI values override the file.

| File | Responsibility |
|---|---|
| [generate.py](generate.py) | Configuration, trajectories, phantom, image formation, recording outputs, and CLI parsing |
| [pose_error.py](pose_error.py) | Tracker-origin bias and noise, calibration-independent corruption, and error diagnostics |
| [random_streams.py](random_streams.py) | Stable named random streams for component ablations |
| [tracking_error.md](tracking_error.md) | Error conventions, defaults, assumptions, and reproducibility guarantees |
| `__main__.py` | Package CLI entry point |

The generator constructs a true probe trajectory, creates a pulsating phantom,
and renders images from those true poses. Tracker corruption is generated
separately and propagated through probe calibration. Images therefore stay
identical when only tracking corruption is changed. Metadata records both true
and corrupted poses, the configuration, and the generated error components.

Useful controlled comparisons:

| Change | Flags |
|---|---|
| Use a recorded trajectory as synthetic truth | `--invivo-trajectory-dir path/to/recording` |
| Freeze anatomy | `--phantom-bpm-range 0 0` |
| Hold the probe still | `--trajectory-translation-speed-mps 0 --trajectory-rotation-speed-dps 0` |
| Disable injected tracking error | `--no-pose-error` |
| Disable speckle | `--appearance-speckle-contrast 0` |
| Disable additive image noise | `--appearance-noise-std 0` |
| Omit the dense anatomy evaluation grid | `--skip-eval-grid` |

Use the same seed for paired experiments. Independent random streams preserve
unrelated components; physical dependencies, such as motion-dependent speckle
or phantom placement along a changed trajectory, still apply. Recorded paths
are adopted as truth and are not automatically cleaned of tracking jumps.

Outputs include images, `sequence_metadata.json`, trajectory and pose-error
plots, an MP4 when video encoding succeeds, and a dense anatomy evaluation grid
unless disabled. Enabled deformation also saves its displacement fields.

For Python callers:

```python
from usgr.reconstruction.synthetic.generate import SyntheticConfig, generate

generate(SyntheticConfig(seed=1, output_dir="./phantom_data"))
```

Known limitations of the lesion artifact model, none yet measured against:
attenuation does not accumulate along the beam, so two lesions in line shadow no more than
one; the enhancing band spans the whole sector rather than only the beams through the
lesion; artifacts come from the plane cut, so a lesion just off-plane but inside the beam's
elevation thickness casts nothing. Shadows are drawn per beam from an exact ray-ellipsoid
intersection, with Beer-Lambert strength in the chord length.

The [tracking experiment summary](../../../docs/steamvr_tracking_error_model.md)
remains a standalone evidence document. Broader reconstruction findings remain
in the [reconstruction notes](../README.md).

Run the focused tests from the repository root:

```bash
python -m pytest tests/reconstruction/synthetic -q
```
