# Cardiac 4D sweep — plan (draft 4)

**Goal:** simulate a freehand 2D sweep of a beating 4D-CT heart phantom (20 cardiac frames), reconstruct each frame into a 512×512 sector image, and write v3 `sequence_metadata.json` in the same layout as `synthetic/`.

**Approach:** one MUSiK `Experiment` per cardiac frame. Images are binned by their nearest CT frame, and each phase experiment holds that frame's phantom plus one transducer per image in the bin.

**Scale:** about 1–2k images × about 32 lines each ≈ 32–64k line simulations. The sim grid only has to enclose one steered beam, because MUSiK rotates it per line. We'll tune frequency and grid size for speed over realism.

## Layout

| File | Does | Status |
|---|---|---|
| `config.yaml` | Every parameter; open values are marked `TODO` | mask + labels filled in; sim/sweep values are placeholders |
| `common.py` | Config/paths, mask loading, cardiac timing, pose + sector helpers (trimmed from `synthetic/generate.py`), label-slice sampling | smoke-tested |
| `01_build_phantoms.py` | Mask → crop + margin (cached as `phantoms/mask.npy`) → resample → tissues → `phantoms/frame_XX`, plus `phantom_info.json` | verified on real mask (frame 0: 80 s, 3.3 GB peak, 1 GB on disk) |
| `02_make_trajectory.py` | Keyframed segments → `trajectory.npz`; QA: path/speed/phase-count plots, heart coverage fraction, label-slice montage | smoke-tested (toy mask) |
| `03_setup_experiments.py` | Phase binning → `experiments/phase_XX` + `image_index.json` (image → phase, transducer, sim indices) | smoke-tested (setup only) |
| `04_run.py` | Runs the phase experiments, resumable; one Slurm array task per phase, with optional `--node` split | untested (needs GPU) |
| `05_reconstruct.py` | Per line: `make_scan_line` + `preprocess` → `polar.npy`; then global log compression → scan conversion → `images/%05d.png` + metadata | untested (needs sim results) |
| `06_qa.py` | Recon vs. label slice at the same pose/phase, with boundary overlay, to catch flip/offset/calibration errors | untested |

## Phantom facts
- Source: `assets/cardiac_4d_phantom/img4d_CT_bavcta021_baseline_seg4d.tif`, `(T=20, Z=585, Y=512, X=512)` uint8 at 0.4 × 0.396 × 0.396 mm. Axes are (inferior, posterior, patient-left), right-handed, and kept in file order.
- Labels: LV/RV myocardium and LA/RA walls → `myocardium`; four chamber blood pools plus aorta and PA → `blood`; label 0 → `soft`. There's no chest wall, lungs, valves or vessel walls. Frame 0 ≈ end-diastole, frames 7–8 ≈ end-systole.
- Crop at 50 mm margin: 682 × 596 × 634 voxels (273 × 236 × 251 mm), 5.2 GB cached for all 20 frames. That's about 129M voxels per frame at 0.5 mm phantom resolution.
- LV long axis (base → apex) ≈ (0.19, −0.74, 0.64) in (Z, Y, X), i.e. anterior, leftward and slightly inferior, 94 mm long.

## Open questions / decisions

**Phantom / tissue**
- Tissue values (c / rho / sigma / scale). For sub-voxel `scale` (all our tissues), the per-voxel sound-speed std is `sigma·√(scale / voxel)`, so effective brightness shifts with `phantom.voxel_size_mm`. At 0.5 mm the current values give myocardium 8.9, soft 4.5, blood 0.18 m/s. Should we tune `sigma` against those effective values, or set `scale` = voxel size so `sigma` is the std directly? (The clip at 1/mean(grid shape) only applies below scale/voxel ≈ 0.002, so it's inactive for us.)
- Phantom voxel size vs. sim voxel size. The sim samples the phantom with nearest-neighbour lookup, so scatterers are phantom-voxel-sized blocks. At 1 MHz (sim voxel 0.385 mm) a 0.5 mm phantom is slightly coarser than the grid. At 2 MHz it's 2.6× coarser. Should phantom voxel ≤ sim voxel, at the cost of size?
- Wall layers for the aorta/PA, and a synthetic body wall between probe and heart. Both are optional realism.
- Storage: 1 GB per baked frame at 0.5 mm (float32), and each phase experiment saves its own copy, so ≈ 40 GB total. Should experiments symlink the phantom instead?

**Simulation / transducer**
- Center frequency: the main cost knob and our only lever on sim voxel size (a quarter wavelength).
- Grid size (depth × lateral × elevation), PML, and attenuation. At 1 MHz the current config gives 512 × 128 × 64 voxels per line (inner 185 × 43 × 18 mm). MUSiK widens lateral/elevation to 1.5× the aperture and rounds sizes up to FFT-friendly ones. Settle on the smallest grid that still gives clean lines.
- Elements, aperture, focus, cycles; lines per image (32?) and FOV (32 lines over 75° ≈ 2.4° per line).
- GPU access: both local GPUs are currently busy. Benchmark time per line on whatever we'll actually use → total GPU-hours.
- Far edge: any part of a beam's grid outside the phantom gets edge-clamped values (streaky baked speckle). `margin_mm` should cover probe positions *and* imaging depth along each beam. 02 can check this per pose.

**Sweep / timing**
- Real segment definitions: which views (parasternal-like, apical-like, subcostal-like?), probe standoff, sweep extents. The current three segments are placeholders.
- Frame rate and segment durations, which together set the image count. The current placeholders give ≈1000 images at ≈50 per phase.
- Heart rate: fixed at 60 bpm (matches the TIFF's nominal 0.05 s frame interval), or varying? Phase at t = 0?
- Continuous sweep or separate segments (probe lifted between them)? Should we add hand jitter?
- Coverage target: is `heart_coverage_fraction` a hard requirement? Should it account for elevation thickness?

**Reconstruction / output**
- Depth (160 mm), FOV, dynamic range, TGC, and the global vs. per-image log-compression reference.
- Verify lateral sign and depth zero-offset (06) on the first simulated images.
- Metadata: `raw_tracker_pose` and `calibration` need usgr's calibration (`compute_tracker_from_transducer`, `DEFAULT_CALIBRATION`). Pull those in, or fill the fields in the usgr env.
- Add tracking-error injection with `synthetic/pose_error.py`? Validate by loading through `ReconstructionDataset`?

## Decisions made
- 4D via per-phase experiments; no core 4D support for now.
- Frequency is the only voxel-size lever (`grid_lambda` not exposed).
- Speckle is baked once per frame. All frames share a seed, so the speckle field is fixed in world space.
- `Phantom.interpolate_phantom` uses a direct rounded-index lookup (equivalent to the old nearest-neighbour interpolator, about 1500× faster per line).
- `Phantom.make_complete` generates each tissue's texture only over its own voxels. It uses the same texture model and parameters, and the identical interpolation geometry (bit-identical for the same noise), but draws different random numbers. It's ~100–300× faster and outputs float32 (was accidentally float64). Labels with no tissue now get the default tissue instead of the label number as a sound speed. This also speeds up MUSiK's default per-line (non-baked) mode.
- Phantom → sim-grid sampling verified on frame 0: labels along all 32 rays of a real pose match the planned image plane (98.9% mean agreement, differences only at boundaries from the 0.4 / 0.5 / 0.385 mm grids). A mirrored lateral sign drops agreement to 81%, so orientation and lateral sign are correct.
- Label → tissue map as above.

## Next steps
1. Simulate one line (or one image) to confirm the k-Wave run and reconstruction path. Needs GPU time; both local GPUs are currently in use.
2. Settle tissue values, phantom voxel size, and margin.
3. Design sweeps against the real label previews (02).
4. Benchmark one phase on GPU → fix sim parameters → run 05 + 06 → scale out with 04.
