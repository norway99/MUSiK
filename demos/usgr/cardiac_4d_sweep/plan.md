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
- Crop: union of labels over the cycle plus a 10 mm background margin. Lookups past the grid clamp to background and texture is generated at true world positions, so probes may sit outside the phantom.
- LV long axis (base → apex) ≈ (0.19, −0.74, 0.64) in (Z, Y, X), i.e. anterior, leftward and slightly inferior, 94 mm long.

## Open questions / decisions

**Phantom / tissue**
- Tissue values (c / rho / sigma / scale): tune from the first sample images. Current relative speckle levels are myocardium +6 dB and blood −28 dB vs. soft. These ratios don't depend on voxel size; boundary (specular) echoes vs. speckle and the noise floor do.
- Phantom voxel size: with per-simulation texture it only sets boundary geometry (scatterers are always sim-voxel-sized, λ/4). 0.5 mm matches the mask's native 0.4 mm and keeps boundary staircasing well under λ at 1 MHz. The λ/2 block effect only applied to baking, which we dropped.

**Simulation / transducer**
- Center frequency: the main cost knob. It sets the sim voxel (λ/4) and therefore scatterer size; phantoms don't need rebuilding when it changes.
- Grid size (depth × lateral × elevation), PML, and attenuation. At 1 MHz the current config gives 512 × 128 × 64 voxels per line (inner 185 × 43 × 18 mm). MUSiK widens lateral/elevation to 1.5× the aperture and rounds sizes up to FFT-friendly ones. Settle on the smallest grid that still gives clean lines.
- Elements, aperture, focus, cycles; lines per image (32?) and FOV (32 lines over 75° ≈ 2.4° per line).
- GPU access: both local GPUs are currently busy. Benchmark time per line on whatever we'll actually use → total GPU-hours.

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
- No baking. Texture is generated per simulation on the sim grid with `Phantom.world_texture`: each voxel's noise is a hash of (seed, tissue, world lattice cell). Scatterers are therefore fixed in world space across lines, images and cardiac phases (but don't move with the tissue).
- Per-simulation random streams: each simulation draws from `Phantom.simulation_rng(index)`, derived from (seed, index). Previously, forked prep workers shared identical copies of `phantom.rng`, so with `workers > 2` paired lines got identical noise.
- The texture exponent stays at 1/2 for now, with a comment in `Phantom.__texture_parameters` explaining the discrepancy and its effects.
- `Simulation` no longer calls `phantom.get_complete()` on every line. It built property maps for the whole phantom and threw them away: 393 s per worker on our phantom, versus 0.6 s for the actual per-line prep. The run worker's "inactive" message now prints once per 5 minutes instead of every second.
- `Phantom.interpolate_phantom` uses a direct rounded-index lookup (equivalent to the old nearest-neighbour interpolator, about 1500× faster per line).
- `Phantom.make_complete` generates each tissue's texture only over its own voxels. It uses the same texture model and parameters, and the identical interpolation geometry (bit-identical for the same noise), but draws different random numbers. It's ~100–300× faster and outputs float32 (was accidentally float64). Labels with no tissue now get the default tissue instead of the label number as a sound speed. This also speeds up MUSiK's default per-line (non-baked) mode.
- Phantom → sim-grid sampling verified on frame 0: labels along all 32 rays of a real pose match the planned image plane (98.9% mean agreement, differences only at boundaries from the 0.4 / 0.5 / 0.385 mm grids). A mirrored lateral sign drops agreement to 81%, so orientation and lateral sign are correct.
- Label → tissue map as above.
- No added anatomy (vessel walls, body wall) for now; revisit after the first sample images.
- Storage is fine to allocate. Without baking, each frame is just a label mask (~1 MB compressed on disk).

- Pilot tuning (image 451, phase 11): 90° FOV, 48 lines, 0.75 MHz, probe 2 cm closer, soft-tissue sigma 3. Myocardium now stands out clearly from the background.
- Chamber haze (blood ≈ −13.5 dB vs myocardium) is accepted as clutter. It's unchanged by Hanning receive, elevation focus 100 mm, and side PML 16. The simulation's own floor (homogeneous medium) is −43 dB, and slice thickness is ruled out (≤3% of cavity pixels have wall within ±5 mm out of plane). Likely reverberation and pulse tail behind the bright anterior wall.
- PML stays at [16, 8, 8] (default). A 32-voxel depth PML lowers a far-boundary artifact at about half the recorded depth (~90 mm here) from −21 to −31 dB re wall, but we accept it as-is. `PML_alpha` has no effect on it. Thicker side PML (16) showed no benefit.
- Package fixes from the pilots: `SimProperties.PML_alpha` is now passed to k-Wave (it was ignored); a single ray points straight ahead for any sweep; `Focused.make_scan_line` applies receive apodization to every element, shifts by exactly the delay (was off by one), and handles negative delays; grid sizes pad only to the next 7-smooth even size.

## Next steps
1. Simulate one line (or one image) to confirm the k-Wave run and reconstruction path. Needs GPU time; both local GPUs are currently in use.
2. Settle tissue values from the first sample images.
3. Design sweeps against the real label previews (02).
4. Benchmark one phase on GPU → fix sim parameters → run 05 + 06 → scale out with 04.
