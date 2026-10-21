# Cardiac 4D sweep — plan (draft 5)

**Goal:** simulate a freehand 2D sweep of a beating 4D-CT heart phantom (20 cardiac frames), reconstruct each frame into a 512×512 sector image, and write v3 `sequence_metadata.json` in the same layout as `synthetic/`.

**Approach:** one MUSiK `Experiment` per cardiac frame. Images are binned by their nearest CT frame, and each phase experiment holds that frame's phantom plus one transducer per image in the bin.

**Scale:** about 1–2k images × 48 lines each ≈ 48–96k line simulations. The sim grid only has to enclose one steered beam, because MUSiK rotates it per line. Pilot cost: ~10–16 s per line on a GPU shared with another job (not yet benchmarked uncontended).

## Layout

| File | Does | Status |
|---|---|---|
| `config.yaml` | Every parameter; open values are marked `TODO` | phantom, transducer, sim and recon settings from the pilots; sweep segments are placeholders |
| `common.py` | Config/paths, mask loading, cardiac timing, pose + sector helpers (trimmed from `synthetic/generate.py`), label-slice sampling | smoke-tested |
| `01_build_phantoms.py` | Mask → crop + margin (cached as `phantoms/mask.npy`) → resample → tissues → `phantoms/frame_XX`, plus `phantom_info.json`; `--reuse-mask` skips re-reading the TIFF | verified on real mask (~1 MB per frame) |
| `02_make_trajectory.py` | Continuous path through keyframed sections → `trajectory.npz`. Timing from speed limits (links at the limits, sweeps fill `duration_s`); look-at interpolation. QA: path/speed/phase plots, per-chamber coverage, images without heart, probe clearance, label-slice montage | works on real mask |
| `03_setup_experiments.py` | Phase binning → `experiments/phase_XX` + `image_index.json` (image → phase, transducer, sim indices); `--images` for pilots | verified (pilot) |
| `04_run.py` | Runs the phase experiments, resumable; one Slurm array task per phase, with optional `--node` split | verified (pilot) |
| `05_reconstruct.py` | Per line: `make_scan_line` + `preprocess` → `polar.npy`; then global log compression → scan conversion → `images/%05d.png` + metadata (only images that were set up) | verified (pilot) |
| `06_qa.py` | Recon vs. label slice at the same pose/phase, with boundary overlay, to catch flip/offset/calibration errors | verified (pilot): orientation and depth correct |

## Phantom facts
- Source: `assets/cardiac_4d_phantom/img4d_CT_bavcta021_baseline_seg4d.tif`, `(T=20, Z=585, Y=512, X=512)` uint8 at 0.4 × 0.396 × 0.396 mm. Axes are (inferior, posterior, patient-left), right-handed, and kept in file order.
- Labels: LV/RV myocardium and LA/RA walls → `myocardium`; four chamber blood pools plus aorta and PA → `blood`; label 0 → `soft`. There's no chest wall, lungs, valves or vessel walls. Frame 0 ≈ end-diastole, frames 7–8 ≈ end-systole.
- Crop: union of labels over the cycle plus a 10 mm background margin. Lookups past the grid clamp to background and texture is generated at true world positions, so probes may sit outside the phantom.
- LV long axis (base → apex) ≈ (0.19, −0.74, 0.64) in (Z, Y, X), i.e. anterior, leftward and slightly inferior, 94 mm long.

## Open questions / decisions

**Simulation**
- GPU access and budget: benchmark time per line on an uncontended GPU (or the cluster) → total GPU-hours for the chosen sweep.

**Sweep / timing**
- Sweep (config): one continuous 100 s trajectory at 10 Hz, 71 bpm → 1001 images (48k lines), 48–52 images per CT frame. Sections: parasternal short-axis fan from above the atria to the apex (122 images) → link (173) → apical 180° roll about the LV axis (391) → link (174) → subcostal front→back fan (141). Links run at 90% of the speed limits (peak 13.6°/s, 8.3 mm/s) and take ~35% of the images (accepted). All four chambers are ≥99.98% within 2 mm of an image plane; 61 images (6%) miss the heart, mostly at the fan ends and mid-link; probe stays ≥26 mm from the heart.
- Frame rate and segment durations, which together set the image count. The current placeholders give ≈1000 images at ≈50 per phase.
- Continuous sweep or separate segments (probe lifted between them)? Should we add hand jitter?
- Coverage target: is `heart_coverage_fraction` a hard requirement? Should it account for elevation thickness?

**Reconstruction / output**
- Dynamic range (50 dB), TGC, and the global log-compression reference: revisit on the first multi-phase results.
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

- Pilot settings (image 451, phase 11): 90° FOV, 48 lines, 0.75 MHz (sim voxel 0.51 mm), 160 mm depth, 20 × 12 mm aperture with 64 elements, rectangular receive, focus 80 mm (azimuth and elevation), grid 170 × 25 × 15 mm (widened to ≥1.5× aperture by MUSiK), soft-tissue sigma 3, phantom voxel 0.5 mm, workers 1. Myocardium stands out clearly from the background; orientation and depth match the labels.
- Chamber haze (blood ≈ −13.5 dB vs myocardium) is accepted as clutter. It's unchanged by Hanning receive, elevation focus 100 mm, and side PML 16. The simulation's own floor (homogeneous medium) is −43 dB, and slice thickness is ruled out (≤3% of cavity pixels have wall within ±5 mm out of plane). Likely reverberation and pulse tail behind the bright anterior wall.
- PML stays at [16, 8, 8] (default). A 32-voxel depth PML lowers a far-boundary artifact at about half the recorded depth (~90 mm here) from −21 to −31 dB re wall, but we accept it as-is. `PML_alpha` has no effect on it. Thicker side PML (16) showed no benefit.
- Package fixes from the pilots: `SimProperties.PML_alpha` is now passed to k-Wave (it was ignored); a single ray points straight ahead for any sweep; `Focused.make_scan_line` applies receive apodization to every element, shifts by exactly the delay (was off by one), and handles negative delays; grid sizes pad only to the next 7-smooth even size.

## Next steps
1. ~~Simulate one image to confirm the k-Wave run and reconstruction path.~~ Done (pilot image 451).
2. ~~Settle tissue values from the first sample images.~~ Done (soft sigma 3; revisit after multi-phase results if needed).
3. ~~Design sweeps against the real label previews (02).~~ Done: continuous five-section sweep, 10 Hz, 71 bpm.
4. Benchmark time per line on an uncontended GPU → size the sweep → run all phases with 04 → 05 + 06.
