# Cardiac 4D sweep — plan (draft 3)

**Goal:** simulate a freehand 2D sweep of a beating 4D-CT heart phantom (20 cardiac frames), reconstruct each frame into a 512×512 sector image, and write v3 `sequence_metadata.json` in the same layout as `synthetic/`.

**Approach:** one MUSiK `Experiment` per cardiac frame. Images are binned by their nearest CT frame, and each phase experiment holds that frame's phantom plus one transducer per image in the bin. 4D support needs no core changes. The one core change we made is a speed fix (see Decisions).

**Scale:** about 1–2k images × about 32 lines each ≈ 32–64k line simulations. The sim grid only has to enclose one steered beam, because MUSiK rotates it per line. We'll tune frequency and grid size for speed over realism.

## Layout

| File | Does | Status |
|---|---|---|
| `config.yaml` | Every parameter; open values are marked `TODO` | placeholder values |
| `common.py` | Config/paths, mask loading, cardiac timing, pose + sector helpers (trimmed from `synthetic/generate.py`, since usgr isn't importable here), label-slice sampling | smoke-tested |
| `01_build_phantoms.py` | Mask → crop + margin (cached as `phantoms/mask.npy`) → resample → tissues → `phantoms/frame_XX`, plus `phantom_info.json` (heart center, spacing, crop origin) | smoke-tested; real-mask run in progress |
| `02_make_trajectory.py` | Keyframed segments → `trajectory.npz`; QA: path/speed/phase-count plots, heart coverage fraction, label-slice montage | smoke-tested |
| `03_setup_experiments.py` | Phase binning → `experiments/phase_XX` + `image_index.json` (image → phase, transducer, sim indices) | smoke-tested (setup only) |
| `04_run.py` | Runs the phase experiments, resumable; one Slurm array task per phase, with optional `--node` split | untested (needs GPU) |
| `05_reconstruct.py` | Per line: `make_scan_line` + `preprocess` → `polar.npy`; then global log compression → scan conversion → `images/%05d.png` + metadata | untested (needs sim results) |
| `06_qa.py` | Recon vs. label slice at the same pose/phase, with boundary overlay, to catch flip/offset/calibration errors | untested |

Smoke tests used a toy 4D mask of concentric shells at 2 mm. They confirmed: phantom build, trajectory + previews, experiment save, and an exact pose round-trip through `TransducerSet` save/load.

## Open questions / decisions

**Input / phantom**
- Mask labels: are 9/10 (atrial walls) the same tissue as ventricular myocardium? Should the aorta/PA get a wall layer? No vessel walls are labelled.
- Tissue table: c / rho / sigma / scale per tissue. Blood should be very dark and myocardium bright.
- Background: the mask has no chest wall, lungs, ribs, pericardium or valves, so everything outside the heart is one homogeneous `soft` tissue. Should we add a synthetic body-wall layer between probe and heart?
- Phantom size: `margin_mm` must enclose every probe position. At 50 mm margin the crop is about 270 × 235 × 250 mm. At 0.5 mm voxels that's ~120M voxels: ~0.5 GB baked per frame, 20 frames, and each phase experiment saves its own copy. Choose `voxel_size_mm` together with the margin.
- Cardiac timing: the TIFF frame interval is 0.05 s (20 frames per 1 s cycle). That's probably nominal, but it matches the 60 bpm default.

**Simulation / transducer**
- Center frequency: the main cost knob, and our only lever on sim voxel size (MUSiK fixes it at a quarter wavelength).
- Grid size (depth × lateral × elevation), PML, and attenuation. Settle on the smallest grid that still gives clean lines.
- Elements, aperture, focus, cycles; lines per image (32?) and FOV (32 lines over 75° ≈ 2.4° per line).
- Benchmark one phase on one GPU first to get time per line → total GPU-hours.

**Sweep / timing**
- Real segment definitions: which views (parasternal-like, apical-like, subcostal-like?), probe standoff, sweep extents. The current three segments (translate, fan, rotate) are placeholders. World axes are (inferior, posterior, patient-left). The LV long axis runs base → apex along about (0.19, −0.74, 0.64), i.e. anterior, leftward and slightly inferior, over 94 mm.
- Frame rate and segment durations, which together set the image count. The current placeholders give ≈1000 images at ≈50 per phase.
- Heart rate: fixed, or varying? Phase at t = 0? Images per phase come out roughly uniform automatically unless the heart rate and frame rate lock together.
- Continuous sweep or separate segments (probe lifted between them)? Is motion within a segment smooth enough, or do we want realistic hand jitter?
- Is the sweep coverage target (`heart_coverage_fraction`) a hard requirement? Should it account for elevation thickness?

**Reconstruction / output**
- Depth (160 mm), FOV, dynamic range, TGC, and the global vs. per-image log-compression reference.
- Verify lateral sign and depth zero-offset with `06_qa.py` on the first real phase.
- Metadata: `raw_tracker_pose` and `calibration` need usgr's calibration (`compute_tracker_from_transducer`). We could run the metadata step in the usgr env, or port that function.
- Extra per-frame fields to keep (cardiac phase/frame and segment are in `generation` for now). Should we add tracking-error injection with `synthetic/pose_error.py`?
- Whether to validate the output by loading it through `ReconstructionDataset`.

## Decisions made
- 4D handled by per-phase experiments (no core 4D support for now).
- Frequency is the only voxel-size lever; `grid_lambda` stays at MUSiK's default and isn't exposed.
- Speckle is baked once per frame (`bake_speckle: true`).
- **Core change:** `Phantom.interpolate_phantom` now does a direct rounded-index lookup instead of building a `NearestNDInterpolator` over the whole phantom on every line. The two are equivalent on a regular voxel grid (clamped at the edges, ties at exact half-voxels aside). This was verified with 0 mismatches against the old code on both branches, including out-of-bounds queries. It's about 1500× faster per line on a 300³ phantom, and all 78 tests pass.
- Mask axes are kept in file order `(T, Z, Y, X)` = (time, inferior, posterior, patient-left). That order is right-handed, while transposing to X, Y, Z would mirror the anatomy.

## Next steps
1. Finish the real-mask phantom build → settle margin/voxel size and tissue values.
2. Design sweeps against the real label previews (02).
3. Benchmark one phase (a few images) on GPU → fix the sim parameters.
4. Run 05 + 06 on that phase → check orientation → scale out with 04.
