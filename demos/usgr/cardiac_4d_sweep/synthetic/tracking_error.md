**Synthetic tracking error and paired recordings — 2026-09-21**

`python -m usgr.reconstruction.synthetic` now uses the jump-free
`tracker_correlated_bias_v1` model in `usgr/reconstruction/synthetic/pose_error.py`.
It replaces the original world/patient and body/probe spline model. Existing
recordings still load, but obsolete generator configuration fields are rejected;
there is no compatibility generation mode. The original `reconstruction`
generator is separate and unchanged.

The [experiment summary](../../../docs/steamvr_tracking_error_model.md) motivates the structure:
small temporally correlated fluctuations, coherent offsets during sweeps, and
weak motion dependence of fast jitter. This is an empirical surrogate, not a
fit of a uniquely identified six-dimensional tracking process. Large jumps and
their transients are deliberately excluded; this does not establish that a
separate jump detector can always correct real recordings.

**Conventions and defaults**

Corruption acts on raw tracker poses, before tracker-to-transducer calibration.
Images are rendered using the true transducer poses. Raw tracker poses and true
transducer poses are saved in the normal v3 recording format.

| Component | Translation RMS norm | Rotation RMS norm | Dynamics |
|---|---:|---:|---|
| Coherent bias | 2 mm | 0.5° | Smooth stationary Matérn-3/2 process on a time/motion clock |
| Correlated fluctuations | 0.15 mm | 0.018° | Stationary OU process, translation time constant 0.08 s and rotation 0.20 s |
| Independent fluctuations | 0.05 mm | 0.005° | Independent Gaussian samples at the saved frames |

Amplitudes are **ensemble RMS vector norms**, not coordinate standard deviations
or guaranteed realized RMS within every finite recording. The initial model is
isotropic: each coordinate has standard deviation RMS/√3. No sequence-wide
normalization or forced zero initial bias is applied. Translation and rotation
processes use separate streams; their cross-covariance is not measured and is
initially modeled as zero at the tracker. Calibration introduces the appropriate
coupling at the transducer and image points.

The bias clock advances by
`dt / 120 s + tracker_path_length / 0.15 m + rotation_path_angle / 45°`.
Its covariance at clock separation `u` is proportional to
`(1 + sqrt(3)*u) * exp(-sqrt(3)*u)`. Exact state transitions produce smooth,
bounded-variance bias without fitting the correction model's spline family.
The time/path scales and 2 mm / 0.5° amplitudes are **modeling assumptions**;
the fixture did not measure them independently. Bias changes slowly in holds
and more quickly along sweeps. Revisiting a pose need not reproduce its bias:
this is path-dependent stochastic evolution, not a fitted spatial bias map.
Set `bias_distance_m` or `bias_angle_deg` to zero to disable that clock term.

The OU covariance is `exp(-abs(dt)/correlation_seconds)`, using actual timestamp
differences. Zero correlation seconds selects independent draws. No speed-based
jitter multiplier is applied: the observed acceleration association does not
identify its causal coefficient. Optional `lag_seconds` defaults to zero and
interpolates earlier true tracker poses, clamped at sequence endpoints.

`frame: world` is the default convention, not an inference about physical origin.
Translation is a displacement **of the tracker origin**; rotation is about that
origin, never the world origin. For summed displacement `d` and rotation vector
`r`, the world convention is `t_raw = t_source + d`,
`R_raw = Exp(r) @ R_source`. With `frame: body`, it is
`t_raw = t_source + R_source @ d`, `R_raw = R_source @ Exp(r)`.
Component arrays contain `[rotation_vector_rad(3), displacement_m(3)]`. They are
**not SE(3) twists**: their translation entries are actual origin displacements.
Components add before the single rotation exponential. Patient motion and
probe calibration error are not included in these tracking components.

**Randomness and component omission**

A global integer seed feeds named SHA-256/SeedSequence/PCG64 streams. Stream names
are stable in the implementation; metadata records the global seed and version
`named_sha256_pcg64_v1`. Trajectory, phantom, cardiac phase, additive image noise,
speckle, and deformation have independent streams. Phantom groups (tissue,
enhancing/attenuating/specular lesions, shells, smooth regions) are isolated.
Each bias/colored/white translation/rotation channel is independently seeded.
No Python `hash()` or conditional sibling-spawn ordering is used.

For the same seed, timestamps, and other settings:

- Freezing cardiac phase (`--phantom-bpm-range 0 0`) preserves phantom parameters,
  probe trajectory, and injected tracking errors.
- Disabling pose corruption (`--no-pose-error`, or YAML `pose_error.enabled: false`)
  preserves rendered image bytes, anatomy, phase, and true trajectory.
- Disabling speckle preserves the trajectory, anatomy, additive-noise stream and
  injected tracking errors. Removing a phantom group preserves all parameters of
  the other groups; their numeric array indices can shift.
- Disabling or resizing a pose-error component does not redraw another component.
  Extending an otherwise identical error trajectory preserves its sampled prefix.

Physical dependencies still apply: stopping the **probe** changes its trajectory,
phantom placement, the bias motion clock and speckle decorrelation. Changing
anatomy changes its appearance and masks. Changing frame times changes process
samples. Changing the count within one phantom group may redraw that group.
These are not promises of identical rendered frames under arbitrary config edits.
Same-seed outputs are numerically reproducible within the same implementation;
metadata creation timestamps and output paths naturally differ. New streams and
the new model intentionally do not reproduce recordings made by the old generator.

**Diagnostics and ground truth**

Every recording includes `pose_error_diagnostics.png` and `.pdf`, linked from
`generation.visualizations.pose_error_plot`. They plot realized tracker-origin
translation and angular errors, separate component norms, RMS changes at short
lags, and per-coordinate lag correlations. Angular lag changes use rotation
composition. Translation and angular units are always separate.

`generation.pose_error` saves true/raw tracker matrices, all component arrays,
their frame and composition convention, effective lag, parameters, and bias
clock. `generation.pose_error_diagnostics` saves realized RMS/P95, component RMS,
actual lag times, pair counts, lag changes and correlations. Constant-signal
correlations are displayed as zero. Overlapping pairs and short sequences make
these descriptive diagnostics, not independent statistical estimates.

The 0.15 mm / 0.018° correlated scales approximate quiet-hold variability; the
white contribution and slowly changing bias also affect measured hold RMS.
An initial 10 s stationary check at 16 Hz for seeds 1–3 gives centered position
RMS **0.204 / 0.155 / 0.206 mm**, rotation RMS **0.0268 / 0.0209 / 0.0234°**.
This is a sanity check, not validation against the complete experimental
distribution. Dynamic covariance, bias timescales and anisotropy remain tunable.

Example paired generation (all other flags identical):

```bash
python -m usgr.reconstruction.synthetic \
  --seed 1 --output-dir ./synthetic_moving --num-frames 1000

python -m usgr.reconstruction.synthetic \
  --seed 1 --output-dir ./synthetic_static_anatomy --num-frames 1000 \
  --phantom-bpm-range 0 0

python -m usgr.reconstruction.synthetic \
  --seed 1 --output-dir ./synthetic_clean_tracking --num-frames 1000 \
  --no-pose-error
```

Use `--invivo-trajectory-dir` for a recorded path. Measured source poses become
synthetic truth, including any original discontinuities; the generator neither
detects nor removes those events. Timestamps must be finite and strictly increasing.
