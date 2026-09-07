# Upstream SONIC physical tracking

This document records the original upstream/TensorRT milestone. The subsequent
[GGML inference and live native physics milestone](SONIC-GGML.md) is implemented
and has its own same-state parity and live-loop boundary checks.

## Milestone and tasks

Official GEAR-SONIC G1 inference now drives the MuJoCo robot from recorded
motion-bricks.cpp motion. Native walking and stop/turn work. Arbitrary Kimodo
motions are not guaranteed physically feasible.

- [x] Pin and hash official source, matching model/config, robot assets and example.
- [x] Provide parameterized Docker build/run commands; sanitize the controller.
- [x] Automate isolated headless startup, suspension removal, playback and shutdown.
- [x] Record clocks, references, qpos/qvel, full observations, latents, actions,
  targets, gains, torques and contacts; preserve failures and provenance.
- [x] Capture native walking and stop/turn with model/library/style identity,
  seeds, sampling mode, original API responses and timestamps.
- [x] Convert G1Skeleton34 to physical G1 hinges and 50 Hz CSVs; validate with
  the actual upstream converter, FK/velocity methods, sanitized reader and
  62 independent joint/root probes.
- [x] Check actual encoder/decoder buffers, previous-action timing, permutations,
  scaling, delivered targets and clipped PD torques.
- [x] Repeat official and native walking; test stop/turn and a real Kimodo clip.
- [x] Report joint/root/body errors, contact slip, saturation, falls and deadlines.
- [x] Add synchronized physical/reference browser overlays, trajectories, error
  lines, pause/replay/scrub and damped tracking; clearly label failed recordings.
- [x] Add offline rejection/timing tests and a full-run evaluator that fails on
  invalid physical state, incomplete playback or boundary-check failures.

Commands: [reference/SONIC.md](../reference/SONIC.md).
This milestone uses upstream TensorRT inference, not a GGML SONIC port.
Live asynchronous planning, measured-state feedback, a native simulation C API
and real hardware are outside this milestone.

## Recorded results

The finalized startup freezes physics through upstream initialization, then
starts integration at CONTROL activation. Suspension is removed one second
later; these runs play immediately at release (`--settle-seconds 0`). Metrics
include the unassisted landing phase. Artifacts live in ignored
`generated/sonic/`.

| Recording | Active duration | Joint RMSE | Final XY drift | Root-relative body RMS | Result |
|---|---:|---:|---:|---:|---|
| `baseline-007` (official) | 8 s | 0.121 rad | 0.712 m | 2.94 cm | Completed |
| `baseline-008` (same input) | 8 s | 0.132 rad | 0.787 m | 3.06 cm | Completed |
| `native-walk-003` | 9.8 s | 0.110 rad | 0.617 m | 2.84 cm | Completed |
| `native-walk-004` (same input) | 9.8 s | 0.105 rad | 0.337 m | 2.85 cm | Completed |
| `stop-turn-003` | 11.8 s | 0.098 rad | 0.408 m | 2.46 cm | Completed |
| `kimodo-return-002` | 10.2 s before fall | 0.182 rad | 2.007 m | 5.17 cm | Failed during authored motion |

Completed runs had no falls, pose resets, non-finite state, sanitizer findings
or missed physics deadlines. Checked PD-to-torque errors are exactly zero.
Mean geometric contact slip was 3.40 cm/s on the official run, 2.67/2.21 cm/s
on native repeats and 1.83 cm/s on stop/turn. Body-actuator saturation fractions
were 0.205%, 0.130%/0.0915% and 0.0687%, respectively.

Physical repeats are not bit-identical: asynchronous DDS/control/physics timing
can change the closed-loop rollout. These measurements are not same-state
neural parity. G1 mode 0 observes future hinge poses/velocities and base
orientation, not reference world root translation or desired root velocity.
Displayed drift is real and is not removed by per-frame alignment.

The functional gate establishes complete unassisted playback with valid
boundaries, not production tracking quality or a universal drift tolerance.
Each run has versioned `provenance.json`, `summary.json`, `diagnostics.json`,
`boundary-checks.json`, `contacts.json` and `acceptance.json`. Failed runs cannot
pass the evaluator; diagnostic-only failed playback is explicitly requested.

## Findings and limits

**Initialization.** Motor commands appear during InitControl, before policy
control. Advancing physics there left the waist at its 0.52 rad limit before
the first policy tick and caused startup falls. The runner now publishes the
initial sensors without stepping until CONTROL. It never resets poses during
tracking or changes policy arithmetic, gains, scene or timestep. Frozen-state
samples and suspension/play events are separated from physical-step checks.

**Kimodo feasibility.** “A person waves at someone then turns around” is inserted
after 3.2 s of native walking. Adapter, observation, action and torque checks
pass, but the robot falls before returning to MotionBricks. In the played
window the reference exceeds ankle limits (left pitch by 0.163 rad, right roll
by 0.099 rad) and waist pitch by 0.0185 rad. This supports a feasibility concern,
not proof of the sole cause of the fall. The reference was not clipped or
repaired to make the test pass. Jump tracking and successful arbitrary Kimodo
handoffs remain unproven. `?physics=kimodo-failed` shows the failed recording.

**Boundary tolerances.** Future joint observations, latent-to-decoder transfer,
previous actions and F32 motor targets match exactly. Quaternion-derived
observations allow 3e-6; rounded state CSVs allow 1e-8. Measured decoder-history
error is below 2e-7. Duplicate identical targets prove content but are excluded
from unique-message latency statistics. Actual mode-0 lookahead is ten samples
separated by five 50 Hz ticks (0.9 s). Contact slip uses penetrating foot/world
contacts and `J(q)*qvel`, checked by finite differences; it is not force-weighted.

**Upstream SLERP difference.** Upstream's small-angle fallback returns a midpoint
independent of interpolation time. The adapter keeps time-correct SLERP.
Regular comparisons use 3e-5; only identified fallback samples use its derived
0.0011 bound. Root and hinge differences are reported separately, not hidden
with blanket tolerances. Upstream's unusual penultimate-difference terminal
joint velocity is retained.

**Runtime.** Physics runs at 200 Hz and policy at 50 Hz in real time. Observed
policy periods are approximately 20 ms. Cached-engine initialization takes
about 24 s before activation on the tested system. CUDA initialization under
ASan required `protect_shadow_gap=0`, independently reproduced with
`sonic_cuda_probe.cpp`; memory-access/UB instrumentation remain enabled.
No unrelated Nix derivations are built.

## Follow-up milestones

- Investigate authored-motion feasibility/contact transitions and failed clips.
- Completed in [SONIC-GGML.md](SONIC-GGML.md): checked conversion, CPU/Vulkan
  same-state neural parity, opaque native MuJoCo C handles, and bounded live
  reference buffers with explicit buffer-exhaustion behaviour.
- Independently validate measured-state feedback into MotionBricks.
- Add perturbations, richer scenes and interactive physics controls.

## Pinned identities

Source: `a0732b642c0333077e127a2f56ab0014c196bca4`.
HF `nvidia/GEAR-SONIC`: `6733128a3d8a523b1418b06bca3cdf61c8b0987f`.
The manifest verifies 86 assets; deployment config is byte-identical to source.
Encoder: 50,100,513 bytes, SHA-256
`013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3`.
Decoder: 40,900,688 bytes, SHA-256
`c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed`.
The official scene has 43 actuators (29 body and 14 hand); the original
implementation controls hand coordinates. Assets retain their respective
licenses; this milestone does not authorize redistribution.

References: [model releases](https://huggingface.co/nvidia/GEAR-SONIC),
[MuJoCo quick start](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/quickstart.html),
[motion format](https://nvlabs.github.io/GR00T-WholeBodyControl/references/motion_reference.html),
[observation configuration](https://nvlabs.github.io/GR00T-WholeBodyControl/references/observation_config.html).
