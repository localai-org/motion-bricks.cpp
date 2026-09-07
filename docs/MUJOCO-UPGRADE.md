# MuJoCo 3.12.0 runtime

The native SONIC controller now targets [MuJoCo 3.12.0](https://github.com/google-deepmind/mujoco/releases/tag/3.12.0),
the latest stable release checked on 2026-09-07. The independent upstream Docker
environment remains on 3.3.5 so upgrading the runtime does not silently replace
our reference. Both SDK versions are explicitly supported and tested; other
versions require validation before extending the compile-time allowlist.

## Install and build

Follow the normal Linux build instructions first, then:

```sh
bash scripts/download_mujoco.sh
MUJOCO_SDK="$PWD/generated/sonic/mujoco-3.12.0"
cmake --preset debug -DMOTIONBRICKS_ENABLE_PHYSICS=ON \
  -DMUJOCO_INCLUDE_DIR="$MUJOCO_SDK/include" \
  -DMUJOCO_LIBRARY="$MUJOCO_SDK/lib/libmujoco.so.3.12.0"
cmake --build --preset debug
(cd demo && CGO_ENABLED=0 go build -o ../build/debug/bin/motionbricks-demo .)
```

The helper downloads the official Linux x86-64 or AArch64 archive, verifies its
SHA-256, and refuses to overwrite an existing versioned directory. A different
parent directory can be its first argument. Other platforms can install the
official SDK directly and supply the two CMake paths. Nix is optional; there is
no MuJoCo Nix derivation or Python dependency in the native runtime.

The pinned archive SHA-256 values are:

| Platform | SHA-256 |
|---|---|
| Linux x86-64 | `a9367911e6d5eaeade17c2197304687421c1fc932cdf7bcd4cb8cfaf0374dcb2` |
| Linux AArch64 | `08fd5627a2ef7d5a42580c40e014ab2c1a644f082010c584ca361a3ed8cad838` |

Restart the demo after rebuilding: an already running process retains the old
loaded library. Its status line and `/api/stream` expose the loaded version.
The flat C API adds `mb_physics_engine_version`; no existing signatures change.

For reference/rollback, configure a **separate build directory** against the
3.3.5 headers and `libmujoco.so.3.3.5`, then rebuild both library and executable.
Never substitute one SDK library under another SDK's headers: `mjModel` and
`mjData` layouts changed. Runtime/header equality is checked before model
allocation or struct access.

## API and performance changes

- Use the new actuator count and scalar-control address APIs. The original G1
  model has 43 actuators and 43 scalar controls; its 29 controlled motors must
  each have one control. Unsupported layouts are rejected, not indexed as if
  actuator IDs were control addresses.
- Avoid repeated forward-kinematics calls when copying skeleton positions or
  collision transforms. Initialization and each completed step refresh the
  cached transforms before consumers read them.
- Enable sleeping infrastructure before data initialization (except RK4, which
  does not support it), but explicitly keep the actuated G1 tree awake. This
  permits static-geometry savings without freezing an idle/fallen robot or
  stopping SONIC ticks. Our single controlled robot is **not a demonstration
  of sleeping inactive dynamic islands**. See the [upstream sleeping guide](https://mujoco.readthedocs.io/en/3.12.0/programming/simulation.html#sleeping-islands).
- Retain exact compiled hull triangles using the layout verified against both
  versions' own renderer. Approximately coplanar polygon groups are not a
  replacement for the triangulated hull; see [collision geometry](COLLISION-GEOMETRY.md).

Motor gains, torque limits, materials, policy weights and the 50 Hz controller /
200 Hz physics rates are unchanged. New PID/DC-motor capabilities are not
silently substituted into the trained controller. See the [release changes](https://mujoco.readthedocs.io/en/3.12.0/changelog.html).

## Measurements

On the development Ryzen 9 7900, median of three alternating old/new CPU runs
of the identical 498-tick recorded walk, four inference threads, Debug project
build and official optimized SDKs:

| Timed region | 3.3.5 baseline | 3.12.0 + integration changes |
|---|---:|---:|
| One physics substep (`mj_step`) | 0.0925 ms | 0.0688 ms |
| Complete controller tick (inference + four substeps) | 3.018 ms | 2.904 ms |

That is about **26% less physics-step time**, but only **3.8% less total tick
time** in this CPU workload, which is dominated by inference. These are
closed-loop workload comparisons, not an isolated attribution to sleeping or
one engine change, and not a promised speedup on other scenes/hardware. Both
versions finish this walk without falling.

The optional Linux diagnostic interposer can be built with
`cmake --build build/debug --target motionbricks-sonic-profile`. Pass its shared
library to `reference/run_native_sonic.py --library ...` and set
`MOTIONBRICKS_PROFILE_OUTPUT` to a new JSON file. Run each SDK in a separate
process/build, with identical `--model`, `--scene`, `--config`, `--motion` and
`--device`; the recorder includes the loaded MuJoCo version. Do not compare
timings from different clips or while compilation/QA is running.

## Verification

- CPU and NVIDIA Vulkan SONIC parity pass on all 2,205 stored observations.
  Physics trajectories need not be bit-identical across engine releases;
  same-state neural parity remains mandatory.
- New-engine Vulkan walking (498 ticks), stop/turn (598) and Kimodo (506,
  through the fall) recordings also pass the independent Docker ONNX boundary
  checks: exact tokens, motor targets and clipped PD torques; maximum action
  error 4.77e-6, observation error 1.20e-7 and reference FK error 4.77e-7 m.
  Walking and stop/turn finish without falling. The updated source also builds
  and passes the physics API/collision tests against the retained 3.3.5 SDK.
- Physics C API tests, independent torque replay, all collision origins/axes,
  exact foot dimensions and outward hull winding pass under Debug and
  ASan/UBSan. An explicit incompatible-version interposition test confirms
  rejection before any model/scene access; the physics-disabled build also
  passes the API test. A 21-second fuzz smoke completed 86,167 inputs without sanitizer
  findings; trusted GGUF/XML assets are loaded once, not fuzzed. The official
  SDK binary itself is not sanitizer-instrumented. Clang's compiler-rt
  development headers must be available for the new SDK's sanitizer header.
- Headless Chromium streaming QA passes with 0 and 100 ms injected delivery
  delay: 1.017x playback rate, zero measured walking underruns, no idle drift,
  and about 24 FPS with the exact 88,604-triangle overlay under software WebGL.
  Kimodo completion resumes walking. The physical Kimodo test still falls,
  then continues for another 4.6 simulated seconds without NaNs, an automatic
  reset, or pausing. The upgrade does not cure that clip's tracking failure.

Local measurement and rollout artifacts live under
`generated/sonic/mujoco-upgrade/`; browser reports/screenshots are under
`generated/sonic/ggml/stream-qa/`. Neither SDKs nor large test artifacts are
checked into Git.
