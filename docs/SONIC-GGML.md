# GGML SONIC and live physics

The original released **GEAR-SONIC G1 mode-0** policy now runs in native C++
on GGML CPU and Vulkan. Its encoder, FSQ32 tokens, decoder and composed actions
are checked against the pinned official ONNX, not against another GGML run.
There is no ONNX Runtime, TensorRT, DDS or Python inference dependency in the
demo. Optional native **MuJoCo 3.3.5** supplies simulation, not inference.

## Scope and parity

The converted file contains 14,415,453 F32 parameters, approximately 58 MB.
It contains the G1 encoder branch and decoder, **not** the other two encoder
modes, later SONIC releases, or a text model. Unsupported modes and incompatible
GGUF identities are rejected. Conversion reads only hash-verified ONNX data,
never pickle. Nothing in this implementation grants redistribution rights to
the weights or robot assets.

The independent suite covers 2,205 observations: every recorded controller
tick from official walking, native walking, stop/turn and the Kimodo test,
plus 16 synthetic inputs. It checks all 12 MLP preactivations, discrete tokens,
decoder actions with captured inputs, and encoder-to-decoder composition.

| Check | CPU | NVIDIA Vulkan |
|---|---:|---:|
| Maximum absolute layer/output error | 1.33514e-5 | 1.33514e-5 |
| Maximum relative L2 error | 6.89e-7 | 6.93e-7 |
| Different FSQ tokens | 0 | 0 |

Both absolute error <= 5e-4 **and** relative L2 <= 1e-5 are required; token
equality is exact. ONNX Runtime uses the original unmodified computation with
graph optimizations disabled. Its tokens also match the captured upstream
TensorRT tokens exactly; decoder differences are below 6.68e-6. F32 matrix
multiplication is explicitly requested. FSQ uses ONNX ties-to-even rounding
and preserves the exported F32 subtract/add sequence.

Neural parity is separate from closed-loop tracking. `check_native_sonic.py`
independently rebuilds the reference from the upstream-validated Python
converter, then checks the **actual native** observation/history buffers,
same-state ONNX tokens/actions, motor targets, clipped PD torques and reference
FK. The 498-tick walk has exact tokens/targets/torques, action error <1.91e-6,
observation error <5.97e-8 and reference FK error <4.77e-7 metres.
The stop/turn (598 ticks) and Kimodo (506 ticks, through the fall) boundary
checks also pass: across all three, action error is <=5.73e-6, observation
error <=1.20e-7, and tokens, motor targets and clipped torques remain exact.

The native walk and stop/turn complete without falling. The tested Kimodo
wave/turn loses balance with the current weights, motor configuration and
materials. That is **not proof that the animation is universally impossible**,
nor does passing neural parity guarantee successful physical tracking.

## Build and prepare assets

Use the normal Linux build instructions first. For physics, install the native
MuJoCo 3.3.5 SDK and enable the optional build:

```sh
cmake --preset debug -DMOTIONBRICKS_ENABLE_PHYSICS=ON \
  -DMUJOCO_INCLUDE_DIR="$MUJOCO_SDK/include" \
  -DMUJOCO_LIBRARY="$MUJOCO_SDK/lib/libmujoco.so.3.3.5"
cmake --build --preset debug
(cd demo && CGO_ENABLED=0 go build -o ../build/debug/bin/motionbricks-demo .)
```

`MUJOCO_SDK` is your installed SDK directory; packages with a different layout
can supply the two CMake paths directly. Without this option, SONIC inference
still works and the physics API returns backend-unavailable. GGML remains the
pinned submodule; no new Nix derivations are needed by this implementation.

Fetch the pinned policy, source and robot assets using [the upstream reference
instructions](../reference/SONIC.md). They also explain how to capture a valid
upstream controller run. The extra reference image adds the tools used only
for conversion and independent tests:

```sh
docker build -f reference/Dockerfile.sonic-ggml \
  -t motionbricks-sonic-ggml-reference:onnx1.18 .
docker run --rm --entrypoint python -v "$PWD:/work" -w /work \
  motionbricks-sonic-ggml-reference:onnx1.18 reference/convert_sonic_gguf.py \
  --policy generated/sonic/policy --output generated/sonic/ggml/sonic-g1.gguf
```

The Dockerfile extends `motionbricks-sonic-reference:trt10.9`; build that base
as described in the reference instructions. Tools refuse to overwrite an
existing output. Set `SONIC_UPSTREAM` to the pinned upstream source checkout.
Export the numeric geometry/gains/scales/torque-limit configuration from a
verified upstream run (paths below are example artifact names):

```sh
docker run --rm --entrypoint python -v "$PWD:/work" \
  -v "$SONIC_UPSTREAM:/upstream:ro" -w /work \
  motionbricks-sonic-ggml-reference:onnx1.18 reference/export_sonic_physics.py \
  --run generated/sonic/baseline-007 \
  --motion generated/sonic/native-straight-verified.json \
  --motion-xml /upstream/motionbricks/assets/skeletons/g1/g1.xml \
  --output generated/sonic/ggml/g1.mbphysics
```

No network or reference Python environment is needed to run the prepared demo.

## Run the demo

```sh
./build/debug/bin/motionbricks-demo \
  -listen 127.0.0.1:8080 -library build/debug/libmotionbricks.so \
  -model generated/g1-f32 -styles generated/styles -device vulkan \
  -sonic-model generated/sonic/ggml/sonic-g1.gguf \
  -physics-scene generated/sonic/assets/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml \
  -physics-config generated/sonic/ggml/g1.mbphysics
```

Enable **Live physics · GGML SONIC**. Blue is the generated reference, green is
the actuator-driven robot, and warm target keyframes remain visible during
MotionBricks planning. Kimodo animations can be uploaded or selected from
`-kimodo-dir`; the same physics session continues through entry, playback and
the return to planning. Kimodo playback has no separate MotionBricks target
keyframes, but still shows reference and physical skeletons. The three new
flags also accept `MOTIONBRICKS_SONIC_MODEL`, `MOTIONBRICKS_PHYSICS_SCENE` and
`MOTIONBRICKS_PHYSICS_CONFIG` environment variables.

**Show collision geometry** overlays the actual MuJoCo collision hulls and foot
soles at 20% opacity; see [geometry sources and validation](COLLISION-GEOMETRY.md).

The server owns 50 Hz policy ticks and four 200 Hz physics substeps per tick;
the browser sends commands and interpolates timestamped WebSocket poses.
Planning runs ahead on a separate bounded worker. A 150 ms playback buffer
absorbs jitter without network round trips driving simulation. One browser
controls the session, with reconnect support and read-only viewers. Reset is
explicit; replanning never resets the physical robot. Falls display a warning
but do not stop physics or animation. Numerically unstable states still stop
visibly. See [streaming architecture and browser QA](STREAMING.md).

## C API

Start with the [API selection guide](API.md) and the [SONIC/physics API guide](API-SONIC-PHYSICS.md)
for the distinction between inference-only calls and the optional controller,
including observation layouts and joint ordering.

`motionbricks/sonic.h` and `motionbricks/physics.h` expose opaque handles and
flat caller-owned buffers. No public C/C++ struct layout needs reproduction in
PureGo. The model must outlive its physical sessions; each session owns its
MuJoCo state and history. Callers serialize physics-session access; SONIC
inference itself is serialized per model. Free requires no calls in flight.

Use `mb_sonic_load/encode/decode` for inference, `mb_sonic_layer` for exact
cached graph outputs, and `mb_physics_create/start/step/status` for simulation.
`mb_physics_trace` exposes actual controller/motor boundary buffers for tests.
All operations validate dimensions and numerical inputs, use fixed caller
error buffers, and catch exceptions before the ABI. Inference rejects finite
values above 1e6 in magnitude to prevent arithmetic overflow; it never clamps
observations to make a test pass.

## Reproduce tests

`capture_sonic_neural.py --policy DIR --trace RUN/boundaries.jsonl --stride 1
--output FILE` creates independent binary fixtures plus hashed JSON provenance.
Repeat `--trace` for multiple runs. Run `motionbricks-sonic-parity MODEL FILE
cpu` and `... vulkan`. Local assets enable CPU CTest cases automatically;
normal tests never download SONIC weights. Vulkan uses the existing
hardware-dependent test option and device selection.

`run_native_sonic.py --library LIB --model GGUF --scene XML --config CONFIG
--motion JSON --output JSON --trace` records native physical and controller
buffers. Pass that recording, the same motion/scene/config, the original G1
XML and official ONNX policy to `check_native_sonic.py`. See each tool's
`--help`. A failed physical rollout still writes its evidence and exits nonzero;
its boundary checker can pass independently without disguising the fall.

ASan/UBSan CPU parity and the native physical walk pass. An overflow found by
libFuzzer is covered by an ABI regression test; a subsequent 286,089-input
sanitized run passed. Enable Clang, `MOTIONBRICKS_SANITIZE` and
`MOTIONBRICKS_ENABLE_FUZZERS`. A second 365,670-input run including physical
API boundaries also passed (peak resident memory 1.16 GB). Use
`motionbricks-sonic-fuzz` with
`MOTIONBRICKS_FUZZ_SONIC` pointing at trusted local weights. Optional
`MOTIONBRICKS_FUZZ_SCENE` and `MOTIONBRICKS_FUZZ_CONFIG` add physical API input
checks. Model/XML loading is outside this fuzz surface. Project code is
instrumented; external GGML, MuJoCo and driver binaries are not thereby
instrumented. Disable leak detection where the environment prevents LSan;
address/UB checks remain enabled.

On the tested NVIDIA 595.71.05 driver, merely preloading Clang 21's ASan runtime
prevents Vulkan ICD initialization, even for the uninstrumented executable.
The current NVIDIA demo therefore uses the debug library; sanitized CPU and
AMD Vulkan tests remain available. This is not a numerical parity failure and
does not silently substitute the AMD device for the NVIDIA validation.

## Deliberate differences and remaining work

- Live simulation initializes once in the selected reference pose. The earlier
  upstream recording runner instead starts suspended and releases the robot.
  These are different initial conditions; rollout bit-equality is not claimed.
- Finite live buffers hold their last pose with zero future velocity. Offline
  upstream CSVs retain their unusual penultimate-difference terminal velocity.
  Native root SLERP is time-correct, not upstream's small-angle midpoint bug.
- Original mode 0 observes future hinge poses/velocities and base orientation,
  not reference world root translation or root velocity. Drift is displayed,
  not removed by per-frame alignment.
- Physical state feeds SONIC, **not yet MotionBricks' context/planner**. Closing
  that feedback loop requires its own independent representation tests.
- General Kimodo/jump tracking, perturbations, materials tuning, other SONIC
  modes/releases and real hardware remain separate work.

The pinned model/source identities and upstream links are recorded in
[SONIC.md](SONIC.md). Graph structure and numeric conventions were implemented
from those official exports; no upstream inference implementation was copied.
