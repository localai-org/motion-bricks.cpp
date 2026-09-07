# SONIC reference runner

Results and task checklist: [../docs/SONIC.md](../docs/SONIC.md).
Official and native walking and native stop/turn run physically, with validated
controller boundaries and browser overlays. The tested Kimodo wave/turn falls;
its failed recording is retained. These tools use the actual upstream C++ controller and Python MuJoCo
`DefaultEnv`, rather than a separately reconstructed policy loop. The only
physics-wrapper override stops on an upstream-defined fall instead of silently
resetting the robot. Body-command reads are locked during each physical step
so recorded targets correspond to the applied torques. Recording FK uses a
separate MuJoCo data object and never writes back to physical state.

## Provision

Prerequisites: Linux x86-64, Docker with NVIDIA GPU support, and the pinned
GR00T-WholeBodyControl source checkout. The image extends the existing reference
image documented in [README.md](README.md); it does not require a Nix build.

From the project root:

```sh
export SONIC_UPSTREAM=/path/to/GR00T-WholeBodyControl
docker build -f reference/Dockerfile.sonic \
  --build-arg SONIC_UID="$(id -u)" --build-arg SONIC_GID="$(id -g)" \
  -t motionbricks-sonic-reference:trt10.9 reference

docker run --rm --user "$(id -u):$(id -g)" --entrypoint python \
  -e HF_HOME=/work/generated/sonic/hf-cache \
  -v "$PWD:/work" -v "$SONIC_UPSTREAM:/upstream:ro" \
  motionbricks-sonic-reference:trt10.9 /work/reference/fetch_sonic.py \
  --upstream-root /upstream --output /work/generated/sonic

docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/work" -v "$SONIC_UPSTREAM:/upstream:ro" \
  motionbricks-sonic-reference:trt10.9 \
  /work/reference/build_sonic.sh /upstream /work/generated/sonic
```

The original checkout is read-only. The build exports a copy of the pinned
source and supplies verified SDK binaries instead of invoking Git LFS on the
original worktree. `instrument_sonic.py` adds observation-only JSON logging
after the original motor-command calculation; `instrumentation.json` records
original/snippet/generated hashes. No observation or policy math is replaced.
The controller is built with ASan/UBSan, one compilation job
by default, and fast-math disabled. TensorRT is 10.9 with CUDA 12.8; ONNX Runtime
is 1.22.0; CycloneDDS is 0.10.2. These are reference-only dependencies.
The image currently uses the distribution's GoogleTest package for configuring
upstream's build; no replacement policy implementation is involved.

Artifacts, the full content manifest and build output live under ignored
`generated/sonic/`. Model and robot assets retain their respective upstream
licenses; this setup does not authorize redistributing them under our license.

## Run the physical baseline

Do not interpret a
zero process status as tracking acceptance: the saved summary explicitly says
tracking has not yet been evaluated. Run the separate evaluator below.

```sh
docker run --rm --device=nvidia.com/gpu=all --user "$(id -u):$(id -g)" \
  --entrypoint python \
  -v "$PWD:/work" -v "$SONIC_UPSTREAM:/upstream:ro" \
  motionbricks-sonic-reference:trt10.9 /work/reference/run_sonic_baseline.py \
  --upstream-root /upstream --bundle /work/generated/sonic \
  --binary /work/generated/sonic/source/gear_sonic_deploy/target/release/g1_deploy_onnx_ref \
  --output /work/generated/sonic/baseline-001 --settle-seconds 0
```

Do not use host networking: the DDS controller and simulator communicate over
loopback in the same container, isolated from hardware and unrelated services.
On systems using the legacy NVIDIA Docker runtime instead of CDI, substitute
`--gpus all` for `--device=nvidia.com/gpu=all`.
Each run requires a new output directory. The initial joint pose uses upstream's
default standing angles. Physics starts at CONTROL activation, not at the
earlier initialization motor commands. Sensors are published while frozen,
so the robot cannot fold during initialization or TensorRT engine loading.
Startup automates upstream's
initialization, start-control, suspension removal and play controls. The runner
saves controller logs, native policy-input/state/target CSVs, simulation events
and physical NPZ data. Leak checking is disabled for the vendor GPU runtime;
ASan memory-access checking and UBSan halt-on-error remain enabled.
The child also uses `protect_shadow_gap=0`: the minimal
`sonic_cuda_probe.cpp` reproduced CUDA initialization error 2 under default
ASan protection despite ample free VRAM; allowing CUDA's address reservations
fixes initialization without removing the sanitizer instrumentation.

Run offline diagnostics and the small provisioning tests with:

```sh
docker run --rm --user "$(id -u):$(id -g)" --entrypoint python \
  -v "$PWD:/work" motionbricks-sonic-reference:trt10.9 \
  /work/reference/analyze_sonic.py --run /work/generated/sonic/baseline-001
docker run --rm --user "$(id -u):$(id -g)" --entrypoint python \
  -v "$PWD:/work" motionbricks-sonic-reference:trt10.9 \
  /work/reference/test_sonic_tools.py
```

The diagnostics reject failed/suspended/non-finite runs and mismatched control
indices instead of silently truncating arrays. Joint RMSE compares actual and
reference angles in matching upstream hardware order. Runs with PD-input
recordings also verify targets/gains/feedforward torque against the actual
clipped body torques, and report actuator saturation. The full evaluator also
checks observation/action timing and geometric contact-point foot slip.

## Record native MotionBricks input

With the normal MotionBricks demo running:

```sh
python reference/capture_native_walk.py --url http://127.0.0.1:8080 \
  --server-pid "$MOTIONBRICKS_SERVER_PID" \
  --seconds 10 --output generated/sonic/native-walk.json
```

This creates a separate session and captures 300 frames with 16-frame replanning
cadence, full API responses and command seeds. It does not alter existing
browser sessions. Set `MOTIONBRICKS_SERVER_PID` to the existing local demo PID.
The capture verifies listener ownership and mapped-library identity, and hashes
the model/style/library/server before and after capture. It records backend,
sampling mode and seeds. Remote captures without `--server-pid` lack that local
identity verification. Use `--pattern stop-turn --seconds 12` for the stop/turn
case, or `--pattern kimodo-return --kimodo-id CLIP_ID --seconds 16` for an
explicit authored clip between native walking phases. This JSON is native input; convert and
validate it separately as below.

## Convert and validate the native recording

The Python adapter needs NumPy/SciPy/MuJoCo, already present in the reference
image. These are offline/reference tools, not native inference dependencies.
For shorter commands, define a reference-container helper:

```sh
sonic_python() {
  docker run --rm --user "$(id -u):$(id -g)" --entrypoint python \
    -v "$PWD:/work" -v "$SONIC_UPSTREAM:/upstream:ro" \
    motionbricks-sonic-reference:trt10.9 "$@"
}

sonic_python /work/reference/export_sonic_motion.py \
  --input /work/generated/sonic/native-walk.json \
  --motion-xml /upstream/motionbricks/assets/skeletons/g1/g1.xml \
  --scene /work/generated/sonic/assets/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml \
  --output /work/generated/sonic/native-motions/walk

docker run --rm --user "$(id -u):$(id -g)" --entrypoint bash \
  -v "$PWD:/work" -v "$SONIC_UPSTREAM:/upstream:ro" \
  motionbricks-sonic-reference:trt10.9 -c '
  g++ -std=c++17 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
    -I/upstream/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include \
    /work/reference/sonic_reader_probe.cpp -o /work/generated/sonic/sonic-reader-probe &&
  ASAN_OPTIONS=halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
    /work/generated/sonic/sonic-reader-probe /work/generated/sonic/native-motions \
    /work/generated/sonic/native-reader.json'

sonic_python /work/reference/validate_sonic_adapter.py \
  --upstream-root /upstream --input /work/generated/sonic/native-walk.json \
  --export /work/generated/sonic/native-motions/walk \
  --reader-output /work/generated/sonic/native-reader.json \
  --output /work/generated/sonic/native-adapter-validation.json

sonic_python -m unittest discover -s /work/reference -p 'test_sonic_*.py'
```

The hinge projection follows the actual upstream MotionBricks converter,
including nonidentity XML rest rotations and virtual endpoint removal. Its
independent implementation is checked against the real upstream skeleton,
representation and converter, not the shape-only neural fixture helper.
Resampling uses shortest-arc SLERP and explicit half-open timestamps. The
upstream small-angle interpolation fallback ignores the requested time and
returns the midpoint; that known difference is explicitly bounded/reported,
not described as exact parity. Original numerical velocity/interpolation
methods are executed unchanged from verified source ASTs to avoid importing
unrelated mesh/training dependencies. Attribution is in the adapter/validator.

Run the same physical-baseline command with
`--motions /work/generated/sonic/native-motions` and a fresh `--output` directory.
Keep each motions directory to the one clip intended for the run.

## Full-run evaluation gate

After adapter validation and a successful physical run:

```sh
sonic_python /work/reference/evaluate_sonic.py \
  --run /work/generated/sonic/native-walk-001 \
  --clip /work/generated/sonic/native-motions/walk \
  --scene /work/generated/sonic/assets/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml \
  --output /work/generated/sonic/playback/native-walk.json \
  --title 'Native MotionBricks walk · SONIC / MuJoCo'
```

For the official baseline, substitute its run directory and clip
`assets/gear_sonic_deploy/reference/example/walking_quip_360_R_002__A428`.
The evaluator requires the requested playback to finish, clean controller and
sanitizer exit, finite unassisted state, exact/tolerance-based buffer checks,
actual physical target delivery, PD-to-torque agreement and checked contact
Jacobians. It writes versioned `acceptance.json` with artifact hashes and
returns nonzero on failure. Root/body drift and slip remain measured quality
metrics, not a universal quality guarantee. Deadline counts are reported.

The constituent tools are `check_sonic_boundaries.py`, `sonic_contacts.py` and
`sonic_playback.py`. Numerical tolerances are declared in the checker/validator
source and summarized in the task document. Full neural CPU-versus-TensorRT
numerical parity is not claimed; the policy here is the upstream implementation.
The controller CSV reader and controller itself run under ASan/UBSan.

## Browser playback and diagnostics

```sh
sonic_python /work/reference/sonic_playback.py \
  --run /work/generated/sonic/native-walk-001 \
  --scene /work/generated/sonic/assets/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml \
  --output /work/generated/sonic/playback/native-walk.json \
  --title 'Native MotionBricks walk · SONIC / MuJoCo'
```

Add `-physics-dir generated/sonic/playback` to the normal demo launch. Choose a
recording in **Playback mode**, or open `/?physics=native-walk`. This is recorded
simulation, not live physics. The normal interactive mode remains available.
Green is actual physical motion, blue is the reference, and pink links expose
error. Pause/replay, scrub, orbit, zoom and reference visibility are supported.
Only one initial heading/XY alignment is applied; drift is not removed.
For failure diagnosis only, add `--include-failed` to `sonic_playback.py` and
use a visibly labelled title. The saved failure is shown in the UI. This flag
is never used by the acceptance evaluator. Shutdown may leave one final policy row beyond the last physical sample; the
exporter reports/excludes that unbracketed endpoint rather than extrapolating.

For headless QA with the actual recordings (no model inference required):

```sh
cd demo
MOTIONBRICKS_PHYSICS_TEST_DIR="$PWD/../generated/sonic/playback" \
MOTIONBRICKS_QA_DIR="$PWD/../generated/sonic/browser-qa" \
CGO_ENABLED=0 go test -run 'TestPhysics|TestCameraFollow' -v .
```

Set `MOTIONBRICKS_PHYSICS_TEST_ID=kimodo-failed` to exercise failure playback
and its explicit failure warning; the default is `native-walk`.

With corrected startup the native walk repeats complete 9.8 seconds without
falling, with 0.617/0.337 m final root drift and 0.110/0.105 rad joint RMSE.
The stop/turn run also completes; the Kimodo wave/turn falls during the authored
clip, before the planned return to native walking. G1 mode 0 does not directly observe the
reference's world root translation/velocity; do not equate successful physical
playback with exact trajectory tracking or neural parity.
