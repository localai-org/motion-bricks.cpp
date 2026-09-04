# Upstream reference

The native implementation is validated against a pinned, unmodified NVIDIA
MotionBricks checkout.

| Component | Repository | Revision |
|---|---|---|
| MotionBricks | `https://github.com/NVlabs/GR00T-WholeBodyControl.git` | `a0732b642c0333077e127a2f56ab0014c196bca4` |
| GGML | `https://github.com/ggml-org/ggml.git` | `8c63e70982c95ceb862e3a1073a2c1beef75d60a` (`v0.20.2`) |

The `ggml/` submodule is pinned to the same revision used by the local
`kimodo.cpp` and `skin-tokens.cpp` reference ports. Compatibility patches, if
required, belong under `patches/ggml/` and will be applied to a build-directory
copy during CMake configuration; the submodule worktree will remain untouched.

## Checkpoint identities

The required MotionBricks checkpoints and G1 meshes have been fetched through
Git LFS. These identities are the initial trusted reference inputs:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `G1-clip.ckpt` | 7,761,041 | `84afc7c229473351a24b0a7d79fc47be9dbb81bd12774285f2f60a3c0e9028df` |
| VQ-VAE `model-step=2000000.ckpt` | 285,607,244 | `f12a09d46ad390a8e2eecbe7219b2472fcab6b59df0a13f6a40c35cb6da4d99a` |
| pose `model-step=2000000.ckpt` | 1,639,126,476 | `0223c352b308ba638a499cc5c92104da36cb1d04cea2f8ce61d54a2489f853f1` |
| root `model-step=2000000.ckpt` | 409,551,754 | `d7299a9b1f5aca35730c36dfe7ea28075708ac8266bf384fe8e2ef9c9aee69c7` |

Saved configuration identities:

| Configuration | SHA-256 |
|---|---|
| VQ-VAE `config.yaml` | `027a2d7ba5f49cadeadbcc9a6b0c6784d657d5a842e41ff820c5233f1cd6c1f3` |
| pose `config.yaml` | `273af770d328b458510ee6049fac8e38fa486c8792cc27b3515dddc094a7cd1f` |
| root `config.yaml` | `b174f03a333f7f7857c3e2b8a32da517caddd198f228efeea2e203d046ce212a` |

The 29-DOF G1 XML identities used by the initial interactive reference are:

- `g1_29dof.xml`: `58660a6f1d0d33ffd8ee967ab3860def53e3327d956cb009dd1385ddaf430f56`
- `scene_29dof.xml`: `e254f11acce2ec6f6efa5bf9b15e288bbd0ca29aeeabb1e1d0fea92f65436bbf`

Legacy Lightning `.ckpt` and demo-clip files are deserialized only inside the
reference container. Normal conversion accepts safetensors and JSON, never
pickle-bearing checkpoints.

Build and run the trusted extractor from the project root:

```sh
docker build -t motionbricks-reference:torch2.4 reference
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/work" \
  -v /path/to/GR00T-WholeBodyControl:/upstream:ro \
  motionbricks-reference:torch2.4 \
  --upstream-root /upstream \
  --output /work/generated/safe
```

The extractor verifies every pinned input hash and the small allowlist of
pickle globals before loading. It writes independent safetensors for the three
model checkpoints, original demo clips, and shared skeleton/statistics, plus a
manifest containing every tensor name, shape, dtype, value count, and hash.

## Native inference bundle

The normal converter reads only those verified safetensors and JSON files:

```sh
nix develop --command python scripts/convert_to_gguf.py \
  --safe-directory generated/safe \
  --output generated/g1-f32
```

It emits three inference-weight components plus one small skeleton/statistics
component. The VQ-VAE encoder and optimizer/training state are excluded. The
resulting learned parameter inventory is:

- pose planner: 136,588,272
- root planner: 34,122,833
- VQ pose decoder and codebook: 12,437,277
- total: 183,148,382

Validate the result through the same public model-loading path applications
use:

```sh
./build/debug/bin/motionbricks-cli inspect generated/g1-f32
```

## Reference fixtures and styles

Generate deterministic PyTorch layer fixtures inside the same pinned
container, then package them as small GGUF test inputs:

```sh
docker run --rm --user "$(id -u):$(id -g)" \
  --entrypoint python \
  -v "$PWD:/work" \
  -v /path/to/GR00T-WholeBodyControl:/upstream:ro \
  motionbricks-reference:torch2.4 \
  /work/reference/generate_fixtures.py \
  --upstream-root /upstream \
  --safe-directory /work/generated/safe \
  --output /work/generated/fixtures

python scripts/convert_fixtures_to_gguf.py \
  --fixtures generated/fixtures \
  --output generated/fixtures-gguf

python scripts/convert_styles.py \
  --safe-directory generated/safe \
  --output generated/styles
```

The native suite checks every released neural component against these actual
upstream PyTorch forwards. On the current reference machine, complete
style-to-animation output also matches between CPU and strict-F32 Vulkan with
the same 44-frame duration; observed maximum absolute differences were
`2.19e-5` for root translations and `1.02e-4` for local quaternion components.

## Automated black-box session baseline

Session capture uses a separate CUDA image because the released upstream
navigation demo constructs its agent on CUDA. The image is pinned independently
from the smaller CPU-only checkpoint extraction image. Its PyTorch 2.7/CUDA
12.8 base supports both the released model and Blackwell GPUs:

```sh
docker build -f reference/Dockerfile.session \
  -t motionbricks-reference-session:torch2.7 reference
```

Validate the image, upstream source, package imports, and every trusted input
without starting a model or requiring a GPU:

```sh
docker run --rm \
  -v "$PWD:/work:ro" \
  -v /path/to/GR00T-WholeBodyControl:/upstream:ro \
  motionbricks-reference-session:torch2.7 \
  --upstream-root /upstream --preflight-only
```

The preflight verifies every version in the Python lock, inventories the
resolved OS packages, checks the upstream Git revision, refuses modified
tracked source, and verifies every checkpoint, configuration, and skeleton
input before loading the demo. Run the documented 345-frame scenario three
times in fresh Python processes with the upstream checkout mounted read-only:

```sh
docker run --rm --device=nvidia.com/gpu=all --user "$(id -u):$(id -g)" \
  -v "$PWD:/work" \
  -v /path/to/GR00T-WholeBodyControl:/upstream:ro \
  motionbricks-reference-session:torch2.7 \
  --upstream-root /upstream \
  --output /work/generated/session-baseline
```

On Docker installations configured with the legacy NVIDIA runtime rather than
CDI, use `--gpus all` in place of `--device=nvidia.com/gpu=all`.

The driver preserves upstream ordering: it emits and advances a playback frame,
then applies that logical frame's command to a possible future buffer. It uses a
fixed camera proxy, with a deterministic 90-degree azimuth ramp for the turn,
and explicit per-plan seeds. It writes three `run-NNN` directories followed by
`comparison.json`. Each run contains:

```text
manifest.json
controls.jsonl
events.jsonl
playback.safetensors
plan-NNN.safetensors
```

Discrete tensors and JSON event streams must agree exactly. By default, every
pair of repeated CUDA runs must remain within `1e-5` maximum absolute error and
`1e-6` symmetric relative L2 error; the worst pairwise values are retained in
`comparison.json`. These are repeatability limits, not the later
PyTorch-to-C++ parity tolerances.

The base recorder has a 25 MiB per-run artifact ceiling and does not edit
upstream, render video, or access neural-layer internals.

Run the pure scenario and synthetic comparator tests without a GPU:

```sh
nix develop --command python -m unittest reference/test_session_tools.py
```

## Target trace and cross-runtime visual replay

Add `--trace-targets` to the three-run command above and use a separate output
directory such as `generated/session-traced`. The option installs one
instance-local wrapper around the target-transform method. It invokes the
original method first, then copies only its returned target tensors plus the
spring inputs and canonical world origin/heading. The upstream checkout stays
read-only and every public replan must produce exactly one boundary record.

Verify that tracing did not affect accepted outputs at zero tolerance:

```sh
nix develop --command python reference/compare_session_captures.py \
  generated/session-baseline/run-000 generated/session-traced/run-000 \
  --core-only --max-abs 0 --max-relative-l2 0 \
  --output generated/session-traced/non-interference.json
```

`--core-only` still compares the pinned upstream identity, scenario, seeds,
demo flags, CUDA environment, JSON controls/events, qpos, model features, and
all playback tensors. It permits the trace tensor superset and ignores only
the changed harness identity and renderer OS-package inventory. The three
traced runs are also compared in full, including every trace tensor.

Build the shared replay artifact. The builder evaluates MuJoCo forward
kinematics for the emitted qpos and restores every target from its per-plan
canonical frame into world space:

```sh
docker run --rm --user "$(id -u):$(id -g)" --entrypoint python \
  -e PYTHONPATH=/work/reference \
  -v "$PWD:/work" \
  -v /path/to/GR00T-WholeBodyControl:/upstream:ro \
  motionbricks-reference-session:torch2.7 \
  /work/reference/build_session_replay.py \
  --capture /work/generated/session-traced/run-000 \
  --upstream-root /upstream \
  --output /work/generated/session-replay

./build/debug/bin/motionbricks-cli replay-info \
  generated/session-replay/session.mbreplay
```

Render the authoritative upstream view and encode its 345 PNGs into an H.264
MP4. Both commands refuse to replace an existing artifact directory/file:

```sh
docker run --rm --user "$(id -u):$(id -g)" --entrypoint python \
  -e MUJOCO_GL=osmesa -e PYTHONPATH=/work/reference \
  -v "$PWD:/work" \
  -v /path/to/GR00T-WholeBodyControl:/upstream:ro \
  motionbricks-reference-session:torch2.7 \
  /work/reference/render_session_replay.py \
  --replay /work/generated/session-replay/session.mbreplay \
  --manifest /work/generated/session-replay/manifest.json \
  --upstream-root /upstream \
  --output /work/generated/session-render

nix develop --command ffmpeg -framerate 30 \
  -i generated/session-render/frames/frame-%04d.png \
  -c:v libx264 -pix_fmt yuv420p -crf 18 -movflags +faststart \
  generated/session-render/upstream-mujoco.mp4
```

The MP4 and selected snapshots show the opaque upstream G1, four distinct
target ghosts, the current target path, the recent animated-root trail, and
frame/plan/mode overlays. Its camera follows the animated qpos root only.
Start the Go/Three.js viewer with `-replay generated/session-replay/session.mbreplay`
as documented in `docs/DEMO.md`; it consumes the identical binary without
loading either planner. Generated captures, binaries, PNGs, and MP4s are
intentionally ignored by Git.

## Open-loop native parity

Capture every neural boundary once, together with the already accepted target
boundary. The upstream checkout remains read-only. Prove that these external
wrappers and forward hooks do not perturb any public result:

```sh
docker run --rm --device=nvidia.com/gpu=all \
  --user "$(id -u):$(id -g)" --entrypoint python \
  -e PYTHONPATH=/work/reference \
  -v "$PWD:/work" \
  -v /path/to/GR00T-WholeBodyControl:/upstream:ro \
  motionbricks-reference-session:torch2.7 \
  /work/reference/capture_session.py \
  --upstream-root /upstream \
  --output /work/generated/session-neural-all \
  --seed 1234 --artifact-limit-mb 40 \
  --trace-targets --trace-neural-all

python reference/compare_session_captures.py \
  generated/session-traced/run-000 generated/session-neural-all \
  --core-only --max-abs 0 --max-relative-l2 0 \
  --output generated/session-neural-all/non-interference.json
```

The demo's accepted observation was produced on CUDA. For strict CPU parity,
replay those exact sparse inputs through the same upstream PyTorch models on
CPU. The container initially loads the unmodified CUDA-only demo, then moves
the inference modules to CPU before any replayed plan is evaluated:

```sh
docker run --rm --device=nvidia.com/gpu=all \
  --user "$(id -u):$(id -g)" --entrypoint python \
  -e PYTHONPATH=/work/reference \
  -v "$PWD:/work" \
  -v /path/to/GR00T-WholeBodyControl:/upstream:ro \
  motionbricks-reference-session:torch2.7 \
  /work/reference/replay_inference_trace.py \
  --upstream-root /upstream \
  --capture /work/generated/session-neural-all \
  --output /work/generated/upstream-cpu-replay
```

Build the compact fixture. MuJoCo is used only to adapt recorded qpos context
to the public 34-joint contract; expected motion comes from the verified
upstream CPU replay:

```sh
docker run --rm --user "$(id -u):$(id -g)" --entrypoint python \
  -e PYTHONPATH=/work/reference \
  -v "$PWD:/work" \
  -v /path/to/GR00T-WholeBodyControl:/upstream:ro \
  motionbricks-reference-session:torch2.7 \
  /work/reference/build_plan_parity.py \
  --capture /work/generated/session-neural-all \
  --upstream-root /upstream \
  --support /work/generated/safe/support.safetensors \
  --expected-replay /work/generated/upstream-cpu-replay \
  --output /work/generated/open-loop-parity
```

Run all plans independently. Omitting `--report-only` makes this a real strict
gate: any failed ceiling produces a nonzero exit. `--trace-directory` performs
a separate diagnostic evaluation and writes one native trace per plan without
changing the public C-API result used by the report:

```sh
./build/debug/bin/motionbricks-parity \
  generated/open-loop-parity/open-loop.mbparity \
  generated/g1-f32 generated/styles \
  generated/open-loop-parity/cpu-report.json cpu \
  --trace-directory generated/open-loop-parity/native-traces

python reference/compare_neural_trace.py \
  generated/session-neural-all generated/open-loop-parity/native-traces \
  --output generated/open-loop-parity/boundary-comparison.json
```

The strict CPU report passes all 14 plans. Duration is exact, placed target FK
matches within 4.5 micrometres, and worst output errors are 0.20 mm root,
0.23 mm FK, and 0.034 degrees local rotation. This is well inside the unchanged
10 mm / 20 mm / 2 degree ceilings.

The original CUDA capture is retained as a separate cross-device observation.
On the same captured pose inputs, upstream PyTorch 2.7 on CPU chooses 23 of
1,136 pose codes differently from CUDA. TF32 is disabled and matmul precision
is `highest`; the differences are ordinary accumulated backend arithmetic near
discrete argmax ties. `replay_pose_trace.py` reproduces and records this fact.
Accordingly, the CUDA boundary comparison is diagnostic and may remain red;
it is not used to weaken or redefine strict CPU parity.

### CPU/Vulkan parity

Vulkan is compared to the strict CPU runtime rather than directly to the CUDA
observation. Configure the opt-in hardware test with an explicit device and
hardware tag, then run the complete suite, including the opt-in component and
all-plan Vulkan gates:

```sh
cmake --preset debug \
  -DMOTIONBRICKS_ENABLE_VULKAN_PARITY_TESTS=ON \
  -DMOTIONBRICKS_VULKAN_PARITY_DEVICE=0 \
  -DMOTIONBRICKS_VULKAN_PARITY_HARDWARE="NVIDIA GeForce RTX 5070 Ti" \
  -DMOTIONBRICKS_VULKAN_PARITY_DRIVER=595.71.05
cmake --build --preset debug
ctest --test-dir build/debug --output-on-failure
```

The CPU and Vulkan runners both evaluate the 14-plan CPU-reference fixture and
emit paired neural traces. `compare_plan_reports.py` checks exact durations and
the public animation using direct backend ceilings of 1 mm root, 2 mm FK,
0.2 degrees local rotation, 0.1 mm target root, 0.2 mm target FK, and
0.05 degrees target rotation. These limits are at least an order of magnitude
tighter than the main root/FK/rotation upstream-behavior gate, but allow errors
to accumulate across each 24--44-frame decoded trajectory.

On the NVIDIA GeForce RTX 5070 Ti with driver 595.71.05, two fresh Vulkan runs
were byte-identical. All 14 durations and all 1,136 pose tokens matched CPU.
Worst CPU/Vulkan differences were 0.028 mm root, 0.041 mm FK, and 0.0121 degrees
local rotation; placed targets were identical. Continuous internal differences
(up to 0.0299 duration-logit, 0.0228 pose-logit, and 0.00224 normalized decoder
output) are recorded diagnostically and do not replace the observable-output
gate.

The fixture is open loop: each native plan receives the recorded four-frame
upstream context, transformed into the public 34-joint animation contract.
Consequently errors cannot compound from one plan into the next. The report
compares duration exactly, roots by Euclidean distance, local XYZW rotations by
sign-invariant angular distance, and skeleton/targets by FK position. The
all-plan diagnostic additionally compares sparse inputs, root outputs, pose
logits/tokens, decoder inputs, and decoder outputs.

Start the report viewer with `-comparison` as documented in `docs/DEMO.md`.
It overlays upstream and native rigs and draws a red vector for every joint's
current positional error.
