# Upstream reference

## SOMA reference and current support

`soma-reference.json` pins the standalone NVIDIA source, Hugging Face revision,
FP32 SafeTensors, rig, and official FP32/mixed-FP16 ONNX exports. This is separate
from the original G1 reference below. Native SOMA inference, a public C API,
independent layer/processing captures, and chained demo overlays are implemented.
FP32 public chains and internal batches pass on CPU and Vulkan with exact
exported normalization scales. Calibrated CPU mixed-FP16 chains and batches pass. The qualified Vulkan
mixed-FP16 public mode passes all three captured chains on RTX 5070 Ti with
driver 595.71.05, including CPU-first loading and FP32 restoration. The sections
below retain the numerical investigation history; the latest evidence is at the
end. See [the API guide](../docs/API-SOMA.md) for the current support matrix.

Use the extended bundle for current parity tests:

```sh
.cache/soma-reference-venv/bin/python scripts/convert_soma_to_gguf.py \
  --reference generated/soma/reference \
  --mixed-onnx generated/soma/reference/motionbricks/assets/onnx/soma/mmm_v6_soma_fp16.onnx \
  --output generated/soma/export-scales
```

CMake's `MOTIONBRICKS_SOMA_TEST_BUNDLE` defaults to this directory. Override it
explicitly to test a different bundle, including the older four-tensor format.
The unchanged strict Vulkan stress test fails with that older bundle. Set
`MOTIONBRICKS_ENABLE_VULKAN_PARITY_TESTS=ON` to register hardware checks and
`MOTIONBRICKS_ENABLE_SOMA_CALIBRATED_TESTS=ON` for the calibrated CPU tests.

The reference environment uses Python 3.12 and the pinned upstream exporter:

```sh
uv venv --python 3.12 .cache/soma-reference-venv
uv pip install --python .cache/soma-reference-venv/bin/python \
  -e /path/to/pinned-motionbricks/tools/onnx onnxruntime==1.32.0
python reference/fetch_soma.py --output generated/soma/reference --onnx
```

Materialize the verified `tools/onnx/assets/soma_model_rig.npz` in the upstream
checkout (it is a Git LFS file). The capture script verifies its content and the
exporter revision before running. Input NPZs contain the seven named SOMA ONNX
inputs, with FP32 feature tensors and INT32 masks/token-count flags.

```sh
.cache/soma-reference-venv/bin/python reference/capture_soma.py \
  --upstream /path/to/pinned-motionbricks \
  --checkpoints generated/soma/reference/soma \
  --inputs /path/to/inputs.npz --output generated/soma/pytorch-capture
.cache/soma-reference-venv/bin/python reference/capture_soma_onnx.py \
  --model generated/soma/reference/motionbricks/assets/onnx/soma/mmm_v6_soma_fp32.onnx \
  --inputs /path/to/inputs.npz --output generated/soma/onnx-capture --tap-layers
.cache/soma-reference-venv/bin/python reference/compare_soma.py \
  --reference generated/soma/pytorch-capture --actual generated/soma/onnx-capture \
  --output generated/soma/comparison.json
.cache/soma-reference-venv/bin/python -m unittest discover -s reference -p test_soma_tools.py
```

CTest also registers `motionbricks-reference-soma-tools` when its interpreter
can import NumPy and PyTorch. To use the dedicated reference environment, set
`-DMOTIONBRICKS_SOMA_REFERENCE_PYTHON=/absolute/path/to/.cache/soma-reference-venv/bin/python`
when configuring CMake. This covers capture immutability, root-decision taps,
masked layer comparisons, and rejection of invalid chain acceptance results.
The full configured CPU integration suite (35 tests, before this additional
Python test was registered) passes with the SOMA implementation, including G1,
shared-library/Go ABI, demo, and SONIC regression tests.

PyTorch captures record module calls, representation operations, T-pose
conversions, codebook lookup and orchestration boundaries. ONNX layer captures
expose every inferred node output, including functional attention projections.
Both check that tracing leaves the observable output unchanged. PyTorch fused
versus explicit transformer differences are reported separately. Module
inventories explicitly identify modules not called through `forward`.

Mixed-FP16 final-output capture works with the `_fp16.onnx` artifact. Exposing
source FP16 intermediates changes ONNX Runtime CPU execution even with graph
optimizations disabled; ordinary `--tap-layers` correctly rejects that capture.
`--tap-layers --tap-executed-graph` instead captures the runtime-lowered graph,
checks bit-exact final-output equivalence, and saves that graph with a SHA-256.
The one-target walking case captures 5,142 tensors without changing either output.

`audit_soma_precision.py` records declared versus executed node types and all
precision-changing casts. The official source declares 42 FP16 convolutions,
25 FP16 Gemm operations and 144 FP16 neural MatMuls, retaining FP32 normalization
and softmax. ONNX Runtime 1.32 CPU promotes all of those arithmetic operations
to FP32 while retaining some FP16 data boundaries. Its executed taps therefore
must not be described as pure FP16 arithmetic. The supported native mixed modes
match the captured CUDA profile within the documented hardware/calibration scope;
FP32 traces and half-weight storage are not substitutes for that contract.

For the mixed-FP16 implementation target, the CUDA provider can capture every
source graph tensor without changing final outputs. On the RTX 5070 Ti, the
one-target walking request captures 7,179 tensors with zero instrumentation error.
The execution profile confirms all 42 Conv, 25 Gemm, 148 MatMul, 48 LayerNormalization
and 24 Softmax nodes execute on CUDA. Neural Conv/Gemm/MatMul outputs have the
upstream-declared FP16 types; normalization, softmax and geometry retain their
upstream FP32 boundaries. Four MatMuls belong to FP32 geometry.

Use a separate environment so GPU dependencies do not alter the PyTorch CPU oracle:

```sh
uv venv --python 3.12 .cache/soma-gpu-reference-venv
uv pip install --python .cache/soma-gpu-reference-venv/bin/python \
  'onnxruntime-gpu[cuda,cudnn]==1.32.0' onnx==1.23.2 numpy==1.26.4
nix develop --command env -u PYTHONPATH .cache/soma-gpu-reference-venv/bin/python \
  reference/capture_soma_onnx.py \
  --model generated/soma/reference/motionbricks/assets/onnx/soma/mmm_v6_soma_fp16.onnx \
  --inputs generated/soma/walk-one-reference/inputs.npz \
  --output generated/soma/walk-one-fp16-cuda --provider cuda --tap-layers
```

The verified GPU environment resolved CUDA runtime/NVRTC 13.5.37, cuBLAS
13.10.0.20, cuDNN-cu13 9.28.1.4, cuFFT 12.5.0.14, cuRAND 10.4.5.25 and
nvJitLink 13.5.37. Captures record execution-provider placement and reject failure
to initialize the requested provider. This establishes the independent reference;
current native parity results and scope are recorded at the end of this document.

New ONNX captures also record the selected CUDA device and compute capability,
driver API version, loaded CUDA library names (including the exact driver library),
Python/NumPy/ONNX and installed CUDA package versions, and an allowlist of arithmetic
environment settings. Chain steps and mixed-precision comparisons preserve this
runtime identity when available; older captures explicitly lack it. A fresh
four-target walking capture on RTX 5070 Ti / compute capability 12.0 / driver
`libcuda.so.595.71.05` reproduced all 7,179 earlier tensors exactly. Its manifest
and repeatability record are in `walk-four16-fp16-cuda-provenance`. This verifies
repeatability of that fixture in this environment, not portability across GPUs.

An experimental native path now applies IEEE half rounding at neural inputs,
weights and arithmetic outputs, with FP32 accumulation, normalization and geometry.
The pose wrapper's positional embedding and its addition remain FP32, matching
the official graph's `/mmm_net_1/` boundary. The transformer casts the resulting
sum to half. This differs from the root wrapper's precision policy.
It retains separate MatMul/Add and fused Gemm behavior. Convolutions round their
product to half before a separate half bias addition, matching the captured CUDA
execution. It also
scales attention inputs before their matrix multiplication, and rounds CUDA's
LeakyReLU coefficient to half. It uses the existing FP32 bundle and casts its
neural weights explicitly; this is not a claim of FP16 hardware acceleration.

```sh
build/sonic-cpu/tests/motionbricks-soma-run generated/soma/fp32 \
  generated/soma/walk-one-io/playback.gguf generated/soma/native-half.gguf cpu mixed-fp16
.cache/soma-reference-venv/bin/python reference/compare_soma_mixed.py \
  --reference generated/soma/walk-one-fp16-cuda --native generated/soma/native-half.gguf \
  --output generated/soma/native-half-errors.json
```

This path is explicitly **not at parity**. Before the convolution bias correction,
on the one-target walking case, duration
matches but CPU has two pose-token mismatches and Vulkan has one. Maximum raw
joint rotation differences are about 5.10 and 3.67 degrees respectively. The first
root-projection inputs match after half rounding, while CUDA and FP32-accumulated
matrix reductions already differ in a few half results. The discrepancy is under
investigation; these errors have not been accepted by widening a tolerance.
Experimental runs also write named layer diagnostics alongside the output GGUF.

Masked selection now multiplies each finite operand by its binary selector before
adding. The previous `fallback + mask*(selected-fallback)` lowering could erase
small selected values through F32 cancellation. The mixed-pose probe checks
exact preservation of half-representable values as small as `2^-20` against
fallback magnitudes of 32, including the broadcast-mask path, on CPU and Vulkan.
This corrects the arithmetic contract but does not eliminate the walking fixture's
remaining mixed-FP16 token mismatches.

`export_soma_mixed_pose.py` exports exact CUDA-captured pose inputs so the native
pose network can be checked without propagating root-predictor differences.
`motionbricks-soma-mixed-pose-probe` tests the single and batch-size-one paths and
asserts the positional addition is exactly FP32 using native intermediate taps.
An optional second fixture also tests a true batch of two different requests in
both orders, recording each request's reference error, single-versus-batch error,
and order sensitivity. The one-target and four-target walking fixtures (both
16 tokens) match all 160 upstream pose-token choices on CPU and Vulkan. CPU
batch logits equal the single-request outputs exactly; Vulkan differs by up to
0.03125. Reversing batch order leaves both backends' outputs unchanged. Maximum
reference logit error across the two requests is 0.0625 on either backend.
This covers the isolated pose network, not full-pipeline mixed-FP16 batching.
On the 16-token walking fixture both CPU and Vulkan match all 80 pose-token
choices with these identical inputs. Maximum logit errors are 0.0625 and 0.03125
respectively, versus 1.05 and 1.28 in the full pipeline after the precision fix.
This isolates upstream/root propagation as a major remaining source of mismatch;
it is not a full-model parity pass. Before the convolution bias correction, the three-transition chain still
has one closed-loop token mismatch per backend. Vulkan replay now has two
mismatches, so this semantic correction is not claimed as a universal error reduction.

```sh
.cache/soma-reference-venv/bin/python reference/export_soma_mixed_pose.py \
  --capture generated/soma/walk-one-fp16-cuda --output generated/soma/walk-one-mixed-pose.gguf
build/sonic-cpu/tests/motionbricks-soma-mixed-pose-probe generated/soma/fp32 \
  generated/soma/walk-one-mixed-pose.gguf generated/soma/walk-one-isolated-mixed-pose-cpu.json cpu
```

For batch-two diagnostics, export a second independently captured fixture with
the same duration and append its GGUF path to that probe command. The measured
second fixture is `generated/soma/walk-four16-mixed-pose.gguf`, captured from
`walk-four16-input.npz` with the official CUDA graph and `--tap-layers`.

For isolated backend arithmetic investigations, `MOTIONBRICKS_SOMA_NATIVE_F16=1`
uses F16 matrix operands with requested FP32 accumulation, including lowered
convolutions. On Vulkan this also permits F16/cooperative-matrix capabilities;
run it in a fresh process without `GGML_VK_DISABLE_*` overrides because backend
capabilities are fixed at initialization. The default remains unchanged.
Runner and chain reports record matrix operand/accumulation types. On the
16-token walking fixture, this experiment uses `NV_coopmat2` on RTX 5070 Ti but
still has one pose-token mismatch and 5.47° maximum raw rotation error; CPU
also has one mismatch and 5.52°. It has not resolved parity and is not promoted
to the default path. Ten default CPU SOMA/G1 regression tests pass.

Retesting native F16 operands after the convolution correction, with exact raw
upstream features for the first turning request, still yields two pose-token
mismatches on both backends. CPU raw rotation error decreases from 13.08° to
5.24°; Vulkan changes from 13.14° to 12.55°. This does not establish parity or
justify changing the default arithmetic. The artifacts are
`zombie-conv-fix-{rounded-f32,f16-operands}-{cpu,vulkan}.{gguf,json}`.

CUDA captures now record the resolved provider options and explicitly select
`ORT_CUDA_GEMM_OPTIONS=0` by default. An additional diagnostic capture using
`--cuda-gemm-options 2` disables reduced-precision reductions as defined by
[ONNX Runtime's CUDA implementation](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/cuda/cuda_common.cc).
On the same walking input this changes 1,426 captured tensors, but leaves two
CPU pose-token mismatches and a 4.88° maximum raw rotation difference. It does
not resolve native parity; the default-policy capture remains the primary target.
Comparing those two upstream CUDA captures directly changes two of 80 pose-token
choices, with matching durations and maxima of 26.25 mm raw joint-position error
and 3.481° raw rotation error. `compare_soma.py` now checks both archive hashes
and records these discrete differences. This is evidence of reduction-policy
sensitivity, not an acceptance tolerance or a native parity pass.

`audit_soma_conv.py` compares fused bias with an intervening half storage boundary
using each convolution's exact captured operands. Across all 42 convolutions,
the latter reduces isolated output mismatches from 161,084 to 1,552 on walking
and from 160,968 to 1,494 on the first turning request. Native root and pose
decoder convolutions now preserve that boundary. Remaining differences include
matrix reduction arithmetic; this correction does not establish model parity.
The shared `precision_convolution_bias` helper has a regression using exactly
representable half operands: rounding the product before bias must produce 1,
while fusing bias produces `1 + 2^-10`. It passes on CPU and Vulkan with both
rounded-F32 and native-F16 operands. CTest registers this and the existing pose
position/binary-selection checks as `motionbricks-soma-precision-contract` when
the mixed-pose fixture exists. This test guards arithmetic rules, not full-model
numerical parity.
The [ONNX Runtime CUDA convolution implementation](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/cuda/nn/conv.cc)
supports separate bias addition when bias fusion is not used.

```sh
OPENBLAS_NUM_THREADS=1 .cache/soma-reference-venv/bin/python reference/audit_soma_conv.py \
  --model generated/soma/reference/motionbricks/assets/onnx/soma/mmm_v6_soma_fp16.onnx \
  --capture generated/soma/walk-one-fp16-cuda \
  --output generated/soma/walk-one-conv-bias-audit.json
```

`audit_soma_elementwise.py` checks isolated Add, Sub, Mul, Div, Sqrt, Relu,
LeakyRelu and Where operations using each operation's captured operands. Across
all 494 half-output operations (6,191,692 values), the explicit half-rounding
rules agree exactly with both default and strict-reduction CUDA captures on the
walking fixture. This covers the graph's neural and surrounding half operations,
and checks the CUDA half-rounded LeakyRelu coefficient. It rules out these local
arithmetic rules on this fixture; it does not establish native model parity or
cover matrix products, convolutions, normalization or softmax.

```sh
.cache/soma-reference-venv/bin/python reference/audit_soma_elementwise.py \
  --model generated/soma/reference/motionbricks/assets/onnx/soma/mmm_v6_soma_fp16.onnx \
  --capture generated/soma/walk-one-fp16-cuda \
  --output generated/soma/walk-one-elementwise-audit.json
```

`export_soma_norm.py` exports all 48 normalization calls with exact CUDA inputs,
scale, bias and epsilon. `motionbricks-soma-norm-probe` evaluates the native GGML
FP32 normalization expression on those inputs, excluding propagated errors.
Across 630,784 values, CPU differs at 72 values after half rounding (13 in the
root network), and Vulkan at 67 (8 in the root network). Maximum FP32 differences
are 3.815e-6 and 9.537e-6 respectively. These are local arithmetic measurements,
not an attribution of final token changes or a full-model parity pass. The
fixture verifies the official model and capture hashes and records source node
names; the report preserves the capture hash and per-node errors.

```sh
.cache/soma-reference-venv/bin/python reference/export_soma_norm.py \
  --model generated/soma/reference/motionbricks/assets/onnx/soma/mmm_v6_soma_fp16.onnx \
  --capture generated/soma/walk-one-fp16-cuda \
  --output generated/soma/walk-one-mixed-norm.gguf
build/sonic-cpu/tests/motionbricks-soma-norm-probe generated/soma/fp32 \
  generated/soma/walk-one-mixed-norm.gguf generated/soma/walk-one-isolated-norm-cpu.json cpu
```

`audit_soma_matmul.py` isolates every FP16 MatMul using captured operands and
official initializer values, without propagating errors between operations.
The default CUDA capture differs from FP32-accumulated, half-rounded products
at 21,959 of 2,583,936 values across 144 operators; the strict-reduction capture
differs at 10,721 values. These diagnostics identify reduction sensitivity but
do not justify accepting native pose-token mismatches. Reproduce with:

```sh
OPENBLAS_NUM_THREADS=1 .cache/soma-reference-venv/bin/python reference/audit_soma_matmul.py \
  --model generated/soma/reference/motionbricks/assets/onnx/soma/mmm_v6_soma_fp16.onnx \
  --capture generated/soma/walk-one-fp16-cuda \
  --output generated/soma/walk-one-matmul-default-audit.json
```

The optional `--node-prefix /mmm_net/_proj --rounding-blocks 8 16 32` probes
partial sums rounded toward zero as an arithmetic hypothesis, not a CUDA
implementation. Across 78,080 projection outputs, 16-term blocks reduce mismatch
counts from 217 to 59 on the one-target fixture and 200 to 49 on the four-target
fixture. They do not reproduce CUDA exactly, and the four-target maximum error
increases from 0.001953125 to 0.00390625. No such heuristic is applied to native
inference. NVIDIA's [PTX specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma)
leaves tensor-core accumulation order and rounding unspecified; FP32 accumulation
alone therefore does not identify a portable, bit-identical CUDA reduction rule.

Mixed-FP16 chains now use the official CUDA graph for every reference transition,
with all graph taps and upstream Warp playback. CPU and Vulkan replay and
independent-feedback errors can be viewed in the demo; see `docs/DEMO.md`.
After the convolution bias correction, the three-transition walking chain has
matching durations and closed-loop pose tokens, but reaches 4.08 mm / 0.854° on
CPU and 3.46 mm / 0.533° on Vulkan. Replay still has one token mismatch on each
backend. These remain experimental failures of numerical parity.

Comparison reports include valid-frame feature errors, root/world-joint position
errors in meters, joint rotation errors in degrees, foot contacts and length
mismatches. These are measurements, not acceptance gates. Synthetic smoke inputs
do not establish real-motion or chained-playback parity.

The initial native SOMA feature bundle and boundary check can be reproduced with:

```sh
.cache/soma-reference-venv/bin/python scripts/convert_soma_to_gguf.py \
  --reference generated/soma/reference --output generated/soma/fp32
.cache/soma-reference-venv/bin/python reference/export_soma_fixtures.py \
  --capture generated/soma/pytorch-capture --output generated/soma/fixtures
nix develop --command cmake --build build/sonic-cpu \
  --target motionbricks-soma-pose-parity-test
build/sonic-cpu/tests/motionbricks-soma-pose-parity-test \
  generated/soma/fp32 generated/soma/fixtures/pose.gguf cpu generated/soma/native-taps
```

Despite its initial name, this test checks root, pose, decoder, and complete raw
feature inference, including duration selection, T-pose conversion, codebook lookup,
normalization and root selection. The optional final argument writes 1,371 native
graph snapshots and `native-boundaries.gguf`. Semantic comparison is available for the full-length fixture described below. Two-request batch checks now compare distinct one-target and four-target requests
with a shared 16-token duration, in both orders. All duration, root, pose and
decoder boundaries pass on CPU and Vulkan, with zero pose-token mismatches.
Maximum pose-logit errors are `1.22e-4` on CPU and `2.29e-4` on Vulkan.

The Vulkan investigation found that GGML packs noncontiguous attention matrix
operands through FP16 even when FP16 is disabled. Explicit contiguous F32 key/query
inputs fix the error without changing GGML or relaxing the provisional guard.

Real BVH fixtures must be remapped to NVIDIA's USD skeleton: the source BVH has
an extra scene root, while the SOMA engine requires Hips at index zero. Earlier
`walk-reference` artifacts without this remapping are unsuitable for playback.
Use `prepare_soma_inputs.py` with the pinned clip and materialized, hash-verified
`motionbricks/assets/human/soma_base_skel_minimal_no_root.usd`. This requires
`warp-lang==1.17.0` and `usd-core==25.11` in the reference environment. The script
calls upstream sampling, remapping, FK, alignment, feature encoding and masks.

For the corrected walking fixture (context 0 seconds, target 1 second), CPU full
feature maximum absolute error is `8.34e-6`; all duration and pose-token choices
match. The selected duration is eight tokens (26 playback frames). Native playback
is separately compared against upstream Warp decoding, endpoint deltas, endpoint
application and FK, using identical upstream features. Maximum world joint position
component error is `2.68e-6` meters; maximum quaternion component error is `1.02e-6`
after sign alignment. This is one fixture, not a general parity guarantee.

```sh
.cache/soma-reference-venv/bin/python reference/capture_soma_playback.py \
  --upstream /path/to/pinned-motionbricks --poses generated/soma/walk-soma-reference \
  --capture generated/soma/walk-soma-trace --output generated/soma/walk-soma-io
build/sonic-cpu/tests/motionbricks-soma-playback-parity-test \
  generated/soma/walk-soma-io/playback.gguf
```

`capture_soma_playback.py --native native-boundaries.gguf` can apply upstream
playback to native features to isolate neural error. It does not establish native
playback parity; the dedicated C++ test above performs that comparison.

Native input encoding now covers FK, heading alignment, padded target poses,
world-space root derivatives, 6D joint rotations and masks. The walking fixture's
root features match upstream exactly; pose feature error is at most `6.41e-7`.
The playback fixture also carries these independent input boundaries and checks
one- and four-target-pose requests.

The native runner consumes poses and skeleton data from the fixture and performs
encoding, inference and playback entirely in C++. Its output contains feature and
playback boundary taps:

```sh
build/sonic-cpu/tests/motionbricks-soma-run generated/soma/fp32 \
  generated/soma/walk-soma-io/playback.gguf generated/soma/native-playback.gguf cpu
.cache/soma-reference-venv/bin/python reference/compare_soma_playback.py \
  --reference generated/soma/walk-soma-io --native generated/soma/native-playback.gguf \
  --output generated/soma/playback-errors.json
```

On this complete pose-to-pose path, CPU maximum world joint distance is `2.02e-6`
meters and angular error is `0.000123` degrees. A one-target-pose variant selects
16 tokens (58 frames), matching upstream, with maximum world joint distance
`2.25e-6` meters and angular error `0.000503` degrees. These single-transition measurements are supplemented by the chain checks below.

Root-selection instrumentation now copies the actual upstream decoder function's
local height-error tensors and `use_root_module` decision at return. It uses a
scoped Python profiler, restores any previous profiler, and leaves upstream
arithmetic unchanged; baseline-versus-instrumented outputs remain bit-exact.
Native boundaries include both mean target-height errors and the selected branch.
The sitting fixture retains the pose root (9.31 mm error, below the 50 mm switch
threshold). Synthetic target-height stress fixtures exercise both choices: shifting
the walking target root down 0.6 m produces 340.72 mm pose error versus 8.18 mm root
error and selects the root predictor; shifting it up 0.6 m retains the pose root.

These explicitly labelled stress fixtures use the last four local poses from
`walk-one-reference/poses.npz`, changing only `target[:, 0, 1]` by ±0.6 and saving
each four-pose array as NPY. `prepare_soma_inputs.py --target-local-poses FILE.npy`
encodes it through upstream Warp and records the override SHA256. They are not
unmodified motion clips. Both decisions, height-error estimates (1e-5 m ceiling),
pose tokens, and corrected playback pass on CPU and Vulkan. Worst stress-fixture
world position error is 6.87 µm; worst local rotation error is 0.00321°.

```sh
.cache/soma-reference-venv/bin/python reference/compare_soma_root_selection.py \
  --capture generated/soma/height-stress-low-trace \
  --native generated/soma/height-stress-low-vulkan.gguf \
  --output generated/soma/height-stress-low-vulkan-decision.json
```

Exported component fixtures retain these decision taps. When the generated
`height-stress-{low,high}-fixtures/pose.gguf` files exist, CTest registers both
branch regressions alongside the existing SOMA fixture tests.

A three-transition walking chain (target clip times 1, 1.5, and 2 seconds) now
compares identical-context replay and independent native feedback. Each transition
feeds its last four corrected poses into the next request. Shared endpoint frames
are emitted once, producing 60 frames. CPU closed-loop maxima are `2.24e-6` meters
world-joint distance and `0.000219` degrees rotation; RTX 5070 Ti Vulkan maxima are
`3.04e-6` meters and `0.000380` degrees. Both have zero duration and pose-token
mismatches in this chain. Reports retain per-transition and per-frame errors;
these measurements do not establish general tolerances.

Additional three-transition chains cover `sit_on_chair_start_R_001__A047.bvh`
(target times 2, 4, 6 seconds) and `zombie_walk_180_R_003__A330.bvh`
(3, 6, 9 seconds). All duration and pose-token choices match on CPU and Vulkan.
Closed-loop maxima against independent upstream FP32 captures are:

| Motion | CPU joint distance | Vulkan joint distance | CPU rotation | Vulkan rotation |
| --- | ---: | ---: | ---: | ---: |
| Sitting | 1.95 µm | 2.40 µm | 0.000200° | 0.000292° |
| Turning walk | 4.31 µm | 5.58 µm | 0.000567° | 0.000688° |

Reuse the same upstream capture for another backend without regenerating it:

```sh
.cache/soma-reference-venv/bin/python reference/run_soma_chain.py \
  --reference-chain generated/soma/chain-sit-cpu \
  --bundle generated/soma/fp32 \
  --runner build/vulkan-profile/tests/motionbricks-soma-run \
  --backend vulkan --output generated/soma/chain-sit-vulkan
```

New reports record runner, bundle-manifest, and reference-fixture SHA256 hashes,
and each native invocation keeps a `.log` with backend initialization details.
Reused captures check source, motion, skeleton, and any recorded fixture hashes.

`check_soma_chain.py` recomputes an acceptance result from the native GGUF outputs
and independent reference captures. It verifies capture checksums, pinned source
and precision, actual duration metadata, exact pose-token decisions, each side's
last-four-pose feedback, unchanged targets/skeletons, and exact correspondence of
the demo arrays to the evaluated steps. It records artifact hashes with its result.
The versioned `soma-chain-tolerances.json` requires at most **0.1 mm** local/world
joint distance, **0.01°** local/world joint rotation, and **1e-4** contact error.
Measured FP32 maxima across all replay and independent-feedback comparisons are
0.00554 mm, 0.001097°, and 6.32e-5 respectively. These limits cover the captured
regression cases; they do not assert generalization to arbitrary requests.

Acceptance reports also verify duration argmax selection against the allowed
mask and record the raw logits, runner-up and winner margin for each side.
The out-of-reach class is excluded, and ties select the first allowed index as
upstream does. The runner records its actual allowed-duration bit mask; older
pose-to-pose outputs used all eleven choices. On the final Vulkan sitting
closed-loop request after the convolution correction, upstream chooses 11 tokens
with a 0.001953125 margin over 10. Native logits for both choices are exactly -3,
so it selects 10. The masks agree and both selections follow argmax; this is
a numerical decision change, not a mask or indexing error. The diagnostic is in
`chain-sit-half-vulkan-conv-fix/acceptance-duration.json`.

Six fresh chains (`chain-{walk,sit,zombie}-{cpu,vulkan}-blend-validated`),
regenerated after the masked-selection correction, pass all
36 transition/mode comparisons. Both experimental mixed-FP16 walking chains
fail these same checks, including a discrete mismatch during independent feedback.

Before the convolution bias correction, mixed-FP16 coverage included CUDA sitting and turning captures,
reused by both native backends (`chain-{sit,zombie}-half-{cpu,vulkan}-coverage`).
All durations match, but all four chains fail the unchanged acceptance policy.
Each folder includes its failure report and a demo overlay. Independent-feedback
maxima across the three transitions are:

| Motion | Backend | Joint position | Joint rotation | Pose-token mismatches |
| --- | --- | ---: | ---: | ---: |
| Sitting | CPU | 20.45 mm | 3.2374° | 1 |
| Sitting | Vulkan | 1.14 mm | 0.1840° | 0 |
| Turning | CPU | 62.52 mm | 14.9068° | 24 |
| Turning | Vulkan | 81.31 mm | 13.9710° | 21 |

Same-context turning replay still differs at 19 CPU and 14 Vulkan pose tokens,
so feedback drift alone does not explain the discrepancy. Isolating its first
pose call with exact upstream inputs restores all 80 token choices on both
backends (maximum logit error 0.0625, including batch size one). This extends the
root-propagation finding beyond walking; it does not establish turning parity.

After the convolution correction, `chain-{walk,sit,zombie}-half-{cpu,vulkan}-conv-fix`
reuses the same independent captures. All six still fail the acceptance gate.
Closed-loop measurements are:

| Motion | Backend | Joint position | Joint rotation | Pose-token mismatches | Duration |
| --- | --- | ---: | ---: | ---: | --- |
| Walking | CPU | 4.08 mm | 0.8539° | 0 | Exact |
| Walking | Vulkan | 3.46 mm | 0.5332° | 0 | Exact |
| Sitting | CPU | 19.93 mm | 3.2413° | 1 | Exact |
| Sitting | Vulkan | Not aligned | Not aligned | Count differs | Final transition 34 vs 38 frames |
| Turning | CPU | 61.77 mm | 13.6918° | 27 | Exact |
| Turning | Vulkan | 57.57 mm | 7.9537° | 14 | Exact |

Walking replay still has one token mismatch on each backend. Sitting replay has
none, but its Vulkan closed-loop duration diverges. Its demo preserves both full
motions and limits error measurements to the matching-duration prefix; the
remainder is visibly unaligned, with no error vectors or numeric frame errors.
All six folders include overlays and failed acceptance reports. Ten FP32 SOMA/G1 regressions and Vulkan batch-two
parity pass after the correction.

The runner's optional final argument `reference-features` bypasses native feature
encoding using the seven captured input arrays in the playback fixture. It still
runs native T-pose conversion, normalization, networks and playback. Outputs are
explicitly marked `input.feature_source=captured-reference`; the chain validator
rejects this diagnostic mode as evidence of native pose-to-pose parity.
`compare_soma_mixed.py` reports the feature source and all three normalized root
input boundaries after half rounding, and checks the full reference pin.

In the pre-convolution-correction diagnostic on the first turning request,
native pose encoding differs by at most 7.16e-7
in raw pose features, changing six normalized half values. Captured features
make all root-input half values match exactly, but CPU still has four pose-token
mismatches and Vulkan two (versus five and three with native encoding). Thus
encoding contributes, but neural arithmetic remains a separate unresolved gap.

```sh
build/sonic-cpu/tests/motionbricks-soma-run generated/soma/fp32 \
  generated/soma/chain-zombie-half-cpu-coverage/step-000/playback/playback.gguf \
  generated/soma/zombie-captured-features-cpu.gguf cpu mixed-fp16 reference-features
```

```sh
.cache/soma-reference-venv/bin/python reference/check_soma_chain.py \
  --chain generated/soma/chain-walk-cpu-blend-validated \
  --output generated/soma/chain-walk-cpu-blend-validated/acceptance.json
.cache/soma-reference-venv/bin/python reference/export_soma_demo.py \
  --chain generated/soma/chain-walk-cpu-blend-validated \
  --acceptance generated/soma/chain-walk-cpu-blend-validated/acceptance.json \
  --output generated/soma/chain-walk-cpu-blend-validated/demo.json
```

The optional acceptance report is bound to the exact chain/report hashes. The
demo then displays **Chain checks passed/failed**, scoped to that capture. This
does not imply completion of the overall SOMA parity objective. Old runner outputs
without precision metadata must be regenerated before this gate accepts them.

`run_soma_chain.py` captures every upstream model call plus essential runtime
boundaries. `export_soma_demo.py` produces a report accepted by the existing demo's
`-comparison` option, with replay/closed-loop selection, overlaid skeletons and
position/rotation errors. Both SOMA and existing G1 headless viewer checks pass.
See `docs/DEMO.md` for commands. The demo labels results as measured pending full
acceptance, rather than claiming the entire parity objective is complete.

The full-length (16-token) one-target walking fixture now maps all 403 captured
root, pose and decoder module outputs to native taps, including dropout/identity
and composite-layer aliases. Aliases are checked for exact equality within the
independent upstream capture before they count toward coverage. Native graph
snapshots also include 24 functional QKV projections without corresponding
PyTorch module-forward calls; these remain listed separately. This coverage does
not yet prove operation-by-operation mixed-FP16 parity.
The eight-token fixture also maps all 403 module outputs on CPU and Vulkan,
passing the provisional `0.001` absolute diagnostic ceiling. Upstream retains
padded sequences; native execution shortens them. The comparator has explicit
temporal-axis rules for the root transformer and convolutional decoders, verifies
the captured upstream masks, and compares only the valid prefix for these layers.
Reports record the padded and valid lengths and mask provenance. Decoder wrapper
aliases additionally verify the exact final masking operation across the entire
upstream tensor. Unknown shapes, changed masks, and wrong lengths remain failures.
Maximum layer errors on this fixture are `6.11e-5` on CPU and `1.23e-4` on Vulkan.

```sh
.cache/soma-reference-venv/bin/python reference/compare_soma_layers.py \
  --capture generated/soma/walk-one-trace \
  --native generated/soma/named-one-cpu/native-neural-taps.gguf \
  --output generated/soma/named-one-cpu/layers.json \
  --require-complete --max-absolute 0.001
```

The comparator verifies the reference archive hash and native trace's originating
capture hash, checks shapes/layouts, records max/RMS errors, and can reject missing
module coverage or a caller-specified error ceiling. The `0.001` example is a
provisional diagnostic ceiling, not a broadly established acceptance tolerance.

The bundle contains 189,141,582 FP32 network values and excludes the VQ encoder.
The dedicated opaque-handle SOMA API now loads it and generates corrected poses;
see `docs/API-SOMA.md`. The shared-library parity test passes with maximum array
error `1.63e-5` on the one-target walking fixture. Mixed-FP16 execution, broader fixtures, full layer correspondence, full-request batching and broader public API coverage remain in progress. Existing G1 regression
checks have passed.

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

### Context-blend parity

Upstream retains the unfiltered qpos whenever its default `FILTER_QPOS` path
runs. Capture that observational buffer and build a compact blend-only fixture:

```sh
docker run --rm --device=nvidia.com/gpu=all \
  --user "$(id -u):$(id -g)" --entrypoint python \
  -e PYTHONPATH=/work/reference \
  -v "$PWD:/work" -v /path/to/GR00T-WholeBodyControl:/upstream:ro \
  motionbricks-reference-session:torch2.7 \
  /work/reference/capture_session.py \
  --upstream-root /upstream --output /work/generated/session-blend \
  --seed 1234 --artifact-limit-mb 40 --trace-targets --trace-blend

docker run --rm --user "$(id -u):$(id -g)" --entrypoint python \
  -e PYTHONPATH=/work/reference \
  -v "$PWD:/work" -v /path/to/GR00T-WholeBodyControl:/upstream:ro \
  motionbricks-reference-session:torch2.7 \
  /work/reference/build_blend_parity.py \
  --capture /work/generated/session-blend --upstream-root /upstream \
  --support /work/generated/safe/support.safetensors \
  --output /work/generated/context-blend.mbblend

./build/debug/tests/motionbricks-agent-blend-test generated/context-blend.mbblend
```

The builder first proves the recorded qpos exactly equals upstream's formula.
The native test then applies the C++ filter to the same 14 unfiltered buffers.
The measured root maximum is exactly zero; the worst physical-joint quaternion
component error is 4.33e-7 (0.000050 degrees). Root orientation and the four
virtual hand/toe endpoints are excluded because upstream's qpos filter cannot
modify them. All generated observation artifacts remain gitignored.

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

### Reproducible mixed-FP16 layer inventory

`compare_soma_mixed_layers.py` compares the runner's named neural trace against
verified official CUDA ONNX captures. It records both artifact hashes, precision,
reference runtime identity when available, every native tap, explicit layout
transforms, numerical errors, and unmapped reference graph outputs. It is a
**diagnostic**, not an acceptance gate: input identity is not established by
this layer-only report. Use the whole-model input comparison and chain gate
alongside it. Shorter requests use explicit temporal axes and validate the
captured root, pose, or decoder token mask before removing padded frames.
Different native/reference durations remain shape failures rather than being
cropped to a common length.

```sh
.cache/soma-reference-venv/bin/python reference/compare_soma_mixed_layers.py \
  --reference generated/soma/chain-zombie-half-cpu-coverage/step-000/capture \
  --native-trace generated/soma/zombie-conv-fix-rounded-f32-cpu.gguf.layers.gguf \
  --output generated/soma/zombie-conv-fix-rounded-f32-cpu.layers.json
```

The four post-convolution-fix turning-motion diagnostics (CPU/Vulkan with
rounded-F32/native-F16 operands) each compare 400 tap executions against 327
unique reference outputs. The root shared transformer executes twice, for
initial duration and full root prediction; those executions are separately
labelled. All 32 identity-normalization outputs are resolved through the
following activation's actual input edge in the captured source graph. The
duration embedding is compared with its ONNX Gather output. The remaining
four taps for rounded-F32 or two for native-F16 are weight conversions,
explicitly unmapped; the report does not claim coverage of
every ONNX operation. All four first differ at the root local-pose projection
(maximum absolute error 0.0001220703125). Downstream decoder errors include the
effect of different selected pose tokens and are not isolated decoder errors.
The unchanged chain acceptance gate still fails for mixed FP16.

The 11-token sitting segments also compare all 400 layer tap executions, with
284 validated temporal prefixes each, on CPU and Vulkan replay and CPU closed
loop. Vulkan closed-loop step 001 has the same coverage. At step 002, Vulkan's
10-token prediction disagrees with the reference's 11 tokens: the report leaves
284 shape mismatches explicit and compares only the 116 compatible taps. Layer
reports are saved as `step-*/{replay,closed}.layers.json` under the corresponding
`chain-sit-half-{cpu,vulkan}-conv-fix` directories. These measurements include
propagated input differences; they do not establish isolated layer accuracy.

### Isolated native half matrix products

`export_soma_matmul.py` exports one named projection MatMul's exact half inputs,
weights, and output from the checksum-verified official ONNX CUDA capture.
`motionbricks-soma-matmul-probe` evaluates the production `precision_matmul`
expression on CPU or Vulkan, with rounded-F32 or opt-in native-F16 operands.
Unlike the NumPy arithmetic audit, this measures the actual native backend.
It deliberately reports differences without claiming full-model parity.

```sh
.cache/soma-reference-venv/bin/python reference/export_soma_matmul.py \
  --model generated/soma/reference/motionbricks/assets/onnx/soma/mmm_v6_soma_fp16.onnx \
  --capture generated/soma/chain-zombie-half-cpu-coverage/step-000/capture \
  --output generated/soma/zombie-root-projection.gguf
nix develop --command cmake --build build/vulkan-profile \
  --target motionbricks-soma-matmul-probe
nix develop --command env MOTIONBRICKS_SOMA_NATIVE_F16=0 \
  build/vulkan-profile/tests/motionbricks-soma-matmul-probe \
  generated/soma/fp32 generated/soma/zombie-root-projection.gguf \
  generated/soma/zombie-root-projection-vulkan-0.json vulkan
```

The first turning-motion root local-pose projection has 3/2,048 unequal output
values on CPU and 4/2,048 on Vulkan, maximum absolute error 0.0001220703125,
in both operand modes. The fixture gives native arithmetic exactly the CUDA
operands, so these differences originate in the matrix operation itself, before
bias addition or propagated feature errors. A nonsquare 2-by-3 times 3-by-4
integer-valued control product matched exactly on both backends and both modes,
checking the export/probe matrix layout. The official results are recorded in
`zombie-root-projection-{cpu,vulkan}-{0,1}.json` under `generated/soma`.

The fixture also stores an independent FP64 matrix product rounded to half as
`fp64_rounded_diagnostic`. On this projection, both CPU operand modes match all
2,048 values of that diagnostic; both Vulkan modes differ in one value; the
official CUDA output differs in three values. The three CUDA differences lie
near half-rounding midpoints (the FP64 sums are approximately 8.75e-8, 4.08e-7,
and 8.80e-8 away). Consequently, simply increasing accumulation precision would
not reproduce the target for this case. These observations identify a reduction
rounding difference, not permission to replace the CUDA target or relax the
chain acceptance criteria. The report keeps this diagnostic separate from the
native-versus-CUDA error fields.

### Complete-feature SOMA batching

The internal `run_soma_features_batch` entry point processes up to 64 raw feature
requests. It normalizes and retargets every request, batches duration prediction
by precision, groups subsequent neural work by predicted duration, and propagates
native root output through pose selection, codebook lookup, decoding and root
selection. Results and optional processing taps retain input order. Invalid
requests publish no partial outputs. Shared single-request processing uses the
same preparation and postprocessing stages; its neural execution path is retained.

`motionbricks-soma-feature-batch-parity-test BUNDLE cpu|vulkan FIXTURE...`
checks three independently captured FP32 requests (16, 16, and 8 tokens), both
input orders, exact upstream duration and pose-token choices, native batch versus
single results, and upstream full outputs. CPU batch/single output is identical;
Vulkan maximum raw-feature difference is 1.47037e-5. Maximum raw-feature error
against upstream is 1.75238e-5 on the generic CPU backend, 1.19507e-5 on the
Zen 4 CPU backend, and 3.36468e-5 on Vulkan. These arrays mix
positions, rotations and contacts; these figures are **not** world-position errors.
The CTest case is `motionbricks-soma-feature-batch-parity`; an optional Vulkan
case is enabled by `MOTIONBRICKS_ENABLE_VULKAN_PARITY_TESTS`.

This establishes complete neural-feature batching for these FP32 fixtures, not
a public batch API. Mixed-FP16 batches still need independent upstream
end-to-end validation; the existing
mixed-FP16 component diagnostics are not promoted to that scope.
Precision isolation is tested separately: two half requests and one FP32 request
produce exactly the same outputs when their precision groups run separately or
together, on CPU and Vulkan. A subsequent FP32 single request remains bitwise
unchanged. This verifies arithmetic-mode isolation, not CUDA FP16 parity.
After the shared-stage refactor, all 28 tensors of the captured-feature turning
mixed-FP16 single-request output remain exactly equal to the pre-refactor result
(`zombie-batch-refactor-single.comparison.json`). This preserves the measured
experimental behavior; it does not turn the failing FP16 parity result into a pass.

The internal `run_soma_motion_batch` adds native skeleton encoding, alignment,
feature inference, playback decoding, endpoint correction and world FK. Requests
borrow their skeleton, joint map, four context poses and one to four target poses
for the duration of the call. Results retain encoded inputs, alignment, input
world transforms, neural predictions and corrected playback; processing boundaries
are optional. Failures leave the result vector empty.

`motionbricks-soma-motion-batch-parity-test BUNDLE cpu|vulkan FIXTURE...` checks
the one-target walk, shorter walk and low-height stress fixtures together, in
both orders. These include 16- and 8-token durations and root-selection coverage.
CPU batch/single motion is identical; worst independent-reference error is
4.43596e-6 m, 0.00108271 degrees and 1.40071e-5 contacts. Vulkan worst reference
error is 9.34772e-6 m, 0.00264085 degrees and 8.66652e-5 contacts; batch/single
differences reach 6.48986e-6 m and 0.00164581 degrees. All remain within the
unchanged chain tolerances. CTest registers `motionbricks-soma-motion-batch-parity`
and the opt-in Vulkan counterpart. This is FP32 request batching; mixed-FP16 and
independent batched chains remain separate validation scopes.

### Complete mixed-FP16 feature batch diagnostic

`export_soma_mixed_pose.py` now includes the seven raw model inputs and full CUDA
output as well as the six isolated pose boundaries. Those original six boundaries
remain unchanged. `motionbricks-soma-mixed-feature-batch-probe` starts from raw
features and propagates native predictions through every neural stage. It does
not substitute reference intermediate activations. It reports both request
orders, upstream duration and pose choices, raw-feature differences, and native
batch-versus-single differences. Its successful exit means the diagnostic ran;
its JSON status explicitly says parity is not established.

```sh
.cache/soma-reference-venv/bin/python reference/export_soma_mixed_pose.py \
  --capture generated/soma/walk-one-fp16-cuda \
  --output generated/soma/walk-one-mixed-full.gguf
.cache/soma-reference-venv/bin/python reference/export_soma_mixed_pose.py \
  --capture generated/soma/walk-four16-fp16-cuda \
  --output generated/soma/walk-four16-mixed-full.gguf
nix develop --command cmake --build build/vulkan-profile \
  --target motionbricks-soma-mixed-feature-batch-probe
nix develop --command build/vulkan-profile/tests/motionbricks-soma-mixed-feature-batch-probe \
  generated/soma/fp32 vulkan generated/soma/walk-mixed-full-batch-vulkan.json \
  generated/soma/walk-one-mixed-full.gguf generated/soma/walk-four16-mixed-full.gguf
```

With rounded-F32 operands, both requests retain the reference's 16-token duration.
The one-target/four-target requests differ in 2/5 of 80 pose choices on generic
CPU, and 4/2 on Vulkan. CPU batch output equals single output. Vulkan batch output
differs from native single inference by 3/5 pose choices and raw-feature maxima
0.111966/0.357193. Reversing input order preserves these per-request results.
Reports are `walk-mixed-full-batch-{cpu,vulkan}.json`. Their raw-feature maxima mix
feature units and must not be presented as world-position error. The references
are independent single-request CUDA captures; this does not characterize CUDA's
own batched execution. Skeleton encoding, endpoint playback and chained feedback
are outside this particular diagnostic. Complete FP16 batch parity **fails**;
precision-group isolation passing does not change that conclusion.

The mixed-feature probe's optional processing comparisons are now included in
`batch_single_boundaries` for every request. On the two walking requests,
`walk-mixed-full-batch-vulkan-boundaries.json` shows identical normalized inputs,
but duration-logit differences up to 0.000976562/0.00195312 and predicted-root
feature differences up to 0.000976562. Root differentiation increases differences
before pose prediction, whose logit maxima differ by 1.125/0.398438. These are
propagated comparisons, not isolated proofs that every downstream layer is wrong.

An isolated width test confirms one cause of shape-dependent Vulkan arithmetic.
Export the first root projection with `export_soma_matmul.py --repeat-rows 2`
to duplicate its eight captured rows and expected outputs. This repeats reference
rows; it does **not** recapture CUDA at the wider shape. The GGML Vulkan backend
uses its matrix-vector path through eight columns and its matrix-matrix path at
sixteen columns. On the turning projection, CPU preserves every repeated value;
Vulkan changes two values in the original 2,048-value block (flattened indices
194 and 1112). The original Vulkan projection differs from CUDA in four values;
the wider projection differs in two per repeated block. Reports are
`zombie-root-projection[-repeated]-{cpu,vulkan}-width.json` and
`zombie-root-projection-width-comparison.json`. This establishes a kernel-width
rounding effect, not a completed fix or a change to the acceptance target.

A zero-padding experiment forced narrow half matrix products to at least sixteen
rows, avoiding that matrix-vector/matrix-matrix switch. On a four-request group
(one-target walk, four-target walk, turning, and 11-token sitting), this made
native batch/single final features identical and removed all batch/single token
differences. However, single-inference CUDA pose mismatches changed from
`[1,3,2,0]` to `[4,0,2,0]`: the one-target walking result worsened. Batch CUDA
mismatches changed from `[4,1,2,0]` to `[4,0,2,0]`. This is not a general parity
fix, so the runtime experiment was **reverted**, including its extra pose-graph
capacity. The probe now records `single_reference_pose_mismatches` separately
from batch errors to expose precisely this tradeoff.

The retained evidence is `coverage-mixed-full-batch-vulkan-{original,padded}.json`,
`narrow-matmul-padding-experiment.patch`, and its JSON manifest with the trial
binary, patch and report hashes. The patch includes a 4096-node single pose graph:
the added padding nodes exceeded the original 2048-node capacity for a short
request during the first trial. The manifest records the trial-only environment
setting; it is not a supported runtime option after the revert. No acceptance
limit or reference result was changed.

### Root batch graph capacity

The advertised 64-request root limit previously exceeded fixed GGML graph-node
capacities and aborted even in FP32, before inference (`root-capacity-before.log`).
Both batched root builders now size their graphs from the tensors actually
constructed, with reserve for external model-weight leaves. This changes graph
metadata allocation, not neural arithmetic. The supported models use fewer than
1024 external root-weight leaves.

`motionbricks-soma-root-capacity-test BUNDLE FIXTURE BATCH fp32|mixed-fp16` exercises
duration prediction and full root prediction at a six-token duration. Both
completed with finite, correctly sized outputs for 64 requests on CPU in FP32 and
mixed FP16. The serial CTest cases `motionbricks-soma-root-capacity-fp32` and
`motionbricks-soma-root-capacity-mixed-fp16` carry the `capacity` label and are
registered when the independent fixture and bundle exist. This is a capacity
regression, not evidence of 64-request CUDA numerical parity or Vulkan memory
availability. Ordinary G1 root and complete SOMA feature-batch parity remain
separate checks.

### Per-product alignment arithmetic diagnostic

`audit_soma_matmul.py --alignment-blocks 8 16` now tests a second reduction
hypothesis: retain the sum of operand exponents, truncate each product to a
shared alignment quantum, add the aligned products and previous accumulator,
then truncate the block result to FP32. It probes 24 and 25 fractional alignment
bits. The motivation is the independently published
[Tensor Core models, version 3](https://arxiv.org/html/2512.07004v3), which describe
unnormalized products and alignment truncation. This finite-half implementation
uses contiguous blocks; it does not claim to emulate every cuBLAS kernel or the
RTX 5070 Ti hardware. The paper's B200 results alone do not establish that claim.

For the walking capture's `/mmm_net/_transformer` prefix, 16-term blocks and
25 fractional bits reduce mismatched half outputs from 9,680 (NumPy FP32
accumulation) to 12 out of 1,900,544. The remaining differences are in six
attention-score products. This is strong evidence for the alignment hypothesis
on those operations, but not whole-model acceptance. Broader audits show:

| Capture | Values | FP32 accumulation mismatches | Aligned 16/25 mismatches |
| --- | ---: | ---: | ---: |
| Turning, official default CUDA policy | 2,583,936 | 21,327 | 11,256 |
| Four-target walking, official default CUDA policy | 2,583,936 | 21,886 | 11,632 |
| One-target walking, strict CUDA diagnostic policy | 2,583,936 | 10,716 | 714 |

Most remaining default-policy discrepancies occur in shared-root `linear2`
products and the first root external-conditioning projection. Strict policy
reduces but does not remove discrepancies; it remains a diagnostic, not a
replacement acceptance target. Different kernel reduction orders and reduced
precision intermediate accumulation remain to be investigated. No production
arithmetic or acceptance tolerance changed in this experiment.

Reports: `walk-one-alignment-audit.json`, `zombie-all-alignment-audit.json`,
`walk-four16-all-alignment-audit.json`, and
`walk-one-strict-all-alignment-audit.json` under `generated/soma`. A cancellation
regression verifies that alignment truncates individual products before summing,
which cannot be modeled by rounding only the final exact dot product. All 16
Python reference-tool tests pass.

After rebuilding all CPU targets, 40 of 41 CTest cases passed on the first run,
including both capacity checks. The Go/browser case timed out waiting for a jump
plan; it passed on an isolated rerun (16.35 seconds). This is recorded as a
transient browser failure, not a clean first-pass full-suite result. No browser
behavior or timeout was changed to make the rerun pass.

### Eight-way half partial reductions in root products

The next isolated audit identifies the large default-CUDA discrepancies more
precisely. `audit_soma_matmul.py --split-k-blocks 256` computes independent
256-term partial products using the aligned 16/25 hypothesis above, rounds each
partial to FP16, then sums the partials in FP32 before the final half conversion.
This matches all six shared-root `linear2` products exactly on one-target
walking, four-target walking, and turning captures (27,648 values per capture).
The first root external-conditioning projection has K=2560 instead of K=2048;
`--split-k-blocks 320` matches all 8,192 outputs exactly on each capture. Thus both
sets fit an eight-way split reduction, with half partials and a FP32 final sum.
There are zero differences across these 21 captured products / 107,520 values.

The evidence is `generated/soma/root-split-k-summary.json`, which records capture
hashes and links the complete source-node reports. These are isolated operations
with exact CUDA operands, not propagated native inference. Applying the same
split universally is incorrect: many other layers become less accurate. The
current runtime is unchanged; selective integration must preserve the captured
kernel's shape/reduction behavior and be validated on complete chained output.
This is an empirically supported arithmetic model for the pinned stack, not a
claim that every CUDA version or GPU uses the same splitting heuristic.

A regression checks the distinction between rounding independent partials and
rounding only the complete dot product. All 17 reference-tool tests pass. The
initial width sweep is retained in `zombie-root-reduced-precision-hypotheses.json`
as exploratory data; the pinned, checksum-verified CLI reports are the evidence
for the exact matches above.

### Native split-only integration trial

A selective native trial split the same seven root matrix products into eight
GGML products, rounded their outputs to half, and summed in FP32. It used the
ordinary backend product arithmetic; it did **not** emulate per-product alignment.
Both CPU and Vulkan completed the three-step turning replay and independent
closed-loop chain, but both failed strict acceptance:

| Backend/mode | Baseline pose mismatches | Split trial | Baseline max position | Split max position |
| --- | ---: | ---: | ---: | ---: |
| CPU replay | 16 | 14 | 73.358 mm | 52.577 mm |
| CPU closed loop | 27 | 21 | 61.770 mm | 68.463 mm |
| Vulkan replay | 13 | 10 | 69.220 mm | 68.762 mm |
| Vulkan closed loop | 14 | 19 | 57.574 mm | 67.371 mm |

CPU maximum rotation error also increased from 13.692 to 13.907 degrees;
Vulkan closed-loop rotation error increased from 7.954 to 10.616 degrees.
Thus splitting alone is insufficient. The trial runtime change was reverted;
no new split environment option is supported. The next integration needs the
alignment arithmetic as well as the split schedule, followed by the same full
chain checks. This does not invalidate the exact isolated Python results, which
include both operations and use identical captured inputs.

Evidence: `chain-zombie-half-{cpu,vulkan}-split-trial` contains reports, taps,
full motion arrays, and failed acceptance reports. `native-split-k-trial.patch`
and `native-split-k-trial.json` retain the rejected code and hashes of the patch,
trial binaries and chain reports. The trial-only environment variable is
recorded in that manifest. Baseline values are from the `conv-fix` chain reports.

### Native aligned-product implementation

`src/soma_aligned_dot.hpp` implements the aligned 16-term / 25-fractional-bit
arithmetic in C++, including unnormalized product exponents and FP32 truncation
at block boundaries. Its finite inputs must be exactly half-representable FP32
values. The isolated native probe now accepts an optional final argument:
`aligned16` or `aligned16-split8`. These modes explicitly run on CPU and reject
Vulkan; they do not invoke a backend matrix kernel or change model inference.
The split mode rounds each of eight independent partials to half and reduces them
in FP32, matching the Python hypothesis.

For example, after exporting an exact-operand fixture:

```sh
build/sonic-cpu/tests/motionbricks-soma-matmul-probe \
  generated/soma/fp32 generated/soma/native-aligned-probes/zombie-0.gguf \
  /tmp/zombie-aligned.json cpu aligned16-split8
```

The native implementation matches **all 107,520 CUDA outputs exactly** across the
same 21 shared-root / root-conditioning products on the three captured motions.
`generated/soma/native-aligned-probes/summary.json` records the capture, fixture,
and probe-binary hashes and links all 21 individual reports. The first turning
root projection using `aligned16` has four differences, matching the Python
alignment model's result; see `zombie-native-aligned-projection.json`. This is a
validated CPU arithmetic building block, not full-model or Vulkan parity. Runtime
integration, reduction-order differences in other products, and closed-loop
validation remain outstanding.

### CPU aligned-root inference trial

The runner now offers an opt-in diagnostic environment setting,
`MOTIONBRICKS_SOMA_ALIGNED_ROOT_TRIAL=1`, requiring `cpu mixed-fp16`.
It replaces only the six shared-root `linear2` products and first root external
conditioning product with the aligned split operation. Public APIs and default
inference do not enable it. Vulkan and FP32 runner requests reject this mode.
The output and layer GGUFs record `arithmetic.root_policy=aligned16-split8-trial`;
new chain and acceptance reports also expose the policy.

The operation is a CPU GGML custom node with the same validated dot arithmetic.
All 21 graph-level isolated products again match all 107,520 CUDA outputs exactly;
see `native-aligned-probes/graph-summary.json`. This validates tensor indexing and
execution through GGML, beyond the standalone scalar probe. The probe's final
argument `aligned16-split8-graph` selects that verification path.

Three-motion native-encoder chain results are retained under
`chain-{walk,sit,zombie}-half-cpu-aligned-root-trial`, with failed acceptance and
viewable `demo.json` exports. Compared with the baseline `conv-fix` runs:

| Motion | Replay pose mismatches, baseline → trial | Closed-loop pose mismatches, baseline → trial |
| --- | --- | --- |
| Walking | 1 → 0 | 0 → 1 |
| Sitting | 0 → 14, with a replay duration mismatch | 1 → 0 |
| Turning | 16 → 4 | 27 → 9 |

Turning replay maximum joint error decreases from 73.358 to 33.091 mm and
13.692 to 6.521 degrees; closed-loop error decreases from 61.770 to 51.759 mm
and 13.692 to 7.613 degrees. Walking replay improves to 3.163 mm / 0.316 degrees,
but closed-loop error increases to 24.341 mm / 2.842 degrees. Sitting closed-loop
error improves to 0.872 mm / 0.184 degrees; its replay duration mismatch precludes
a complete motion-error comparison. All still fail the unchanged strict limits.
`aligned-root-chain-summary.json` records these results and labels common-prefix
errors where durations differ. These mixed results make this a diagnostic for
extending the arithmetic implementation, not a promoted default parity fix.

After integration, rebuilt G1 root parity, default SOMA feature-batch parity and
all 17 reference-tool tests pass. The default arithmetic path remains selected
unless the runner explicitly enables the trial.

### Broader CPU alignment diagnostic

`MOTIONBRICKS_SOMA_ALIGNED_MATMUL_TRIAL=1` extends the CPU-only runner experiment
to all calls through `precision_product`, while also enabling the seven verified
root split products. It records `arithmetic.matmul_policy=aligned16-trial` in
output and layer GGUFs; new chain and acceptance reports expose this field.
Default inference is unaffected. This is a broader hypothesis, not a supported
CUDA-equivalent arithmetic policy.

The custom operation now supports either one reduction or eight half-rounded
partials. The fixture exporter and native graph probe also support rank-4
attention matrices. On the turning pose transformer layer 2 attention scores,
the 16-head native `aligned16-graph` output reproduces the Python hypothesis,
including precisely the same three CUDA mismatches out of 4,096 values. This
rules out a tensor-layout discrepancy for that captured attention operation.
Evidence is `zombie-attention-alignment.gguf` and its JSON diagnostic.

The first native-encoder turning transition completes with all 400 semantic taps
compared, but still has three pose-token mismatches and a 13.948-degree maximum
raw rotation error. Using captured features isolates neural arithmetic: the
normalized encoder inputs are no longer a variable, but four pose-token
mismatches and 14.044 degrees remain. These raw-feature errors are not corrected
playback errors. Reports and traces are named `zombie-aligned-all-first*` and
`zombie-aligned-all-reference-features*`. Both select the correct duration.
The captured-feature run explicitly does not validate the native encoder and
cannot satisfy the chain acceptance gate. Neither experiment is a promoted fix;
remaining per-operator reduction behavior still needs to be identified.

### Remaining narrow root-projection differences

With exact captured operands, all three end-target MLP matrix products match the
aligned 16/25 model exactly on turning (6,144 outputs); the corresponding native
full-run differences propagate from the earlier local-pose projection. See
`zombie-end-projection-alignment-audit.json`. This isolates the earlier
projection as the next arithmetic discrepancy, rather than justifying a change
to end-target bias handling.

A sweep of block permutations and independent FP32 partial accumulators across
one-target walking, four-target walking and turning found no order that matches
all three local-pose projection captures. No candidate was selected for runtime
use. Exploratory code, results, hashes and capture identities are retained in
`root-projection-order-experiment.py`, `root-projection-order-hypotheses.json`,
`root-projection-fp32-partials-hypotheses.json` and the experiment JSON manifest.
The alignment helper can now return its FP32 accumulator for such diagnostics,
without imposing a half rounding before testing a later reduction.

The repeated-row Vulkan probe with `MOTIONBRICKS_SOMA_NATIVE_F16=1` reports
`NV_coopmat2` matrix cores on the RTX 5070 Ti. Its 4,096 outputs contain eight CUDA
differences: four per repeated 2,048-value block, at indices 430, 554, 1112 and
1754 within each block. Those differences and their half values exactly match
the CPU aligned-product model. This supports that model for this Vulkan matrix
kernel, while confirming that it still differs from the pinned CUDA kernel.
Evidence: `zombie-root-repeated-native-f16-vulkan.json` compared with
`zombie-native-aligned-projection.json`. The wider input repeats captured rows;
it is not a fresh CUDA capture at that shape and does not prove general Vulkan
parity. All 17 reference-tool tests remain passing.

Mixed whole-output and layer comparison reports now expose both
`native_matmul_policy` and `native_root_policy`, so diagnostic arithmetic modes
remain explicit alongside operand precision.

### Native Vulkan split-reduction implementation

The isolated probe's `backend-split8` mode performs eight native GGML products,
rounds each partial to half, and sums them in FP32. With
`MOTIONBRICKS_SOMA_NATIVE_F16=1` on the RTX 5070 Ti's `NV_coopmat2` path, it matches
**all 107,520 CUDA outputs exactly** across the 21 previously verified root
products. This uses actual Vulkan FP16 matrix products, not the CPU custom
arithmetic callback. `native-aligned-probes/vulkan-split-summary.json` records
fixture/capture/probe hashes and links the individual reports and device logs.

The runner exposes this as a separate diagnostic with both
`MOTIONBRICKS_SOMA_NATIVE_F16=1` and
`MOTIONBRICKS_SOMA_VULKAN_SPLIT_ROOT_TRIAL=1`. It requires Vulkan mixed-FP16,
rejects CPU-alignment modes, and records
`arithmetic.root_policy=backend-f16-split8-trial`. It changes only the same seven
root products. It is not enabled by the public API or default runtime.

A matched three-step turning comparison uses native FP16 operands in both the
baseline and trial, avoiding a simultaneous operand-mode change:

| Mode | Baseline pose mismatches | Split trial | Baseline max joint error | Split max joint error |
| --- | ---: | ---: | --- | --- |
| Replay | 14 | 9 | 60.892 mm / 10.076° | 62.103 mm / 10.967° |
| Closed loop | 16 | 13 | 61.266 mm / 10.076° | 57.061 mm / 7.516° |

Durations match in both runs, but both fail unchanged strict acceptance. The
exact isolated products therefore establish a usable Vulkan arithmetic building
block, not whole-model parity. Reports, failed acceptance results and demo
exports are under `chain-zombie-half-vulkan-native-f16-{baseline,split}`; the
summary is `vulkan-native-f16-split-chain-summary.json`. All 17 reference-tool
tests pass after the change.

### Remainder-first alignment order

The remaining K=268 local-pose projection discrepancies are explained by placing
the short reduction block first: 12 products followed by sixteen 16-product
blocks, rather than sixteen full blocks followed by 12 products. This hypothesis
matches all 2,048 outputs on each of eight captures (one-target walking,
four-target walking, and three steps each of sitting and turning), for 16,384
exact half outputs. Later chain captures were checked after finding the rule on
the initial captures. `root-projection-residue-first-validation.json` records
capture hashes and before/after mismatch counts. Ordinary FP32 lane-reduction
hypotheses did not reproduce the target.

`audit_soma_matmul.py --alignment-blocks 16 --residue-first` makes this ordering
explicit. The native aligned-dot diagnostic now also processes the remainder
first, and records `arithmetic.aligned_block_order=residue-first-v1` when enabled.
The native graph probe matches the turning local-pose projection exactly;
`zombie-root-native-residue-first.json` supersedes the older four-mismatch
aligned-projection result for this updated arithmetic model. Default inference
and native Vulkan arithmetic have not changed to this order.

In the full captured-feature CPU trial, all root input projections, start/end
MLPs, first attention projection/output, and first-layer feed-forward FP16 taps
now match exactly. LayerNorm FP32 outputs still differ by up to 4.768e-7 in the
first layer. The first differing FP16 semantic tap is
`_shared_transformer_model.layers.1.linear1` (67 values, max 0.001953125), providing
a later, narrower point for the next investigation. Full inference still has
three pose-token mismatches and a 14.367-degree maximum raw rotation error;
correct duration and exact early layers do not establish motion parity.
Evidence: `zombie-aligned-residue-reference-features*` and
`zombie-residue-first-all-audit.json`.

The new `motionbricks-soma-aligned-dot` test verifies remainder placement using
cancellation and verifies FP32 truncation for both signs. It passes, along with
all 18 Python reference-tool tests. The CPU trial remains diagnostic; the full
precision and chain acceptance objective is unfinished.

### LayerNorm statistics and output arithmetic

The next discrepancy is isolated using the
[ONNX Runtime v1.32.0 CUDA implementation](https://raw.githubusercontent.com/microsoft/onnxruntime/v1.32.0/onnxruntime/core/providers/cuda/nn/layer_norm_impl.cu).
The reference uses four-value Welford accumulations per thread, warp/inter-warp
Chan reductions, CUDA `rsqrtf`, and a final fused multiply-add after multiplying
scale by inverse standard deviation. This differs from composing a separately
normalized tensor, scale multiplication and bias addition.

`capture_soma_norm_stats.py --model MODEL --capture CAPTURE --output DIR`
constructs isolated CUDA LayerNormalization nodes with optional mean and inverse
standard deviation outputs. It validates the official model/capture checksums,
requires every original output to remain bitwise equal to the full-graph
capture, and records runtime provenance. All 48 nodes / 630,784 output values
passed this check on turning; artifacts are `zombie-norm-stats` and the reusable
CLI validation run `zombie-norm-stats-cli`.

A vectorized arithmetic experiment reproduces every captured mean exactly.
Using captured statistics and the source's fused output expression reproduces
all 630,784 outputs exactly. Recomputing inverse standard deviation with a
correctly rounded reciprocal square root differs from CUDA in 193 of 720 rows,
leaving 18 half-rounded output differences. Calling CUDA `rsqrtf` on the
independently reconstructed variances matches all 720 inverse-standard-deviation
values exactly. Combining reconstructed Welford statistics, that CUDA operation,
and fused output arithmetic gives **zero output differences** across all 48
normalizations.

This is a causal arithmetic result, not native CPU/Vulkan parity: the diagnostic
still uses CUDA for reciprocal square root. Native implementation and full-chain
validation remain outstanding. The experiment scripts, source snapshot and
hashes, `welford-audit.json`, `rsqrt-audit.json`, `rsqrt-values.npz`, and
`combined-audit.json` are retained under `generated/soma/zombie-norm-stats`.
The direct GPU probe compiles a cubin for `sm_120`, since the installed NVRTC's
PTX version is newer than the driver's JIT supports. All 18 existing reference
tool tests pass after adding the capture utility.

### Native CPU Welford normalization trial

`src/soma_welford.hpp` implements the verified four-value Welford accumulation,
warp/inter-warp reduction order, and fused scale/output expression for SOMA's
512- and 1024-wide normalizations. Its reciprocal square root is rounded from a
double-precision calculation, so it deliberately does **not** claim to reproduce
CUDA `rsqrtf`. The native isolated probe (`soma-norm-probe ... cpu welford`)
measures 18 half-rounded differences across 630,784 values on turning, matching
the Python diagnostic. See `zombie-welford-native-norm.json`.

`MOTIONBRICKS_SOMA_WELFORD_TRIAL=1` enables a CPU-only GGML custom normalization
node in the mixed-FP16 runner; it can be combined with the aligned-matrix trial.
FP32 and public/default model inference retain their original normalization.
Output/layer GGUFs and comparison/chain reports record
`arithmetic.norm_policy=welford-rsqrt-rn-fma-trial` (or the corresponding report
field). Vulkan and FP32 runner requests reject this normalization mode.

With captured input features and aligned remainder-first products, the first
turning transition improves from three pose-token mismatches to one; maximum
raw rotation error is 12.958 degrees. All 400 semantic taps are compared. The
first differing FP16 semantic tap moves from shared-transformer layer 1 to
`_shared_transformer_model.layers.4.linear1` (147 values, max 0.001953125), after
an isolated normalization boundary known to differ under the CPU reciprocal
square root. Reports are `zombie-welford-reference-features*`. This diagnostic
still does not validate native encoding or satisfy motion parity.

The new `motionbricks-soma-welford` test covers constant-input normalization,
exact symmetric mean/variance, both supported widths, and rejection of an
unsupported width. It passes along with the aligned-dot test, default SOMA
feature-batch parity, G1 root parity, and all 18 Python reference-tool tests.

The three-step native-encoder turning chain also completes with this combined
trial. Replay has 12 pose-token mismatches, maximum joint error 43.469 mm and
13.335 degrees; closed loop has 29 mismatches, 67.733 mm and 13.335 degrees.
Durations match, but both modes fail strict acceptance. Compared with the older
CPU `conv-fix` baseline (16 replay / 27 closed-loop mismatches), this is not a
uniform improvement. The better captured-feature diagnostic must not be used to
claim native-encoder or closed-loop parity. The full report, failed acceptance,
and demo are in `chain-zombie-half-cpu-welford-trial`.

### Source-matching SOMA encoder geometry

Native quaternion multiplication now preserves Warp's Y/Z summation order, and
hip heading normalizes the across vector before computing its angle, as the
pinned upstream kernel does. Algebraic simplification of either operation can
change a normalized FP16 input and ultimately a discrete pose decision.

Across three replay inputs each for walking, sitting, and turning, all raw
global-root, local-root, and pose features now match the independent captures
exactly. Quaternion ordering alone left heading-related differences; reports
for the separate stages are `quaternion-order-encoder-validation/report.json`
and `heading-order-encoder-validation/report.json`. Two captures still each
have one differing normalized FP16 pose value after T-pose conversion and
normalization; exact raw encoding does not establish exact neural inputs.

The playback test's `--exact-encoding` mode checks raw features and masks against
these fixtures without running a decoder. Nine conditional CTest registrations
pass in both builds. FP32 CPU playback, C API, batched motion, and G1 root
regressions also pass after the geometry corrections.

The combined CPU aligned-product/Welford trial with the corrected native
encoder is `chain-zombie-half-cpu-geometry-welford-trial`. All durations match.
Replay has 11 pose-token mismatches, 43.469 mm maximum joint-position error,
and 12.049 degrees maximum joint-rotation error; closed loop has 24 mismatches,
67.248 mm and 10.865 degrees. Strict acceptance still fails in every step.
The directory contains the report, acceptance result and demo overlay. These
experimental arithmetic modes remain opt-in; SOMA mixed-FP16 parity is unfinished.

After rebuilding Vulkan, the FP32 C API check passes (max component error
2.46763e-5). The three-request motion-batch regression now **fails** its contact
ceiling: contact error 1.75744e-4 exceeds 1e-4. Position error 1.23597e-5 m and
rotation error 0.00364384 degrees remain within their limits. This supersedes the
older passing Vulkan motion-batch figures above for the current implementation;
the source-matching encoder correction is retained, and this downstream
numerical discrepancy needs resolution without relaxing acceptance.

### Low-height Vulkan error isolation and full FP32 tracing

The corrected low-height native encoder matches all captured raw inputs exactly.
Running with `reference-features` produces identical output tensors, including
the same 1.55449e-4 single-request contact error. The discrepancy is downstream
of encoding. Three duplicate low-height requests also fail (contacts 1.26839e-4);
three duplicate one-target and shorter-walk requests pass (1.20997e-5 and
6.31809e-6 contacts). Logs are `*/geometry-vulkan-motion.log` in those fixture
directories.

An experimental centered-variance Vulkan LayerNorm shader lowered the mixed
three-request contact maximum from 1.75744e-4 to 1.31190e-4, still above the
unchanged ceiling. It was reverted and the standard shaders rebuilt. The patch
is retained only as `generated/soma/vulkan-centered-norm-trial.patch`; GGML has
no retained modifications from this experiment.

`MOTIONBRICKS_SOMA_TRACE_LAYERS=1` now enables the runner's semantic neural taps
for FP32 as well as the already traced mixed-FP16 path. Layer-file precision is
recorded correctly. On the low-height Vulkan request, enabling tracing preserves
every ordinary output tensor bitwise. `compare_soma_layers.compare` understands
the sequential root/pose/decoder labels, with a regression test preventing
confusion between the identically named root and pose projections. All 19
reference-tool tests pass. The CLI's existing capture-provenance gate remains
unchanged; a playback-input trace does not automatically satisfy its isolated
fixture checksum contract.

The isolated low-height trace maps 403 module outputs with no uncovered
reference modules. Its decoder output maximum error is 3.86238e-5. Full native
inference, verified to use identical raw inputs, has a 3.91245e-4 decoder-layer
maximum and 1.38855e-3 pose-logit maximum (pose choices still match). The ordinary
final output maximum is 1.55449e-4. Reports are
`height-geometry-isolated-layers/report.json` and
`height-geometry-full-layers.json`; the latter records its diagnostic provenance
and was produced through the comparison function after explicitly verifying raw
input equality, not by bypassing the CLI's provenance gate. This identifies
propagated neural error as the next investigation target; it is not a parity pass.

### FP32 root accumulation and CUDA reciprocal-square-root audits

Splitting the seven largest root projections into eight FP32 reductions did
not resolve low-height Vulkan parity. With contiguous operand slices, the
three-request maximum contact error was 1.61469e-4 (baseline 1.75744e-4).
Noncontiguous slices were substantially worse, consistent with the previously
identified Vulkan packing precision issue. The experiment was reverted and the
normal runner rebuilt; its patch and measurements are
`generated/soma/fp32-root-split-trial.{patch,json}`.

For the remaining CPU mixed-FP16 LayerNorm discrepancy, a model-independent
CUDA `rsqrtf` audit tests exponent scaling. Normalizing positive inputs into
[1,4), applying the instruction there, and scaling back gives identical results
for all 100,720 tested values (100,000 seeded random values plus 720 captured
normalization statistics). See `rsqrt-scale-audit/{probe.py,report.json,values.npz}`.

An exhaustive scan of all 16,777,216 FP32 inputs in [1,4) finds that the CUDA
instruction differs from a double-derived, FP32-rounded reciprocal square root
by -2, -1, 0 or +1 ULP. There are eight -2 cases; sampling alone missed them.
Encoding these four corrections in two bits produces a 4,194,304-byte table,
SHA256 `002e5ef8ce6e36e6beec010a9e941a82c6a8ba997f273b3ab05fbe5d1eb86ba2`.
Lookup plus exponent scaling reproduces all 100,720 separately captured values
exactly. Artifacts are `rsqrt-mantissa-table/{generate.py,manifest.json,rsqrt-sm120-corrections.bin}`.
This is diagnostic instruction emulation for the measured sm_120 CUDA runtime,
not model weights, a portable CUDA guarantee, or a native parity result.
Native normalization integration and full-chain verification remain outstanding.

### Native CPU instruction-calibrated normalization trial

`MOTIONBRICKS_SOMA_RSQRT_TABLE=PATH` loads the pinned 4 MiB correction table
into the CPU runner only when `MOTIONBRICKS_SOMA_WELFORD_TRIAL=1` is also set.
The loader checks its exact size and pinned content fingerprint before exposing
immutable bytes to graph workers. Output and layer GGUFs record the table's
SHA256 and `welford-sm120-rsqrt-table-fma-trial` arithmetic policy. Comparison
and acceptance reports expose that identity. No CUDA call occurs during native
normalization. Public/default inference remains unchanged.

The native isolated probe now reproduces **all 630,784 values from 48 captured
normalizations bitwise**, including the FP32 outputs before half rounding:
`zombie-calibrated-native-norm.json`. The new conditional rsqrt-table CTest also
reproduces 100,720 independent CUDA values exactly and rejects unsupported
zero, negative, subnormal and nonfinite inputs. It passes alongside Welford,
aligned-dot, default SOMA feature-batch, G1 root and 19 reference-tool tests.

Combined with aligned products, the first native turning transition now has
one pose-token mismatch and 5.253 degrees maximum raw rotation error. Every
FP16 semantic tap through all six shared root-transformer layers agrees.
The first differing FP16 tap is the first root-token transformer's input
projection (`_root_token_transformer_model.layers.0.self_attn.in_proj_bias`):
50 output values differ. Reports are `zombie-calibrated-native{,-layers}.json`.
The isolated source MatMul has 80 mismatches under the current aligned16/25-bit
model; neither ordinary FP32/FP64 accumulation nor tested 8/32/64-term variants
match it. Its two attention products do match the existing aligned model.
See `zombie-root-token-qkv{,-block}-audit.json`.

The full three-step `chain-zombie-half-cpu-calibrated-trial` has exact durations
but still fails strict acceptance at every step. Replay has 13 pose-token
mismatches, 48.438 mm maximum joint error and 12.049 degrees; closed loop has
16 mismatches, 58.116 mm and 9.533 degrees. This improves closed loop relative
to the uncalibrated Welford trial but worsens replay; it is not uniform motion
parity. The failed acceptance and corresponding demo are retained in that
directory. Hardware calibration remains diagnostic-only pending broader
arithmetic and chain validation.

### Root-token projection kernel attribution

The first remaining calibrated FP16 discrepancy is independently reproducible
in the root-token transformer's first QKV MatMul. cuBLAS logs show
`cublasGemmEx`, FP16 inputs/output with `CUBLAS_COMPUTE_32F`, non-transposed
column-major dimensions M=1536, N=25, K=512. The isolated CUDA output matches
all 38,400 full-capture values exactly.

`profile_soma_matmul.py --model MODEL --capture CAPTURE --node SOURCE_NODE
--cupti LIBCUPTI --output REPORT` now validates pinned source/capture checksums,
requires exact isolated-output agreement, records the arithmetic-relevant CUDA
runtime, hashes the profiling library, and captures launch kernel names through
CUPTI. It does not alter the reference or claim full-model parity. CUPTI
13.5.25 was installed separately under `/tmp/soma-cupti`, without modifying
the reference environment.

`root-token-qkv-kernel-profile.json` identifies
`nvjet_sm120_hsh_mma_64x16x128_4_16x16x128_tmaAB_alignCD4_bz_NNNN`.
The already-matching shared-transformer projection instead uses
`cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align8`
(demangled stem), with all 13,824 isolated output values matching its capture;
see `root-shared-qkv-kernel-profile.json`. Thus the current aligned arithmetic
model's success on the shared transformer does not establish the NVJET kernel's
reduction behavior.

Additional half-partial, FP32-partial, SIMD-lane, permutation, and 128/256/512-term
alignment hypotheses all leave discrepancies. These exploratory candidates were
not adopted. Artifacts are `zombie-root-token-qkv-{split,wideblock}-audit.json`,
`root-token-qkv-{fp32-partials,simt,order}-hypotheses.json`, and their retained
experiment scripts. The SIMD sweep uses a diagnostic subset (all FP64-disagreeing
outputs plus seeded random outputs), not an exhaustive projection pass. All 19
reference-tool tests still pass. Native FP16 and Vulkan FP32 contact parity
remain unfinished; the next arithmetic investigation should target the observed
NVJET kernel rather than infer its behavior from the CUTLASS kernel.

### Disassembly-derived native NVJET QKV reduction

CUPTI module callbacks captured the loaded QKV cubin. NVIDIA nvdisasm 13.5.37
shows `HMMA.16816.F32` operations feeding three persistent accumulators per
output, followed by sequential FP32 additions. Within each 128-wide K tile,
16-term blocks go to accumulator `(block_within_tile % 3)`; this assignment
restarts at the next tile. The accumulator values themselves persist across
tiles. The earlier power-of-two partial-reduction sweeps did not model this.
Cubin, disassembly, scripts and content hashes are retained under
`root-token-qkv-cubins`; the capture still reproduces the upstream output exactly.

Using this order with the previously verified 16-term alignment arithmetic
matches every value for both root-token transformer QKV projections across
one-target walking, four-target walking and turning (six independent products).
Resetting the assignment every 128 inputs matters: a continuous modulo-three
assignment leaves 17–30 mismatches per product. Reports are
`root-token-qkv{,-layer1}-nvjet-order.json`.

The native `soma_nvjet_qkv_dot` and GGML custom product implement this order.
Only the CPU aligned-matrix trial selects it, for the two 512→1536 root-token
QKV projections. Its recorded matmul policy is now
`aligned16-nvjet-qkv-trial`. FP32 and default/public inference are unchanged.
`motionbricks-soma-matmul-probe ... cpu nvjet-qkv-graph` reproduces all 38,400
turning projection values exactly (`zombie-root-token-qkv-native.json`); a
conditional CTest requires zero error against those independent operands.

Full native turning inference with calibrated normalization now first differs
at root-token transformer layer 0's attention output projection: 24 half values,
maximum 0.00048828125. Duration matches, but one pose token still differs and
maximum raw rotation error is 5.181 degrees. Reports are
`zombie-nvjet-qkv-native{,-layers}.json`; these are not parity passes.

The profiler now also supports source Gemm nodes, preserving their attributes
and bias inputs. The next failing output projection matches all 12,800 captured
CUDA values in isolation and uses a broadcast-bias kernel followed by
`nvjet_sm120_hsh_mma_16x32x128_8_16x8x128_tmaAB_alignCD4_bx_TNNN`
(`root-token-outproj-kernel-profile.json`). Its arithmetic has not yet been
assumed equivalent to the QKV kernel.

### Native NVJET attention output projection and internal taps

The captured `16x32x128` output-projection kernel uses eight independent
FP32 HMMA accumulators, interleaving 16-term blocks across the 128-wide tile.
It sums the accumulators sequentially in FP32 and adds the bias before storing
half output. This reproduces both root-token attention output projections
across all three independent captures exactly. Artifacts are
`root-token-outproj-nvjet-order.json` and `root-token-outproj-cubins` (binary,
disassembly, capture scripts and hashes).

The aligned CPU trial now uses three accumulators for root-token QKV and eight
for its attention output projection, with the appropriate separate MatMul/Add
versus fused Gemm bias treatment. Its matmul policy is
`aligned16-nvjet-attention-trial`. A dispatch correction is essential: GGML
truncates the 64-character output weight name to 63 bytes, so the selector uses
the shorter bias name. Default/public arithmetic is unchanged.

`export_soma_matmul.py` can export unit-alpha/beta Gemm nodes with vector bias,
preserving transpose attributes and independent expected outputs.
`motionbricks-soma-matmul-probe ... cpu nvjet-outproj-graph` matches all 12,800
turning output-projection values exactly; the new conditional CTest requires
zero error. The QKV regression, aligned-dot, default SOMA feature-batch, G1 root,
and 19 reference-tool tests also pass.

Root attention now records score-product, FP32 probability, and merged-value
taps. The mixed comparator maps these to explicit ONNX nodes, distinguishing
the shared transformer's Reshape_9 from the root-token transformer's Reshape_12.
The turning full-run report compares 442 of 446 retained native taps against
351 unique reference outputs. These taps showed that probability values can
differ in FP32 while merged half values still agree, and exposed the dispatch
problem without incorrectly blaming propagated softmax error.

With the corrected selector, root-token layer 0 attention output agrees in full
native inference. The first differing FP16 tap is now its `linear1` projection:
34 of 51,200 values, maximum 0.001953125. The transition still has one pose-token
mismatch and 5.067 degrees maximum raw rotation error. Reports are
`zombie-nvjet-attention-dispatch-native{,-layers}.json`. Earlier
`zombie-nvjet-attention-native` and `zombie-attention-boundaries-native` artifacts
predate the selector correction. Full-chain parity remains unestablished.

### Root-token feed-forward reductions and the next softmax boundary

The first feed-forward MatMul uses the same `64x16x128` three-accumulator NVJET
kernel as QKV. Both transformer layers match all three captures under that
model. The second projection uses its split-K variant plus a cuBLAS reduction.
CUPTI records grid `[16,1,4]`: four partitions consume interleaved 128-wide K
tiles. Each partition uses the three-accumulator model, writes FP32 partials,
and the final reduction sums those partials before half rounding. This also
matches both layers across all three captures exactly. See
`root-token-linear{1,2}-kernel-profile.json`,
`root-token-linear1{,-layer1}-nvjet-order.json`,
`root-token-linear2-nvjet-order.json`, and `root-token-linear2-cubins`.

The opt-in CPU trial now implements these feed-forward reductions, recording
`aligned16-nvjet-root-trial`. `soma_nvjet_split4_dot` preserves FP32 partials;
it must not be confused with the older eight-way half-partial root projection
rule. The native `nvjet-split4-graph` probe matches all 12,800 captured turning
values exactly, with a conditional zero-error CTest. Six targeted native tests
(exact QKV, output projection, split4, aligned-dot, default SOMA feature batching,
and G1 root) and all 19 reference-tool tests pass.

In full turning inference, every FP16 tap through root-token transformer layer
0 now matches. The first differing half tap is layer 1's merged attention values
(13 of 12,800; maximum 0.00030517578125). Its probabilities have three differing
half-rounded values, at head/query/key `(0,19,15)`, `(11,10,3)`, `(14,1,7)`.
The subsequent value multiplication is exact when given captured probabilities
(`zombie-root-token-layer1-values-audit.json`), isolating softmax arithmetic as
the next target rather than another matrix reduction.

The full transition still fails parity: duration agrees, but two pose tokens
differ and maximum raw rotation error is 13.464 degrees. Reports are
`zombie-nvjet-root-native{,-layers}.json`. Correcting earlier local arithmetic
has worsened this transition's aggregate error while moving the first mismatch
later; these results must not be presented as a motion-parity improvement.
Default/public inference remains unchanged, and full-chain validation remains
outstanding for this experimental policy.

### Root softmax calibration (diagnostic, sm_120 only)

ONNX Runtime v1.32.0's warp softmax uses lane-strided accumulation followed by
an XOR reduction and division. Feeding the measured CUDA `expf` results through
that reduction reproduces all 93,312 captured values across 24 turning softmax
nodes exactly. Disassembly shows fused range reduction followed by `MUFU.EX2`.
Correctly rounded CPU exponentials alone still produce different half values.
The source snapshot, probe scripts, and script checksums are retained under
`generated/soma/zombie-softmax-audit/scripts`; instruction cubins, disassembly,
and detailed audits are in its parent directory.

An exhaustive diagnostic table records EX2 ULP corrections for all 201,326,592
float32 arguments in `[2^-24,1)`. The 48 MiB artifact and manifest live under
`generated/soma/exp2-fraction-table`; SHA256 is
`11d89bff841e859226e57a64735f294cd2cc2b77665f25b2b953ea3ff83fcbc5`.
This is hardware instruction calibration, not model weights or a portable CUDA
contract. Reduced arguments outside that interval use double `exp2` rounded to
float. Against this capture, that fallback leaves two exponential differences
and one softmax difference of one minimum subnormal float; all half-rounded
softmax values agree. The native regression independently reproduces these
counts on the 24 captured nodes. This does not establish other inputs' parity.

Set `MOTIONBRICKS_SOMA_EXP2_TABLE` to the table path to enable the CPU
mixed-FP16 runner's root softmax diagnostic. It rejects other backend/precision
combinations and records the table hash plus
`warp-sm120-exp2-table-with-fallback-trial` in output and layer trace metadata.
Default inference and pose-model softmax remain unchanged. Combined with the
existing aligned-matmul and calibrated Welford trials, the turning transition
now agrees at every half tap through both root-token transformer layers.
The first half mismatch is `_conv_output.model.0`: 6/8,192 values, maximum
0.00390625. Reports are `zombie-softmax-native{,-layers}.json`.

End-to-end parity still fails: duration matches, two pose tokens differ, and
maximum raw joint rotation error is 13.484 degrees. This moves the first local
mismatch later without improving aggregate motion accuracy. Full-chain and
Vulkan acceptance remain outstanding; no tolerances were relaxed.

Validation after this change: nine targeted native tests pass (including the
new softmax fixture, prior arithmetic probes, default SOMA feature batching,
and G1 root parity); all 19 reference-tool tests pass.

### First root convolution: 64-channel tile traversal

The profiler now accepts pinned Conv nodes as well as MatMul/Gemm and warms the
isolated operation before recording launches, excluding cuDNN algorithm search.
It still requires exact agreement with the independently captured full graph.
The first root Conv uses cuDNN's
`sm80_xmma_fprop_implicit_gemm_f16f16_f16f32_f32_nhwckrsc_nhwc_tilesize64x32x64_stage5_warpsize2x2x1_g1_tensor16x8x16_execute_kernel__5x_cudnn`
with grid `[16,1,1]`, block `[128,1,1]`. Separate layout conversions and bias
addition are observed. See `root-first-conv-kernel-profile.json`,
`root-first-conv-kernels.json`, and the cubin/disassembly/scripts/checksums under
`root-first-conv-cubins`.

Its reduction traverses 64-channel tiles, then spatial kernel positions, then
channels within each tile. Aligned 16-product arithmetic in that order matches
all first-Conv outputs in the walk-one, walk-four16, and turning captures.
The native channel/spatial order differs in 5, 7, and 6 values respectively;
a simple spatial/channel reorder differs in 16, 35, and 20. These alternatives
are retained in `root-first-conv-layout-audit.json`.

The CPU aligned-matmul trial now reorders both im2col and kernel operands for
`_conv_output.model.0` only, checking its 3-by-512 reduction shape. It records
`aligned16-nvjet-root-conv64-trial`. Default inference is unchanged. Applying
the same rule indiscriminately fails other root convolutions
(`root-all-conv-layout-audit.json`); the first residual Conv instead selects
a `64x64x64_stage4` cuDNN kernel (`root-residual-conv1-kernel-profile.json`).

In full turning inference, the first Conv, its ReLU, and external conditioning
projection now match exactly. The first differing half tap is
`_conv_output.model.2.0.model.0.conv1`: 23/8,192 values, maximum 0.0009765625.
Reports are `zombie-conv64-native{,-layers}.json`. Duration still matches, two
pose tokens still differ, and maximum raw rotation error remains 13.484 degrees.
This is a verified local correction, not an end-to-end parity pass.

### Residual convolution partitions and root-kernel survey

The first residual Conv launches the `64x64x64_stage4` cuDNN kernel with grid
`[8,1,12]`, block `[128,1,1]`. The twelve-way reduction uses FP32 scratch and
FP32 additions before half storage, as confirmed by disassembly. On all three
captures, twelve contiguous 128-value partitions of the tiled operands followed
by a sequential FP32 sum reproduce its outputs exactly. Half-rounded partials
produce thousands of mismatches. See `root-residual-conv1-split-audit.json` and
`root-residual-conv1-cubins` (scripts, cubin, disassembly, checksums).

A separate survey profiles all 21 root Conv nodes, requiring each isolated
output to match the full turning capture. `root-conv-kernel-profiles.json`
records every selected kernel and launch grid. Non-split spatial convolutions
use the validated 64-channel traversal. Pointwise convolutions use two NVJET
variants with eight persistent K-block accumulators. Together with the first
residual split rule, these explain all outputs of 18 root Conv nodes across
three captures. The complete audit covers 861,120 values in 63 node/capture
pairs (`root-conv-profiled-order-audit.json`).

Three other split Conv nodes still reject that reduction rule: stage 2 block 1,
and stage 3 blocks 0 and 1. They account for 41 isolated differences. An epilogue
hypothesis that sums the last partition first worsens the aggregate to 46 and
also breaks the first residual Conv; it was rejected, not integrated
(`root-conv-profiled-last-first-audit.json`). Their native arithmetic remains
unchanged pending a verified partition/reduction model.

The opt-in CPU trial now records `aligned16-nvjet-root-conv-profiled-trial` and
implements only the validated families. In full turning inference, the first
residual block now matches. The first differing half tap is
`_conv_output.model.2.0.model.1.conv1`: 43/8,192 values, maximum 0.001953125.
The maximum raw rotation error drops from 13.484 to 12.553 degrees, but duration
agreement and two pose-token mismatches are unchanged. Reports are
`zombie-conv-profiled-native{,-layers}.json`; the intermediate first-residual-only
run is `zombie-conv-split12-native{,-layers}.json`. These are diagnostic results,
not a parity pass. All nine targeted native arithmetic/default SOMA/G1 tests
pass after rebuilding the affected targets.

### Split convolutions partition channels, not flattened K

Direct capture of the launch block and bounded FP32 workspace resolved the
remaining split Conv discrepancy. The kernel metadata declares one 1,488-byte
parameter block at constant-bank offset 0x380. Its scratch-base/counter-base
bounds delimit eleven 131,072-byte slots. Eight contain active partition
outputs; the remaining slots are zero. Each active partition covers 64 channels
and all three spatial positions (192 flattened products). The twelve-block
launch does not mean twelve equal slices of flattened K.

For stage 2 block 1, every active partition's sorted FP32 scratch values matches
the independently modeled partition, including zero padding. This comparison
does not decode the scratch's physical swizzle. Evidence, launch capture,
parameter block, scratch bytes, scripts and checksums are retained under
`root-split-params`, including `scratch-validation.json`.

Using eight 192-value aligned partitions and a sequential FP32 sum now matches
all 861,120 outputs of all 21 root Conv nodes across the three captures:
`root-conv-channel-partition-audit.json`. The earlier twelve-by-128 hypothesis
happened to match the first residual Conv, whose large dilation masks many
spatial inputs, but was not the correct general model. Spatial-major partitioning
and last-partition-first summation were also rejected; their reports remain
available. No failed hypothesis was used to relax acceptance.

The CPU trial now records `aligned16-nvjet-root-conv-channel-split-trial` and
implements the channel split for the profiled dilated root convolutions. Full
turning inference first differs at `_conv_output.external_cond_blocks.2`:
78/16,384 half values, maximum 0.00390625. Thus later native convolution taps can
still differ due to conditioning input propagation even though isolated
convolution arithmetic is exact. `zombie-conv-channel-native{,-layers}.json`
records duration agreement, two pose-token mismatches, and 12.722 degrees maximum
raw rotation error. Nine targeted native arithmetic/default SOMA/G1 regressions
pass. Full-motion parity remains unestablished.

The current three-transition turning replay is
`chain-zombie-half-cpu-root-conv-trial`, with freshly recomputed
`acceptance.json` and `demo.json`. All durations agree. Replay has 8 pose-token
mismatches, 66.235 mm maximum joint-position error and 10.539 degrees maximum
rotation error; closed loop has 25 mismatches, 79.308 mm and 10.539 degrees.
All six per-transition replay/closed-loop comparisons fail the unchanged strict
policy. These numbers are measured corrected chained playback, not the raw
single-transition rotation metric above. The demo preserves both full motion
tracks and displays the explicit failed status.

The next conditioning projection is independently attributed to NVJET's
`64x16x128` split-K kernel plus cuBLASLt reduction, with exact isolated CUDA
output (`root-external-cond2-kernel-profile.json`). Its native reduction rule
is the next unresolved boundary; no change to that projection was made here.

### Conditioning projection and pose attention boundaries

`_conv_output.external_cond_blocks.2` has reduction width 1,536 and output width
512. It uses NVJET's `64x16x128` split-K kernel with grid `[16,1,4]`, followed by
cuBLASLt reduction. The existing four-partition, three-accumulator model matches
all three captures exactly (`root-external-cond-nvjet-order.json`). The native
`nvjet-split4-graph` fixture regression matches all 16,384 captured values.
Launch evidence and audited scripts are in `root-external-cond2-cubins`.

The first dispatch guard incorrectly used the earlier conditioning projection's
2,560-wide shape and therefore did not execute. `zombie-cond2-native` artifacts
predate that guard correction. The corrected run is
`zombie-cond2-dispatch-native{,-layers}.json`: all retained root FP16 taps match,
duration and pose-token decisions agree, and maximum raw rotation error drops
from 12.722 to 0.125337 degrees. This is still above strict motion tolerance.

Pose attention now names score-product, FP32 probabilities, and merged-value
taps in both single and batch graphs. The mixed comparator maps these to
explicit source MatMul_1, Softmax, and Reshape_12 nodes. Coverage increases to
490/494 retained native taps and 399 unique reference outputs. The opt-in
calibrated softmax also handles pose attention's additive mask, with a separate
`arithmetic.pose_softmax_policy` recorded in native, transition, layer, and
chain metadata. Default arithmetic remains unchanged.

`zombie-pose-softmax-native{,-layers}.json` reveals that the first remaining
half difference precedes softmax: pose layer 1's score product differs in one
of 4,096 values; layer 2 differs in three. The new softmax path leaves this
transition's raw motion metric unchanged. CUPTI identifies the score kernel as
`nvjet_sm120_hsh_mma_32x16x128_8_16x8x128_tmaAB_alignCD4_bz_NNNN`
(`pose-score-kernel-profile.json`).

The 64-value score reduction uses four active 16-product accumulators, with the
remaining four NVJET accumulator slots zero, followed by FP32 additions. It
matches every score output in all sixteen pose layers across all three captures
(`pose-score-nvjet-three-capture-audit.json`). Sequential aligned-16 arithmetic
has 42 differing half values across the same cases; other precision/block
hypotheses also fail. Scripts and hashes are under `pose-score-audit`.
The CPU trial now handles this observed 64-wide case explicitly and records
`aligned16-nvjet-root-conv-pose-scores-trial`; the captured native batched-head
regression `motionbricks-soma-nvjet-scores` passes with zero error.

The full score-corrected turning run is
`zombie-pose-scores-native{,-layers}.json`. Every retained FP16 root and pose tap
now matches the source capture. Duration and all pose-token decisions also
agree. The first differing half tap is `decoder.model.0`: 4/8,192 values,
maximum 0.00390625. Maximum raw motion rotation error remains 0.125337 degrees,
so full motion acceptance is still failing. This establishes the observed
root/pose layer boundaries for this input, not general chained, batch, Vulkan,
or portable FP16 parity. Twelve targeted native regressions (including the two
new captured matrix fixtures and G1 root/pose parity) and all 19 reference-tool
tests pass after rebuilding the affected targets. The previously exported
`chain-zombie-half-cpu-root-conv-trial/demo.json` predates these conditioning and
pose-score fixes and remains historical evidence, not a current-chain result.

### Decoder convolution parity and first passing replay chain

The first decoder Conv uses CUTLASS's `f16_s16816fprop_optimized_f16_64x64_32x10`
kernel. Its 160 input channels traverse 32-channel tiles, then spatial kernel
positions, then channels. This matches all three captures exactly; native
channel/spatial order differed in 1, 1, and 4 values. The remaining twenty
512-channel decoder convolutions use the already identified root families;
their 905,664 outputs across three captures also match exactly. All four
conditioning MatMuls agree with the existing residue-first aligned arithmetic.
Evidence is in `decoder-kernel-full-profiles.json`,
`decoder-first-conv-layout-audit.json`, `decoder-conv-channel-partition-audit.json`,
and `decoder-matmul-audit.json`; scripts/checksums are in
`decoder-arithmetic-audit`. The earlier `decoder-kernel-profiles.json` omitted
ordinary (non-Ex) launches and is superseded by the full profile.

`soma_conv_ggml.hpp` now shares the profiled arithmetic between root and decoder
in the opt-in CPU trial, labeled `aligned16-nvjet-soma-conv-trial`. The pinned
fixture exporter supports one-batch Conv lowering with source attributes and
bias preserved. Four zero-error CTests exercise first-Conv, split, spatial,
and pointwise branches using independent captured operands. Seventeen targeted
native regressions, including default SOMA and G1 root/pose/decoder parity,
and all 19 reference-tool tests pass.

`zombie-decoder-conv-native{,-layers}.json` reports zero differing retained FP16
layer taps across root, pose, and decoder. Duration and pose-token decisions
agree. Maximum raw joint-position error is 1.78814e-7 m, rotation error
9.95663e-6 degrees, and contact error 5.96046e-8. This full turning transition
is now within the numerical limits, while wider acceptance remains necessary.

The refreshed three-step chain is `chain-zombie-half-cpu-full-conv-trial`, with
recomputed `acceptance.json` and `demo.json`. All three identical-input replay
transitions pass strict acceptance, with zero pose-token mismatches. Across the
replay chain, maximum corrected joint-position error is 2.33264e-6 m and rotation
error 6.73995e-5 degrees. All durations agree in both modes.

Closed-loop transition 0 also passes, but transitions 1 and 2 fail position,
rotation, contact, and pose-token checks. Closed loop has 22 token mismatches,
75.396 mm maximum joint-position error, and 13.184 degrees maximum rotation
error. Its own prior corrected poses feed the next request, so passing replay
must not be reported as closed-loop parity. The overall acceptance and demo
status remain failing. Feedback pose processing is the next unresolved boundary.

### Exact Warp playback and passing turning closed loop

The native playback implementation now follows Warp CPU's arithmetic order:
quaternion normalization multiplies by reciprocal length, matrix-to-quaternion
conversion preserves branch ties and grouped sums, SLERP uses axis-angle with
`atan2`, and root endpoint translations use left-associative additions. These
operations were mathematically equivalent before, but their FP32 differences
changed feedback features and subsequent mixed-FP16 token decisions.

All nine independent walking, sitting, and turning playback fixtures now have
exactly equal decoded/corrected/world transforms, start/end deltas, input world
transforms, and alignments. Nine new `--exact-playback` regressions check direct
component equality (including quaternion sign), alongside the existing exact
encoder checks; all 19 encoder/playback tests pass.

`chain-zombie-half-cpu-warp-playback-trial` supersedes the failing chain above.
Its independently recomputed `acceptance.json` passes all six comparisons:
three replay and three closed-loop transitions. Duration and all pose-token
decisions agree. Maximum closed-loop world joint-position error is
1.45788e-6 m, rotation error 8.37221e-5 degrees, and contact error 5.96046e-8.
The corresponding verified `demo.json` shows both modes and their errors.
This is evidence for the calibrated CPU mixed-FP16 arithmetic on this sequence,
not a claim of portable, batched, or Vulkan mixed-FP16 parity.

The same revision was also rerun as `chain-walk-half-cpu-warp-playback-trial`
and `chain-sit-half-cpu-warp-playback-trial`; both include independently checked
acceptance and demo exports and still fail. Walking passes two of three replay
transitions; maximum replay position/rotation errors are 1.645 mm / 0.328 degrees.
Its closed loop reaches 27.473 mm / 4.760 degrees with one pose-token mismatch.
Sitting fails all three replay transitions (0.768 mm / 0.157 degrees maximum);
closed loop reaches 19.691 mm / 3.252 degrees with one pose-token mismatch.
Replay pose tokens agree for both sequences. These failures remain visible in
their demo exports; the passing turning chain does not supersede them.

`walk-warp-playback-layers.json` finds the earliest compared differing half tap
at root-token transformer layer 1 merged attention values (one value). The
native sequence has 17 entries while the source keeps 25 with masked padding;
attention score/probability shapes are currently unmatched in that comparison,
so this is not yet a proven first causal operation. The sitting report
`sit-warp-playback-layers.json` first differs at duration logits (one half value).
Further model-arithmetic work is required. Six rebuilt public FP32/batch and G1
parity tests and all 19 reference-tool tests pass after the playback change.

### Masked attention padding and reduction order

The mixed-layer comparator now checks both query and key axes for explicitly
named attention taps, retaining the head axis and validating the captured mask.
It rejects mismatched durations, head counts, and masks. Walking coverage rises
from 454 to 490 compared native taps (399 unique source outputs). Its root
attention scores and probabilities match exactly; the first difference is the
weighted-value reduction. Native uses 17 valid keys in that transition while
the source retains 25 slots, including zero-probability masked keys.

The CPU trial now zero-pads that reduction to 25 and processes the full
16-product block before the remainder. Isolated upstream operands from both
root-token attention layers in all three motion captures agree exactly with
this order (`root-attention-three-capture-audit.json`, six operations). The
remainder-first and eight-product alternatives each differ in four values.
The new captured `sit-root-attention-values.gguf` regression checks the native
graph with zero allowed error. All 15 targeted arithmetic, FP32 feature-batch,
and G1 regressions pass; all 20 reference-tool tests pass.

Fresh chains and verified demo exports are under
`chain-{walk,sit,zombie}-half-cpu-attention-tail-trial`, labeled
`aligned16-nvjet-soma-attention-tail-trial`. Turning still passes every replay
and closed-loop comparison. Walking now passes all three replay transitions;
closed-loop transitions 1 and 2 still fail, reaching 2.421 mm position and
0.2613 degrees rotation error. Sitting passes one replay transition; maximum
replay error is 0.715 mm / 0.1647 degrees and closed-loop error is 0.836 mm /
0.1317 degrees. All duration and pose-token decisions agree in all three chains.
These numerical failures remain failures under the unchanged strict policy.

`walk-attention-tail-closed-step1.json` shows feedback already differs at the
normalized half input boundary (one global, one local, six pose values).
`sit-attention-tail-layers.json` now matches root attention; remaining differences
include one duration logit and the first decoder external-conditioning output.
The independent duration Gemm operands are exported as `sit-duration-gemm.gguf`.
The padding-only intermediate chain is historical and still fails closed loop.
The source decoder's concatenated external-conditioning slice differs from the
native prepared half input in exactly one value (index 116: 0.01271820068359375
versus 0.0127105712890625). Thus the sitting decoder discrepancy is already
present before its MatMul; root-to-local conditioning arithmetic is the next
boundary to investigate.

### Exported normalization scales and rotation reduction

The pinned mixed export contains folded square-root scales that differ from
native scalar recomputation by one FP32 step in some components. For example,
the first global-root scale is 1.4425353 in the export versus 1.4425352 natively.
This changes a sitting root-velocity value across a half-rounding boundary.
The converter can now retain the exact scales:

```sh
.cache/soma-reference-venv/bin/python scripts/convert_soma_to_gguf.py \
  --reference generated/soma/reference \
  --mixed-onnx generated/soma/reference/motionbricks/assets/onnx/soma/mmm_v6_soma_fp16.onnx \
  --output generated/soma/export-scales
```

The optional graph must match the pinned checksum. Global/local/body scales
come from explicit constant nodes with validated shapes and agreement with the
checkpoint statistics. The extended support component has five tensors and
1,650 values; legacy four-tensor bundles remain valid. Mixed requests use the
exported scales, while FP32 requests retain their prior computation. Public
FP32 API parity passes with both bundle formats; all 20 reference tests pass.

`sit-export-scales-native{,-layers}.json` shows the decoder conditioning half
mismatch is gone, as are all downstream half layer differences. Only one
duration-logit value still differs, without changing the duration. Raw rotation
error is 6.93609e-6 degrees. The independent duration projection audit across
three captures (`duration-gemm-three-capture-audit.json`) does not establish an
exact arithmetic replacement; no duration-projection change was made.

Verified chains and demo exports are in
`chain-{walk,sit,zombie}-half-cpu-export-scales-trial`. Turning passes. Walking
passes all replay transitions but still fails two closed-loop transitions
(maximum 2.421 mm / 0.3043 degrees). Sitting passes two replay transitions;
transition 1 still fails, and closed-loop transition 2 chooses 10 tokens against
the reference's 11. Its demo retains both complete motions and marks their
post-mismatch frames unaligned. These failures must not be hidden by the improved
single-transition result.

A subsequent wrapper correction matches CUDA's three-component ReduceL2 order:
`sqrt((x*x+z*z)+y*y)`. All 12 independently captured input/output normalization
operations across the three motions match (12,960 values;
`rotation-l2-three-capture-audit.json`). It applies only to mixed requests.
`walk-rotation-order-native.gguf` reduces differing final rotation components
from 934 to 308, with maximum absolute error 1.19209e-7. Position, velocity,
and contact features are exact; 16 root components differ by at most 5.96046e-8.
The chain exports above predate this last correction. T-pose matrix arithmetic
still differs: the independent `rotation-matmul-order-three-capture-audit.json`
rejects all tested simple sequential/reordered and fully fused dot products.

### Passing replay and closed loop for all three CPU trial chains

The remaining output T-pose product uses a hybrid reduction: round product 0,
fuse product 1 into it, then add separately rounded product 2. This matches the
captured `/MatMul_2` output for walking, sitting, and turning exactly
(`rotation-matmul-hybrid-audit.json`). Other tested fusion/order combinations
do not. The mixed wrapper now applies this pattern only to that output product;
the other T-pose products and FP32 path keep their existing arithmetic.

`chain-{walk,sit,zombie}-half-cpu-rotation-matmul-trial` now passes all 18 strict
comparisons: three transitions, replay and closed loop, across three motions.
Every duration and pose-token decision agrees. Verified `acceptance.json` and
`demo.json` files accompany each chain. All three motions' playback arrays
are exactly equal to upstream for decoded, corrected, world, and contacts:
1,199,540 values across 72 array comparisons
(`rotation-matmul-exact-playback.json`). The approximately 3.4e-6-degree
reported angle for identical quaternions is numerical roundoff in the acos
metric, not motion divergence. Public FP32 API and feature-batch tests pass
after the mixed-only correction.

The isolated duration projection remains a separate diagnostic. CUPTI identifies
the source operation as a cuBLAS `internal::gemvx::kernel` specialization, and
re-executing it with captured operands is exact (`sit-duration-kernel-profile.json`).
Several lane-reduction hypotheses match the one sitting fixture
(`sit-duration-gemv-reduction-audit.json`), so that experiment does not uniquely
identify an implementation. No duration-projection change was made.
These passing chains still validate the calibrated CPU trial, not general
portable, batched, or Vulkan mixed-FP16 inference.

### Calibrated CPU batch validation

The single runner and batch probes now share `soma_trial.hpp` configuration.
Previously the feature batch probe did not read the CPU trial flags, so setting
the runner's environment could not reproduce its arithmetic in that probe.
The batch report now records the enabled arithmetic flags and calibration hashes.
Public model APIs still do not read these diagnostic environment variables.

`mixed-feature-batch-calibrated.json` compares four independent source fixtures
(walk-one, walk-four16, sit-short, and zombie) in both request orders. All eight
rows match source duration and pose-token decisions; maximum raw feature error
is 8.9407e-8. Batched output and every captured processing boundary equal native
single-request inference exactly. The test includes three requests sharing a
16-token duration and one selecting 11 tokens. This compares native batching to
independently captured source requests, not a separate upstream batched export.

The end-to-end motion batch test now accepts an optional `mixed-fp16` argument.
With the calibrated CPU settings and `export-scales` bundle, the walking,
sitting, and turning step-000 fixtures pass in both input orders with zero
position, angular, and contact error against upstream, and zero error against
native single-request playback. This also covers native skeleton encoding,
endpoint correction, mixed durations, duration grouping, and invalid-request
handling. The existing FP32 batch regression still passes.

An opt-in CTest registration reproduces that end-to-end command:

```sh
nix develop --command cmake -S . -B build/sonic-cpu \
  -DMOTIONBRICKS_ENABLE_SOMA_CALIBRATED_TESTS=ON
nix develop --command cmake --build build/sonic-cpu \
  --target motionbricks-soma-motion-batch-parity-test
nix develop --command ctest --test-dir build/sonic-cpu \
  -R '^motionbricks-soma-motion-batch-calibrated$' --output-on-failure
```

It requires GGML, the extended bundle, three captured playback fixtures, and
the pinned rsqrt/exp2 calibration tables documented above. Missing inputs fail
configuration rather than silently omitting the test. The registration supplies
the same four trial environment settings used in the successful direct run.
This extends CPU batch evidence; it does not establish Vulkan mixed precision
or integrate calibration into the public model API.

### Current Vulkan FP32 revalidation and trace provenance

Rebuilt Vulkan targets pass the walk and low-height public API checks and the
feature-batch check (`vulkan-fp32-current-validation.json`). The strict motion
batch still fails contacts: 0.000175744 versus 0.0001. Its position error is
1.2087e-5 m and rotation error 0.00364384 degrees, both within policy. The public
API test's broader component check must not be substituted for this strict gate.

Two normalization alternatives were evaluated and rejected. Centered variance
reduces contact error to 0.000131518; centered RMS gives 0.00017032. Neither
passes. Disabling Vulkan graph fusion changes neither result. The trial source
snapshots and measurements are retained under `vulkan-normalization-audit` and
`vulkan-fp32-fusion-norm-audit.json`; the unsuccessful trial branches were removed
from production source. The pinned GGML submodule was not modified.

The height-stress encoder itself matches source features exactly. To trace the
remaining inference discrepancy, `capture_soma_playback.py` now adds the verified
neural-capture hash to newly captured playback fixtures. The native runner
propagates it into output and layer-trace provenance. Legacy inputs still work;
native-derived and closed-loop fixtures do not claim an unchanged source capture.
Use `MOTIONBRICKS_SOMA_TRACE_LAYERS=1` to retain FP32 neural layer traces.

`height-provenance-io` reproduces the old upstream playback arrays exactly, with
the added provenance. `height-provenance-vulkan-layers.json` passes the strict
capture-hash check and maps 403 layers. Largest absolute discrepancies include
pose logits (0.00138855) and final normalized decoder features (0.000391245).
These are diagnostic layer measurements, not accepted end-to-end tolerances.
All 20 reference-tool regression tests pass after the provenance addition.
The same-build CPU trace (`height-provenance-cpu-layers.json`) maps successfully
too. Root final-feature maximum error is 9.53674e-7 on CPU versus 5.96046e-6 on
Vulkan; pose-logit error is 0.000268936 versus 0.00138855; decoder final-feature
error is 4.97103e-5 versus 0.000391245. This localizes a backend-dependent growth
of error through inference while shared skeleton encoding is exact.

### Exact exported scales also fix FP32 stress parity

The extended `export-scales` bundle now uses its stored normalization scales for
both FP32 and mixed-FP16 requests. All 370 values match the upstream FP32 PyTorch
expression bit-for-bit; scalar recomputation differs in 63 components. Existing
four-tensor bundles retain their previous behavior. Use the converter's
`--mixed-onnx` option documented above to produce the recommended extended bundle.

`fp32-export-scales-batch-validation.json` supersedes the strict Vulkan failure
above for this bundle: CPU and Vulkan pass all three fixtures in both orders.
Vulkan maxima are 6.78474e-6 m position, 0.00324619 degrees rotation, and
5.51045e-5 contacts (ceiling 1e-4). No tolerance was changed. Public API walking
and low-height checks pass on both backends;
`fp32-export-scales-api-validation.json` records commands and output.

All 18 Vulkan FP32 comparisons across walking, sitting, and turning pass with
independent replay and closed-loop feedback. Reports, acceptance results, and
viewer exports are under `chain-{walk,sit,zombie}-fp32-vulkan-export-scales`.
Across these chains, maximum world position error is 5.58826e-6 m, rotation
error 0.00106334 degrees, and contact error 6.23167e-5. All duration and pose-token
choices agree. These are captured-fixture results, not bounds for arbitrary
motion. The earlier four-tensor-bundle diagnostics remain useful historical
measurements but do not describe the corrected bundle.

CPU regeneration with the same extended bundle also passes all 18 chain
comparisons (`chain-{walk,sit,zombie}-fp32-cpu-export-scales`), with maxima
3.54288e-6 m, 0.000622633 degrees, and 5.03660e-5 contacts. Demo exports include
verified acceptance results. The corrected-bundle Vulkan feature-batch test
passes too: reference maximum 4.82202e-5, batch/single maximum 2.58088e-5,
precision groups isolated, and FP32 state restored after mixed requests.
Its exact command is in `fp32-export-scales-feature-batch-validation.json`.

### Explicit calibrated CPU public API

The shared library now exposes `mb_model_set_arithmetic_calibration` and
`MB_PRECISION_MIXED_FP16_SM120_EMULATED`. This uses the same calibrated CPU arithmetic as the
successful mixed-FP16 chain runner, with explicit table paths rather than trial
environment variables. It requires the extended normalization-scale bundle and
a CPU model. Ordinary experimental mixed precision remains a separate option.
See `docs/API-SOMA.md` for the call contract and limitations.

Table loading is serialized; repeated configuration validates the supplied bytes
but never replaces a published table. Public generation scopes and restores all
profile switches, including the native-half operand override. The three new
`motionbricks-soma-api-calibrated-{walk,sit,zombie}` CTests pass using independent
step-000 playback fixtures. Maximum absolute error over returned arrays is
5.96046e-8, 1.49012e-8, and 5.96046e-8 respectively. Each test also verifies
rejection before calibration, missing-file handling, repeated configuration,
output ownership, and bit-identical FP32 features before and after a calibrated
request. The preserved log is `public-api-calibrated-validation.log`.

Reproduce with the existing opt-in calibrated test configuration:

```sh
nix develop --command cmake --build build/sonic-cpu --target motionbricks-soma-api-parity-test
nix develop --command ctest --test-dir build/sonic-cpu \
  -R '^motionbricks-soma-api-calibrated-' --output-on-failure
```

This validates public single-request integration. The previously recorded chain
and batch results use the internal runner; public-API chain coverage remains a
separate validation step. The calibrated profile deliberately rejects Vulkan;
it does not establish Vulkan mixed-FP16 parity or hardware acceleration.

### Persistent public API chains

`check_soma_api_chain.py` loads one shared-library model and skeleton for a whole
chain and calls only the public API for generation. It executes all reference
contexts and independently feeds back its own last four corrected poses. It
checks the pinned neural capture and playback provenance, input fixture hashes,
reference feedback continuity, exact durations, and the existing position,
rotation, and contact ceilings. Reports retain library/calibration hashes,
per-frame errors, and hashes of saved output/context arrays. Pose tokens are not
exposed by this API and remain validated by the internal chain harness.

All 36 FP32 public API comparisons pass across walking, sitting, and turning on
CPU and Vulkan (`public-api-chain-{walk,sit,zombie}-fp32-{cpu,vulkan}`). Maximum
world position/rotation/contact errors are 4.81126e-6 m / 0.000496459 degrees /
3.63589e-5 on CPU and 5.58826e-6 m / 0.00106334 degrees / 6.23167e-5 on Vulkan.
See `docs/API-SOMA.md` for the invocation. The reference-tool suite now has 21
passing checks, including rejection of wrong durations, truncated playback,
contact-limit failures, and nonfinite contacts in the public chain comparator.

All 18 calibrated CPU mixed-FP16 public API comparisons pass as well
(`public-api-chain-{walk,sit,zombie}-mixed-validated`). Position and contact error
are zero; maximum reported angular error is 3.81820e-6 degrees (the geometric
metric's roundoff). A separate equivalence audit compares all returned raw
features, corrected/world transforms, and contacts to the validated internal
runner captures: 72 tensor comparisons, 1,111,048 values, zero differences.
The audit is `public-api-mixed-runner-equivalence.json`. Thus the existing
`chain-{walk,sit,zombie}-half-cpu-rotation-matmul-trial/demo.json` exports display
exactly the public API's chained motion for these fixtures.

### Current semantic layer coverage and browser validation

The calibrated step-000 captures for walking, sitting, and turning each map all
490 retained semantic output taps to the independent mixed-FP16 CUDA export
(399 distinct upstream outputs). The four additional retained taps are weight
copies, not layer outputs. The comparator now reports these as explicitly
excluded weight conversions and separately reports semantic coverage; a missing
or shape-mismatched semantic output makes coverage incomplete. This does not
claim every ONNX constant/shape operation is a model layer.

Reports are `{walk,sit,zombie}-current-semantic-layer-coverage.json`. Walking
matches all compared values exactly. Sitting retains one duration logit differing
by 0.0078125 (observed twice in the native trace) and one smallest FP32 subnormal
softmax difference. Turning retains the same subnormal-sized softmax difference.
Duration and pose-token selections and all tested final playback remain equal.
These reports describe layer measurements, not a claim of bit-exact arithmetic
at every internal operation. All 22 reference-tool regression checks pass.

The actual Chromium `TestComparisonHTTPFlow` and `TestHeadlessComparison` pass
with the calibrated walking demo. They exercise reference/native skeletons,
error vectors, per-frame root/joint/rotation errors, chain acceptance text, and
both replay and closed-loop plans through the application.

The same two browser tests pass for sitting. The retained screenshot
`public-api-sit-demo.png` was visually inspected: upstream and native skeletons
overlap, mixed-FP16 acceptance is visible, replay controls are usable, and the
current-frame position and rotation errors are displayed beside the animation.

### Reproducible instruction calibration

`reference/generate_soma_calibration.py` now generates both pinned tables from
GPU instructions, with no model or fixture inputs. It requires compute capability
12.0, compiles an sm_120 cubin with NVRTC, records source/cubin hashes, measures
every float input in the specified domains, and rejects a resulting checksum
that differs from the pinned artifact. Existing output table directories are
never overwritten. Reproduction command used here:

```sh
nix develop --command env -u PYTHONPATH .cache/soma-reference-venv/bin/python \
  reference/generate_soma_calibration.py \
  --nvrtc .cache/soma-gpu-reference-venv/lib/python3.12/site-packages/nvidia/cu13/lib/libnvrtc.so.13 \
  --output generated/soma/calibration-reproduced
```

Both regenerated files match the existing hashes exactly: rsqrt
`002e5ef8ce6e36e6beec010a9e941a82c6a8ba997f273b3ab05fbe5d1eb86ba2` and exp2
`11d89bff841e859226e57a64735f294cd2cc2b77665f25b2b953ea3ff83fcbc5`.
The output directories retain manifests, CUDA source and cubins. This closes the
previous reliance on local, untracked generator experiments.

### Current Vulkan mixed-FP16 revalidation

Rebuilt the current runner and repeated all three chains using exact exported
normalization scales and corrected rotation conversion. Both the rounded-FP32
operand path and the opt-in native-FP16/split-root path fail all 18 comparisons
per mode under the unchanged strict gates. Results and failed-verdict demo
exports are under `chain-{walk,sit,zombie}-mixed-vulkan-current-{rounded-f32,native-f16-split}`.
The aggregate is `vulkan-mixed-current-validation.json`.

For the native-FP16 split trial, walking/sitting retain exact pose-token choices
but reach 2.54101 mm / 0.335533 degrees and 1.02925 mm / 0.167576 degrees,
respectively. Turning has 33 pose-token mismatches and reaches 67.1716 mm /
13.9084 degrees across replay and closed loop. Rounded-FP32 operands also have
pose-token mismatches; sitting additionally has duration/shape mismatches, so
its comparable-shape error maxima are not whole-chain bounds. These results
supersede older Vulkan mixed trials for the current source. The public calibrated
CPU mode rejects Vulkan explicitly; no Vulkan mixed-FP16 parity is established.

After the public calibration integration, a full rebuild and all 77 configured
CPU CTests pass (`current-regression-build.log`, `current-regression-tests.log`).
This includes G1 root/pose/decoder and open-loop parity, agent integration,
SONIC/SMPL checks, Go/demo tests, SOMA API tests, and calibrated motion batching.
The full run completed in 202.21 seconds; calibrated batching passed its strict
fixture gates. This broad regression result does not change the separately
recorded Vulkan mixed-FP16 failures.

### Registered exact-scale parity gates

`MOTIONBRICKS_SOMA_TEST_BUNDLE` now selects the SOMA bundle consistently across
registered tests, defaulting to `generated/soma/export-scales`. Tests no longer
silently use the historical scalar-scale bundle while documentation recommends
the extended one. Explicitly selecting that older bundle still reproduces its
known failures. The calibrated tests also use the selected bundle and retain
all their required fixture/table checks.

All six registered CPU/Vulkan API, feature-batch, and motion-batch tests pass
with RTX 5070 Ti / driver 595.71.05 and the exact-scale bundle. Evidence is
`current-vulkan-{configure,parity-build,registered-parity}.log`; reproduction
commands are in `docs/API-SOMA.md`.

The current native-FP16 Vulkan turning trace maps all 490 semantic taps (two
additional taps are weight copies). Its first nonzero discrepancy is already
at root `_proj_local_pose`: two values differ by at most 0.0001220703125.
Differences then appear in start/end MLP projections and grow before attention.
`zombie-current-vulkan-split-layers.json` records the full comparison. This
localizes remaining work to neural arithmetic, rather than endpoint playback
or a missing layer correspondence; it does not establish Vulkan mixed parity.

### Vulkan projection accumulation isolation

Fresh `export_soma_matmul.py` fixtures from the verified walking, sitting, and
turning CUDA captures isolate `/mmm_net/_proj_local_pose/MatMul`, before bias.
Each contains identical upstream half operands and 2,048 reference outputs.
The native-FP16 Vulkan baseline differs at 6, 2, and 4 outputs respectively.
The calibrated CPU `aligned16` control matches every output in all three cases.
This establishes a matrix-accumulation discrepancy independently of encoding,
bias addition, and playback.

The matrix probe now has three explicit operand-layout diagnostics. Tail-padding
to a multiple of 16 leaves mismatch counts unchanged (6/2/4). Moving the short
remainder first gives 5/4/3; spacing each 16-product group into a zero-padded
32-element block gives 4/3/4. None reproduces all source projections, so none is
used by production inference. Fixtures, commands' device logs, numerical reports,
and controls are retained under `vulkan-residue-audit`.

Example (fresh process so Vulkan capability selection is explicit):

```sh
MOTIONBRICKS_SOMA_NATIVE_F16=1 \
  build/vulkan-profile/tests/motionbricks-soma-matmul-probe \
  generated/soma/export-scales generated/soma/vulkan-residue-audit/zombie.gguf \
  /tmp/soma-residue-probe.json vulkan backend-residue-first
```

Other layout arguments are `backend-pad-tail` and
`backend-residue-first-spaced`; ordinary `backend` is the control. These modes
preserve the real-valued dot product but intentionally change finite-precision
accumulation grouping. Vulkan mixed-FP16 parity remains unresolved, and the
pinned GGML submodule is unchanged.

### Vulkan short-prefix matrix trial

The validated CUDA accumulator consumes the FIRST `K % 16` inputs as its short
initial group, then groups of 16. The earlier tail-to-front diagnostics did not
represent that ordering. More importantly, GGML's Vulkan dispatch sends up to
eight input columns through its vector kernel (`mul_mat_vec_max_cols = 8`), so
advertised matrix-core support did not mean those eight-row probes used it.
Switching cooperative-matrix extensions therefore left their results unchanged.

Repeating captured input rows to force matrix dispatch and padding after the
short prefix gives exact results in all three projections (12,288 repeated
values). These repeated outputs are a dispatch diagnostic, not a new upstream
batch capture. The `backend-prefix-matrix` probe applies the same transformation
with zero-padded extra rows and trims them from its result. It matches all 6,144
original, un-repeated CUDA outputs. Three hardware CTests enforce zero mismatches:
`motionbricks-soma-vulkan-prefix-{walk,sit,zombie}`. All three pass, as does the
ordinary strict FP32 Vulkan motion-batch regression.

`MOTIONBRICKS_SOMA_VULKAN_PREFIX_TRIAL=1`, together with
`MOTIONBRICKS_SOMA_NATIVE_F16=1`, enables this only for the root local-pose
projection in the diagnostic runner. It requires Vulkan mixed-FP16 and records
`arithmetic.root_projection_policy=backend-f16-prefix-matrix-trial` in output and
layer provenance. Public API requests explicitly disable and restore the trial
flag. No GGML submodule changes are required.

With the existing split-root trial also enabled, both root-model executions in
each walking/sitting/turning step-000 capture now match the source local-pose
projection exactly. The first remaining differences occur in start/end MLP
layers. `vulkan-residue-audit/*-prefix-full-layers.json` records these captures.
The turning chain improves from 33 to 19 pose-token mismatches and from
67.1716 mm / 13.9084 degrees to 62.1029 mm / 10.9666 degrees across replay and
closed loop, but still fails every strict transition gate. Its report, failed
acceptance result, and demo are under `chain-zombie-vulkan-prefix-trial`.
All 22 reference-tool regression checks pass after adding the trial provenance.

### Vulkan start/end MLP matrix trial

All six start/end MLP MatMul nodes were exported independently for walking,
sitting, and turning. Across these 18 fixtures (36,864 outputs), ordinary
Vulkan execution has 70 mismatches while `backend-prefix-matrix` has zero.
Artifacts and device logs are under `vulkan-mlp-matrix-audit`; `summary.json`
records the per-product comparisons.

The existing `MOTIONBRICKS_SOMA_VULKAN_PREFIX_TRIAL=1` now applies the matrix
path to the root local-pose projection and all six start/end MLP layers. Its
provenance policy is `backend-f16-prefix-matrix-mlp-trial`, distinguishing it
from the earlier projection-only capture. With the trial enabled, all retained
start/end MLP outputs, including biases and activations, match upstream exactly
in both root executions across the three full step-000 captures. The earliest
remaining discrepancy is shared-transformer attention softmax (up to 1.19210e-7),
followed by normalization and downstream rounding effects.

Eighteen registered `motionbricks-soma-vulkan-mlp-*` zero-mismatch checks pass,
along with the ordinary strict FP32 Vulkan motion-batch test (19/19).
`registered-tests.log` records the run. No default/public precision path changes.

The turning chain remains a failure: `chain-zombie-vulkan-mlp-trial` has 35
pose-token mismatches versus 19 for the projection-only trial; maximum position
and rotation errors are 61.6254 mm and 10.7967 degrees. Thus exact earlier layers
do not imply improved discrete decisions or whole-model parity. Its report,
failed acceptance result, and demo export preserve this outcome explicitly.

### Vulkan softmax arithmetic isolation

`export_soma_softmax.py` now exports all 24 Softmax inputs/outputs from a verified
pinned CUDA source capture. It verifies the model/capture hashes, explicit axes,
FP32 types and the 93,312-value inventory. The turning export reproduces the
previous binary fixture exactly. `motionbricks-soma-backend-softmax-probe` runs
those inputs directly through backend softmax, recording FP32 and half-rounded
mismatches plus raw output arrays. This removes propagated model error from the
comparison.

On turning, standard CPU softmax has 10 half-rounded mismatches and standard
Vulkan has 13. Replacing the Vulkan shader's reciprocal multiplication with a
written division changes no output bits. CUDA-style fused exponential range
reduction reduces the Vulkan count to one. Adding an FMA residual correction to
the quotient removes that final half mismatch. The GLSL experiment uses GPU
`exp2` directly; it does not read the CPU calibration table.

Across walking, sitting, and turning, the combined experiment matches all
279,936 half-rounded outputs from 72 independent source operations. Remaining
FP32 differences are only subnormals (6/126/87 values respectively; maxima
1.524e-39 / 1.17549e-38 / 1.12532e-38), which round to zero in half precision.
The reports, shader variants, source fixture manifests and build logs are in
`backend-softmax-audit`; `three-capture-summary.json` records final measurements.

The experimental shader diff is retained as
`soma-vulkan-softmax-diagnostic.patch`, relative to the pinned `ggml` directory.
The original shader was restored and rebuilt after each experiment; the GGML
submodule remains clean. This patch is an isolated arithmetic experiment, not
yet an integrated runtime mode or an end-to-end Vulkan parity claim. Example
fixture and baseline probe commands:

```sh
.cache/soma-reference-venv/bin/python reference/export_soma_softmax.py \
  --capture generated/soma/chain-zombie-half-cpu-coverage/step-000/capture \
  --model generated/soma/reference/motionbricks/assets/onnx/soma/mmm_v6_soma_fp16.onnx \
  --output /tmp/soma-softmax-fixture
build/vulkan-profile/tests/motionbricks-soma-backend-softmax-probe \
  generated/soma/export-scales /tmp/soma-softmax-fixture/softmax-reference.bin \
  /tmp/soma-softmax-vulkan.json vulkan
```

### Full-model Vulkan softmax experiment and runtime provenance

Applied the isolated softmax patch temporarily and rebuilt the Vulkan runner,
with the verified root projection/MLP and split-root trials enabled. In walking,
sitting, and turning full step-000 captures, the first attention output now
matches through half rounding and the following output projection. Only tiny
subnormal softmax differences remain before the first LayerNormalization.
That normalization is now the first significant discrepancy, with FP32 output
errors around 4.76837e-7 to 7.15256e-7. Full layer reports and exact runtime
identities are under `vulkan-softmax-full-audit`.

The turning chain remains rejected: pose-token mismatches change from 35 to 34,
maximum joint-position error from 61.6254 mm to 59.3963 mm, and maximum rotation
error from 10.7967 to 10.7675 degrees. Its report, failed acceptance verdict,
and demo are in `chain-zombie-vulkan-softmax-trial`. This is evidence of a local
arithmetic correction, not accepted end-to-end Vulkan mixed-FP16 parity.
The original GGML shader was restored and rebuilt after the experiment; the
submodule is clean.

`run_soma_chain.py` accepts repeatable `--runtime-artifact PATH` arguments. It
records each file's resolved path and SHA256 before execution and verifies that
all remain unchanged through the run. This experiment records the diagnostic
patch, actual shader, and compiled `libggml-vulkan` used during capture. These
are capture-time identities: a later intentional backend restoration does not
rewrite them. Demo exports now preserve runner/bundle/runtime identities in
`native_provenance`. The new experiment's demo has all three runtime identities
and its verified failing verdict. All 22 reference-tool regression checks pass.

### Isolated Vulkan Welford normalization experiment

`reference/soma-vulkan-norm-diagnostic.patch` adds an explicitly diagnostic
negative-epsilon protocol to the GGML Vulkan normalization shader. The input is
flattened to `[rows, width]`, with scale and bias appended as two extra rows.
It uses the validated CUDA four-sample Welford update, warp and block reduction
order, corrected FP32 division, hardware inverse square root, and fused affine
output. It does not read reference outputs or calibration tables at inference.
The probe's `vulkan-packed` mode requires this experimental shader; it is not a
production GGML operation or a supported model precision mode.

On the RTX 5070 Ti, walking, sitting and turning source captures each passed all
48 LayerNormalization operations with **zero float-bit mismatches**, zero
half-rounded mismatches and zero maximum error: 1,892,352 values across 144
operations. The ordinary Vulkan norm/multiply/add expression had respectively
67, 89 and 66 half-rounded mismatches. Reports, baseline comparisons and the
patch checksum are in `generated/soma/vulkan-norm-audit/three-capture-summary.json`.
The walking fixture here is `walk-one-mixed-norm.gguf`; its capture identity is
recorded in the report and should not be confused with other walking fixtures.

To reproduce, export verified captures with `export_soma_norm.py`, apply the
patch relative to `ggml/`, rebuild `motionbricks-soma-norm-probe`, and run:

```sh
build/vulkan-profile/tests/motionbricks-soma-norm-probe \
  generated/soma/export-scales FIXTURE.gguf OUTPUT.json vulkan vulkan-packed
```

Omit the final argument for the stock norm/multiply/add baseline. Restore the
original shader and rebuild after the experiment. The measured run restored
both source and backend binary successfully; the GGML submodule remains clean.
This is isolated kernel evidence only. Full-model integration and strict chained
Vulkan mixed-FP16 acceptance remain outstanding, and previous failed chain
verdicts remain failed.

### Full-model Vulkan normalization trial

`MOTIONBRICKS_SOMA_VULKAN_NORM_TRIAL=1` routes SOMA mixed-FP16 layer
normalization through the packed Welford diagnostic operation above. It requires
the experimental norm shader and is rejected for CPU or FP32. Public API calls
disable and restore this diagnostic flag explicitly. Runner output and layer
traces record `vulkan-packed-welford-diagnostic` as their normalization policy.

The full-model experiment combined this flag with the softmax shader patch,
native half operands, prefix matrix/MLP trial, and split-root trial. All three
captures retained complete mapping of 490 semantic taps. The first meaningful
numerical difference moved past 155 compared taps to
`_root_token_transformer_model.layers.0.self_attn.in_proj_bias` in root execution
1. Its maximum errors were 0.00390625 (walking), 0.001953125 (sitting), and
0.001953125 (turning). Earlier differences were confined to subnormal softmax
values. This identifies the root-token QKV matrix accumulation order as the
next full-model issue; normalization now reproduces the independent CUDA output
when its inputs agree.

`generated/soma/vulkan-norm-full-audit` contains full traces, comparisons, build
logs and hashes of the exact temporary backend and shaders. The turning chain
and demo are in `generated/soma/chain-zombie-vulkan-norm-trial`. **Strict chain
acceptance still fails**: across replay and closed-loop comparisons there are
25 pose-token mismatches, maximum world-position error 0.0603748482 m, and maximum
rotation error 13.3263492 degrees. The preceding softmax-only trial had 34 token
mismatches, 0.0593962004 m and 10.7674935 degrees. Fewer local arithmetic errors
therefore have not yet produced lower worst-case motion errors. These failed
verdicts remain visible in the exported demo, with exact runtime provenance.
The original shader sources and backend were restored and rebuilt afterward.

After restoring the shaders, all seven focused public API and batch parity tests
passed on the current Vulkan build (CPU and Vulkan API, motion batches, feature
batches, and the existing SOMA batch test). Logs are
`vulkan-norm-full-audit/regression-{build,tests}.log`.

### Vulkan root-token QKV and feed-forward accumulation groups

`precision_backend_nvjet3` gathers successive 16-wide K blocks into the three
persistent streams identified in the CUDA NVJET kernel. Stream assignment is
`((offset % 128) / 16) % 3`; each stream runs as an F16 Vulkan matrix product with
FP32 accumulation, then the three outputs are added in FP32 order. Only the
combined result is rounded to half. This reproduces CUDA's arithmetic without
using captured outputs or the CPU dot-product emulator during inference.

Independent walking, sitting and turning captures each isolate root-token layer
0 QKV (38,400 values) and linear1 (51,200 values). All six grouped Vulkan results
match exactly: **268,800 values, zero mismatches**. The ordinary Vulkan products
have respectively 44/64, 54/65 and 43/59 QKV/linear1 mismatches (329 total).
Fixtures, baseline reports and exact reports are under
`generated/soma/vulkan-nvjet3-audit`. Six opt-in CTests named
`motionbricks-soma-vulkan-nvjet3-*` enforce zero maximum error and mismatches.
The standalone probe mode is `backend-nvjet3`.

`MOTIONBRICKS_SOMA_VULKAN_NVJET3_TRIAL=1` enables this calculation for QKV and
linear1 in the root-token transformer, and requires Vulkan mixed-FP16 with
native half operands. Public API arithmetic scopes disable and restore the
flag. Full traces and chain reports record `backend-f16-nvjet3-trial` as the
NVJET3 policy.

The combined full-model trial also uses the norm/softmax diagnostic shaders and
previous prefix/split trials. All 490 semantic taps map in all three captures.
The first meaningful discrepancy moves to root-token layer 0 attention output
projection: walking 11 unequal values (max 0.0009765625), sitting 20 (max
0.000244140625), turning 24 (max 0.00048828125). The preceding QKV, score and
merged-attention boundaries now agree; softmax retains only subnormal residuals.
Full traces and exact temporary runtime hashes are in
`generated/soma/vulkan-nvjet3-full-audit`.

The chain in `generated/soma/chain-zombie-vulkan-nvjet3-trial` still **fails**
strict parity: 29 pose-token mismatches across replay and closed-loop, maximum
world-position error 0.0557946626 m and rotation error 15.1624270 degrees. Its
demo retains the failed verdict and provenance. Shader sources and backend
binary were restored and rebuilt afterward. These measurements support the
isolated arithmetic change, not a claim of full Vulkan mixed-FP16 parity.

Validation after restoration: all six new isolated Vulkan tests and four public
CPU/Vulkan API/motion-batch regressions passed (10/10); all 22 SOMA reference-tool
unit tests passed. Logs are retained alongside the isolated audit reports.

### Vulkan attention output projection: eight accumulation streams

The grouped matrix helper is now named `precision_backend_nvjet` and accepts
three or eight streams. For root-token attention output projection it gathers
16-wide K blocks into eight streams, sums their FP32 products in order, adds the
half-representable bias in FP32, and rounds the final output to half. This keeps
the fused Gemm contract; rounding before bias would change the result.

All three independently exported layer-0 output-projection fixtures match
exactly: **38,400 values, zero mismatches**, compared with 18 walking, 20 sitting
and 24 turning mismatches for ordinary Vulkan matrix accumulation plus fused
bias. `generated/soma/vulkan-nvjet8-audit` holds fixtures and reports. Probe modes
are `backend-nvjet8-bias` and `backend-bias`; three opt-in CTests named
`motionbricks-soma-vulkan-nvjet8-*` require zero error.

The full-model opt-in flag is `MOTIONBRICKS_SOMA_VULKAN_NVJET8_TRIAL=1`, requiring
Vulkan mixed-FP16 and native half operands. Public API calls disable and restore
it. Runner metadata and comparison/chain reports record
`backend-f16-nvjet8-trial` as the NVJET8 policy. With the preceding verified
trials, all 490 semantic taps remain mapped in each capture. The first meaningful
mismatch is now root-token layer 0 `linear2`: walking 27 values (max
0.000244140625), sitting 46 (max 0.0009765625), turning 40 (max 0.0009765625).
See `generated/soma/vulkan-nvjet8-full-audit` for full traces and temporary backend
and shader hashes.

The turning chain `generated/soma/chain-zombie-vulkan-nvjet8-trial` still fails
strict acceptance: 12 pose-token mismatches across replay and closed-loop,
maximum world-position error 0.0615592703 m and rotation error 14.0646072 degrees.
Its demo preserves the failed verdict. The next arithmetic boundary to resolve
is the CUDA four-way split reduction in `linear2`. Original shaders and backend
binary were restored and rebuilt after the trial.

After restoration, all nine isolated three/eight-stream Vulkan tests and four
CPU/Vulkan public API/motion-batch checks passed (13/13), as did all 22 SOMA
reference-tool tests. Validation logs are retained in the NVJET8 audit directory.

### Vulkan four-way root-token and conditioning reduction

`precision_backend_nvjet_split4` gathers interleaved 128-wide tiles into four
partitions, runs three persistent accumulation streams per partition, then sums
the four FP32 results in order. Half rounding occurs only after the combined
product, before the separate bias operation. The independently exported layer-0
`linear2` and root `external_cond_blocks.2` products match CUDA exactly for
walking, sitting and turning: **87,552 values, zero mismatches**, compared with
379 ordinary Vulkan product mismatches. Fixtures, baseline results and grouped
results are in `generated/soma/vulkan-nvjet-split4-audit`; six opt-in CTests named
`motionbricks-soma-vulkan-nvjet-split4-*` enforce zero error. The standalone probe
mode is `backend-nvjet-split4`.

`MOTIONBRICKS_SOMA_VULKAN_NVJET_SPLIT4_TRIAL=1` selects these root projections in
the full-model Vulkan mixed-FP16 trial. It requires native half operands; public
API scopes disable and restore it. Runner traces and comparison/chain reports
record the `backend-f16-nvjet_split4-trial` policy.

The expanded graph exposed the fixed 4,096-node capacity in the single-request
root planner. It now uses the same tensor-count capacity function as the batched
planner. The failed pre-fix log is retained as
`vulkan-nvjet-split4-full-audit/walk-capacity-failure.log`. The successful rerun
retains all 490 mapped semantic taps in each capture. The first meaningful
numerical difference moves beyond the root-token transformer to the initial
root decoder convolution (`_conv_output.model.0`): walking one unequal value
(max 0.00000762939453125), sitting six (max 0.001953125), turning six (max
0.00390625). Earlier softmax differences remain confined to subnormal values.
Full traces and exact temporary runtime hashes are in
`generated/soma/vulkan-nvjet-split4-full-audit`.

The turning chain `generated/soma/chain-zombie-vulkan-nvjet-split4-trial` still
fails strict parity: 25 pose-token mismatches across replay and closed-loop,
maximum world-position error 0.0698185076 m and rotation error 10.6366186 degrees.
The exported demo retains the failed verdict. Original shaders and backend were
restored and rebuilt after the run. Root decoder convolution arithmetic is the
next full-model boundary to resolve.

Validation after restoration: all 15 Vulkan accumulation tests and four public
CPU/Vulkan API/motion-batch checks passed (19/19), all 22 SOMA reference-tool tests
passed, and the rebuilt G1 root parity test passed. Logs are retained in the
split-four audit directory.

### Vulkan root decoder convolution traversal

`precision_backend_conv` implements the profiled channel/spatial traversal with
Vulkan matrix products. Three-position kernels reorder channel tiles (64
channels for the root decoder); dilations of nine or greater use eight FP32
partial sums, while pointwise convolutions use eight persistent accumulation
streams. The convolution output is rounded to half before adding its bias.
No reference outputs or CPU inference callbacks are used in this calculation.

Independent root initial, split-dilation, spatial and pointwise convolution
fixtures for walking, sitting and turning all match exactly: **98,304 values,
zero mismatches**, compared with 89 ordinary Vulkan mismatches. The fixtures and
reports are in `generated/soma/vulkan-conv-audit`. Probe modes are `backend-conv`
and `backend-conv-plain`; twelve opt-in CTests named
`motionbricks-soma-vulkan-conv-*` require zero error.

`MOTIONBRICKS_SOMA_VULKAN_ROOT_CONV_TRIAL=1` enables these root decoder families
in Vulkan mixed-FP16 with native half operands. It is explicitly disabled and
restored by public API scopes, and traces/reports record the
`backend-f16-root_conv-trial` policy. The helper also supports the independently
profiled 160-channel final-decoder family, but this flag changes only root decoder
convolutions; final-decoder integration remains separate work.

Full-model reports in `generated/soma/vulkan-root-conv-full-audit` preserve
complete mapping of all 490 semantic taps. Sitting and turning now agree through
the root decoder, with the next meaningful mismatches in pose-attention scores
(layer 0 for sitting and layer 1 for turning). Walking first differs in root
`external_cond_blocks.0`: six values, maximum 0.001953125. That remaining
conditioning mismatch requires further isolation; convolution agreement alone
does not establish the whole walking root path.

The turning chain in `generated/soma/chain-zombie-vulkan-root-conv-trial` still
fails strict parity: nine pose-token mismatches across replay and closed-loop,
maximum world-position error 0.0311032866 m and rotation error 7.1594687 degrees.
The demo preserves these errors and the failed verdict. Exact temporary backend
and shader hashes are retained; original shader sources and backend binary were
restored and rebuilt afterward.

Validation after restoration: all 27 isolated Vulkan matrix/convolution checks
and four CPU/Vulkan public API/motion-batch checks passed (31/31), along with all
22 SOMA reference-tool tests. Logs are retained in the convolution audit directory.

### Vulkan pose-attention score accumulation

The grouped Vulkan matrix helper now supports K=64 with four active 16-wide
streams from CUDA's eight-accumulator score kernel. Each partial remains FP32;
the sum is rounded to half at the score-product boundary. Query and key scaling
retain the pinned mixed-FP16 contract. All 16 pose-transformer layers were
exported independently for walking, sitting and turning: **48 operations,
196,608 values, zero mismatches**, versus 52 ordinary Vulkan mismatches. Fixtures,
baseline and grouped results are under `generated/soma/vulkan-pose-scores-audit`.
The standalone probe mode is `backend-nvjet-scores`; 48 opt-in CTests named
`motionbricks-soma-vulkan-pose-scores-*` enforce zero error.

`MOTIONBRICKS_SOMA_VULKAN_POSE_SCORES_TRIAL=1` selects this arithmetic for
64-wide attention heads in Vulkan mixed-FP16 with native half operands. Public
API scopes disable and restore it; traces and reports record the
`backend-f16-pose_scores-trial` policy. The matrix helper still supports the
previous three/eight-stream root projections.

The full-model score trial exposed the pose planner's fixed 2,048-node graph
capacity. Single and batched pose graphs now size capacity from constructed
tensors plus external-weight headroom; the pre-fix failure is retained as
`vulkan-pose-scores-full-audit/walk-capacity-failure.log`. After this repair, all
490 semantic taps map in all three captures. Sitting and turning agree through
the pose transformer (apart from subnormal softmax residuals), then first differ
in the final decoder's initial convolution: sitting one value, maximum
0.0001220703125; turning four values, maximum 0.00390625. Walking still first
differs in root `external_cond_blocks.0` (six values, maximum 0.001953125), before
the pose model. Traces and exact temporary runtime hashes are retained in
`generated/soma/vulkan-pose-scores-full-audit`.

The turning chain `generated/soma/chain-zombie-vulkan-pose-scores-trial` still
fails strict acceptance: ten pose-token mismatches across replay and closed-loop,
maximum world-position error 0.0315757934 m and rotation error 7.1594687 degrees.
Its demo retains the failed verdict and provenance. Original shaders and backend
binary were restored and rebuilt afterward. Remaining work includes final
decoder arithmetic and walking's root-conditioning mismatch; isolated score
agreement is not full-model acceptance.

Validation after restoration: all 75 isolated Vulkan arithmetic checks and four
CPU/Vulkan public API/motion-batch checks passed (79/79), all 22 SOMA reference-tool
tests passed, and the rebuilt G1 pose parity test passed. Logs are retained in
the pose-score audit directory.

### Vulkan final decoder convolution trial

The same profiled Vulkan convolution helper now covers final-decoder fixtures:
160-channel initial convolution, split-dilation, spatial and pointwise families.
All twelve independent walking/sitting/turning fixtures match exactly:
**98,304 values, zero mismatches**, versus 120 ordinary Vulkan mismatches.
Artifacts are in `generated/soma/vulkan-decoder-conv-audit`; twelve opt-in CTests
named `motionbricks-soma-vulkan-decoder-conv-*` enforce zero error.

`MOTIONBRICKS_SOMA_VULKAN_DECODER_CONV_TRIAL=1` enables these operations in the
final decoder for Vulkan mixed-FP16 with native half operands. Public API scopes
disable and restore the flag; traces and reports record the
`backend-f16-decoder_conv-trial` policy. Both decoder graph paths now size node
capacity from constructed tensors plus external-weight headroom, consistent
with the expanded root and pose planners, instead of fixed/estimated node counts.

Full traces in `generated/soma/vulkan-decoder-conv-full-audit` retain complete
mapping of all 490 semantic taps. Sitting and turning now first differ in
`decoder.external_cond_blocks.0`: one value each, maximum 0.000244140625 and
0.000030517578125 respectively. Each has 36 taps with non-subnormal differences,
propagating from decoder-conditioning arithmetic. Walking still first differs
in root `external_cond_blocks.0` before pose generation. The initial final-decoder
convolution agrees wherever its upstream inputs agree.

The turning chain `generated/soma/chain-zombie-vulkan-decoder-conv-trial` still
fails strict parity: 14 pose-token mismatches across replay and closed-loop,
maximum world-position error 0.0680391550 m and rotation error 11.2874675 degrees.
Its demo retains that failed verdict and runtime provenance. Exact temporary
backend and shader hashes are retained in the full audit; original shaders and
backend binary were restored and rebuilt afterward. Conditioning projections
remain the next arithmetic issue to isolate.

Validation after restoration: all 87 isolated Vulkan arithmetic checks and four
CPU/Vulkan public API/motion-batch checks passed (91/91), all 22 SOMA reference-tool
tests passed, and the rebuilt G1 decoder parity test passed. Logs are retained
in the final-decoder convolution audit directory.

### Vulkan conditioning projections and native row-count dispatch

The final decoder's external conditioning matrices have inner widths 520 and
516. Their CUDA kernels process the short K remainder first; applying the
existing prefix-aware matrix helper matches all six independent conditioning
fixtures (73,728 values, zero mismatches versus 22 ordinary Vulkan mismatches).

Walking's root `external_cond_blocks.0` already matches CUDA at the captured
16-row shape, but native walking uses eight rows and Vulkan selected its vector
kernel. An eight-row prefix of the verified capture reproduces the six full-model
mismatches (maximum 0.001953125). Forcing each split-eight partial through the
matrix helper removes all six. `export_soma_matmul.py --take-rows 8` constructs
this diagnostic prefix and records `fixture.take_rows`; it is explicitly a
shape control derived from a capture, not an independent upstream batch capture.
The probe mode for this control is `backend-split8-matrix`.

`MOTIONBRICKS_SOMA_VULKAN_CONDITIONING_TRIAL=1` enables prefix-aware final-decoder
external conditioning and forces the root split-eight products through matrix
kernels. It requires Vulkan mixed-FP16 and native half operands; it accompanies
the split-root trial for root projections. Public API scopes disable and restore
it, and traces/reports record `backend-f16-conditioning-trial`. Ten opt-in CTests
named `motionbricks-soma-vulkan-conditioning-*` cover the six decoder products,
three full-row root products and eight-row walking control. Artifacts are in
`generated/soma/vulkan-conditioning-audit`.

Full-model traces in `generated/soma/vulkan-conditioning-full-audit` retain all
490 mapped semantic taps. Sitting and turning step-0 captures have **zero
non-subnormal layer differences**. Walking now first differs at pose
`_proj_input.0` (16 values, maximum 0.0078125), after its root path agrees.

The turning chain `generated/soma/chain-zombie-vulkan-conditioning-trial` passes
strict step-0 replay and closed-loop acceptance, and step-2 replay acceptance.
Step 1 replay/closed-loop and step 2 closed-loop still fail. Across the chain,
there are two pose-token mismatches, maximum world-position error 0.0524648187 m
and rotation error 6.2977478 degrees. The whole-chain verdict remains failed and
is preserved in the demo. Exact temporary runtime hashes are retained; original
shaders and backend were restored and rebuilt afterward. Remaining failures
require further native-shape investigation, not relaxed tolerances.

The failed turning step-1 replay was compared separately against its own source
capture. Its first meaningful difference is final-decoder `target_cond_blocks.2`
(one value, maximum 0.00048828125), before external conditioning. See
`vulkan-conditioning-full-audit/zombie-step1-layers.json`; this provides a concrete
next projection to isolate. Validation after shader restoration passed all 97
isolated Vulkan arithmetic checks plus four public CPU/Vulkan API/motion-batch
checks (101/101), and all 22 SOMA reference-tool tests.

### Vulkan decoder target and short pose-input projections

The conditioning trial now also applies prefix-aware matrix arithmetic to final
decoder target conditioning and pose `_proj_input.0`. Thirteen independent or
explicitly derived shape fixtures cover full/eight-row pose inputs, both decoder
target projections for all three motions, and the previously failing turning
step-1 target projection: **163,840 values, zero mismatches**, versus 77 ordinary
Vulkan mismatches. Eight-row pose controls reproduce the vector-kernel errors;
forcing the matrix path removes them. Fixtures and comparisons are in
`generated/soma/vulkan-target-audit`, with thirteen exact-error CTests named
`motionbricks-soma-vulkan-target-*`. The existing conditioning flag now records
`backend-f16-conditioning-target-pose-trial` to distinguish this expanded policy.

Full traces retain all 490 mapped semantic taps. Sitting and turning have no
non-subnormal layer differences. Walking now first differs in the first pose
attention QKV projection (`_transformer_model.layers.0.self_attn.in_proj_bias`),
after its pose input projection agrees. This is another short native-sequence
kernel-dispatch issue to isolate.

All three chains were rerun with unchanged strict tolerances:

| Chain | Passed comparisons | Pose-token mismatches | Max world position | Max rotation | Max contact error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sitting | 6/6 | 0 | 0 m | 0.0000034151 deg | 0 |
| Turning | 6/6 | 0 | 0 m | 0.0000038182 deg | 0 |
| Walking | 3/6 | 2 | 0.0027104026 m | 0.64800614 deg | 0.022674680 |

Sitting and turning each pass all three steps in both independent replay and
native closed-loop feedback. Their tiny angular metrics are the known quaternion
metric roundoff. Walking remains failed. Reports and demos are
`generated/soma/chain-{walk,sit,zombie}-vulkan-target-trial`; the summary and exact
temporary backend/shader hashes are in `generated/soma/vulkan-target-full-audit`.
Original shaders and backend were restored and rebuilt afterward. These are
opt-in diagnostic results, not yet a production Vulkan calibrated API mode.

After restoration, all 110 isolated Vulkan arithmetic checks and four public
CPU/Vulkan API/motion-batch checks passed (114/114); all 22 SOMA reference-tool
tests passed. Logs are retained in the target-projection audit directory.

### Short pose-transformer matrix dispatch: all captured chains pass

The conditioning trial now forces matrix arithmetic for pose-transformer linear
projections with at most eight native rows, preserving the fused attention-output
bias contract. Twelve eight-row controls cover QKV, attention output, linear1
and linear2 for all three motions: **172,032 values, zero mismatches**, compared
with 2,706 vector-kernel mismatches. The exporter gained `--flatten-rows` for the
captured singleton batch axis (only with rank-two weights), then `--take-rows 8`;
metadata explicitly marks both shape transformations. These are controls derived
from independent captures, not new upstream batch captures. Probe mode
`backend-prefix-bias` preserves fused output bias. Twelve CTests named
`motionbricks-soma-vulkan-pose-matrix-*` require zero error.

Runner policy is now `backend-f16-conditioning-target-pose-matrix-trial` under
the existing conditioning flag. Full traces retain 490 mapped semantic taps.
Sitting and turning have no non-subnormal layer differences. Walking retains two
pose-output-logit differences (maximum 0.00390625), without changing decisions.

**All 18 strict chain comparisons pass**: three steps each for walking, sitting
and turning, both replay and independent closed-loop feedback. Duration and pose
tokens agree exactly; maximum world-position and contact errors are zero. Maximum
rotation metrics are 0.0000034151 degrees for walking/sitting and 0.0000038182 for
turning (the known quaternion metric roundoff). Reports and demos are in
`generated/soma/chain-{walk,sit,zombie}-vulkan-pose-matrix-trial`; full traces and
exact temporary runtime hashes are in `generated/soma/vulkan-pose-matrix-full-audit`.
Original shader sources and backend were restored and rebuilt afterward.

This establishes captured single-request chained parity for the opt-in trial.
It does **not** yet establish calibrated Vulkan batch parity or a production
public Vulkan mode. Temporary shader integration and diagnostic arithmetic
selection still need production integration, followed by the remaining scope
checks. Tolerances were not relaxed to obtain these passing results.

Post-restoration validation passed all 122 isolated Vulkan arithmetic checks and
four public CPU/Vulkan API/motion-batch checks (126/126), plus all 22 SOMA
reference-tool tests. The comparison HTTP and headless Chromium tests passed
using the newly passing walking demo. Its screenshot,
`generated/soma/vulkan-walk-passing-demo.png`, was visually inspected: overlay,
passing chain verdict, replay controls and zero position-error display are present.
Logs are retained in the pose-matrix audit directory.

### Vulkan mixed-FP16 batch validation

The complete passing Vulkan arithmetic trial was exercised through the existing
batched C++ paths, with both request orders. Four independent raw-feature fixtures
(`walk-one`, `walk-four16`, `sit-short`, `zombie`) produced eight comparisons:
exact duration/token decisions, zero batch-versus-single feature error, and zero
differences at every recorded batch/single processing boundary. Maximum raw-feature
error against upstream was 8.9407e-8. This uses captured inputs, so native skeleton
encoding is tested separately by the full-motion batch checks.

Full-motion batches combined walking, sitting and turning at each of their three
captured chain steps, in both orders: **18 comparisons, zero reported world or
corrected position/rotation error, zero contact error, exact duration and native
single/batch pose decisions**. Invalid requests also preserved the test's no-partial-
output guarantee. The input fixtures are independent per-request upstream
captures; this proves native batch agreement with those references, not a claim
that the upstream ONNX graph executed the same batch layout.

`generated/soma/vulkan-mixed-batch-audit/validation.json` records the checked
results, all 13 fixture checksums, temporary backend/shader and executable hashes,
and reproduction-script checksum. `run-batches.py` alongside it retains the exact
experiment, including arithmetic environment and restoration in `finally`.
`features.json` contains all feature and boundary comparisons;
`motion-step{0,1,2}.log` contains the strict full-motion checks. The GGML shader
sources and backend binaries were restored and rebuilt successfully afterward.

This closes the captured **internal** Vulkan mixed-FP16 single/batch numerical
validation scope for the trial. Production shader integration, explicit public
API precision selection, and public-API validation still remain; the experimental
public precision mode does not silently inherit these diagnostic flags.

### Reproducible, operation-scoped SOMA Vulkan shader build

`MOTIONBRICKS_ENABLE_SOMA_VULKAN_SHADERS=ON` replaces temporary checkout patches
with a build-only GGML source overlay. `cmake/SomaVulkan.cmake` verifies the pinned
original norm/softmax shader hashes, copies dependency sources under the build
directory with configure dependencies, and substitutes the versioned shaders in
`src/vulkan/`. Deleted source files are removed from the overlay on reconfigure.
The dependency checkout stays unchanged; build metadata retains the original
GGML commit with a `soma-shaders` suffix. Incremental rebuilds do not recompile
the backend when inputs are unchanged (the upstream shader-generator external
target still runs its no-op build/install steps).

Positive-epsilon normalization retains the original GGML operation. The SOMA
packed affine operation uses negative epsilon. Softmax now selects the calibrated
exponent/division path only when its operation has `max_bias == -1`; ordinary
operations retain the original math and ALiBi behavior. The host wrapper selects
this marker under `MOTIONBRICKS_SOMA_VULKAN_SOFTMAX_TRIAL=1`, requiring the build
feature. Public API scopes disable this diagnostic flag. See
`src/vulkan/README.md` for the internal protocols.

The clean build at `build/vulkan-soma` passes all **18 strict chained comparisons**
without patching or restoring sources. Its sitting/walking/turning reports and
demos are `generated/soma/chain-{walk,sit,zombie}-vulkan-scoped-trial`; full traces,
semantic policy metadata and runtime hashes are in `vulkan-scoped-full-audit`.
Position and contact errors remain zero and decisions agree exactly.

The same scoped build passes all **18 full-motion batch comparisons** with zero
reported error and all eight raw-feature comparisons with zero batch/single
boundary differences (maximum upstream raw-feature error 8.9407e-8). Reports are
in `generated/soma/vulkan-scoped-batch-audit`. Ordinary CPU/Vulkan FP32 API and
motion-batch tests pass 4/4 with these shaders compiled in. Ordinary CPU G1 root,
G1 pose and SOMA API tests also pass 3/3 after the host softmax refactor. Build and
regression logs are in `generated/soma/vulkan-scoped-build-audit`.

This completes reproducible shader integration for the trial. It does not yet
provide a public calibrated Vulkan precision mode or hardware qualification at
model load; those are the next integration requirements. The existing public
experimental mode retains its previous contract.

Configuration guards were also exercised: enabling the feature without Vulkan
and supplying a shader source with an incompatible pinned hash both fail with
the intended errors. `vulkan-scoped-build-audit/build-guards.json` records these
checks. The pinned GGML checkout remained clean throughout this integration.


### FP32 and mixed-FP16 Vulkan capability coexistence

Enabling native-half capabilities on the unmodified scoped backend made the
FP32 three-motion batch fail with `batch decisions differ`. GGML's cooperative
matrix-2 path stages FP32 matrices as half even when the model requests FP32.
Disabling those capabilities process-wide protects FP32 but prevents a public
mixed-FP16 mode from coexisting with it.

The pinned backend overlay now also instantiates the upstream scalar FP32
matrix pipelines, selects their scalar tile policy for FP32 operands, and skips
the automatic half conversion for contiguous FP32 × FP32 inputs. Explicit half
operands keep the previously validated cooperative path. Non-contiguous staging
remains unchanged. This is a SOMA regression result, not a claim about every
possible GGML operation or graph.

The initial isolated probe is recorded in
`generated/soma/vulkan-f32-coexist-probe/summary.json`, with its transformation,
commands, logs and mixed runtime hashes alongside it. With native-half enabled,
FP32 walking/target/height-stress batches passed in both orders: maximum position
error `6.78474e-6 m`, rotation `0.00324619 degrees`, contacts `5.51045e-5`.
All 18 mixed full-motion batch comparisons retained zero final motion/contact
error, and eight raw-feature batch comparisons retained exact batch/single
boundaries and decisions (maximum upstream feature error `8.9407e-8`).
The same-process public API mode and hardware/driver qualification remain to be
implemented and validated; these probe runs use separate processes.

The permanent CMake overlay was rebuilt and independently rerun. Its batch
results are in `generated/soma/vulkan-f32-coexist-audit/summary.json` and match
the probe. All 18 replay/closed-loop comparisons also pass in
`chain-{walk,sit,zombie}-vulkan-coexist-trial/acceptance.json`; corresponding
`demo.json` files carry the error overlay. The full trace audit in
`vulkan-coexist-full-audit/summary.json` confirms 490 semantic taps for each
motion. Default public Vulkan FP32 API and full-motion batch CTests both pass
(2/2). A deliberately unvalidated backend source is rejected by the new hash
guard. The GGML dependency checkout remains unmodified.


### Loaded Vulkan backend qualification

The scoped backend now exposes a private versioned registry query, and
`neural_soma_vulkan_sm120` checks the actual loaded backend/device before a
future public precision mode can select its arithmetic. The qualified tuple is
protocol 1, PCI `10de:2c05`, NVIDIA driver version `0x94d1c140` (595.71.05),
Vulkan driver ID 4, subgroup size 32, enabled FP16 and enabled NV cooperative
matrix 2. Other GPUs/drivers are not claimed to reproduce the captured CUDA
profile. Runner output records `backend.soma_sm120_qualified` and a rejection
reason independently of the requested inference precision.

`generated/soma/vulkan-capability-audit/summary.json` records actual integration
checks: the qualified loaded backend is accepted; disabled capabilities, CPU
models, and the available AMD integrated Vulkan GPU are rejected; the stock Vulkan backend has no overlay procedure. Null
arguments and an incorrect result count are also rejected. Runtime library and
runner hashes accompany the report and saved diagnostic scripts. These checks
do not yet establish public precision switching: default public model loading
still disables half capabilities and public Vulkan integration remains pending.


### Public qualified Vulkan mixed-FP16 mode

`MB_PRECISION_MIXED_FP16_VULKAN_SM120` now selects the validated native Vulkan
arithmetic without diagnostic environment flags or CPU calibration tables.
A pre-initialization overlay query identifies the validated PCI device/driver
without creating a logical device. Default loading retains half capabilities
only for that qualified overlay/device (or an explicit diagnostic override).
Ordinary backends and unqualified devices retain scalar FP32 initialization;
explicit user disable flags are respected. The generation-time check still
requires the actual enabled capabilities, driver ID, subgroup size and protocol.
CPU-first model loading no longer disables a later qualified Vulkan model.

`generated/soma/public-vulkan-sm120-chain-audit/{walk,sit,zombie}/report.json`
records all 18 passing public replay/closed-loop comparisons. Each run cleared
all diagnostic and Vulkan environment flags and loaded a CPU model before the
Vulkan model. Every mixed request was bracketed by FP32 requests whose entire
public output arrays matched exactly. Shared library and backend binary hashes
were checked before and after each run. Public pose tokens remain unexposed;
the existing internal chain captures separately verify those decisions.

The same audit directory's `rejections.json` verifies CPU, AMD Vulkan, and
explicitly disabled capabilities reject mode 3 while preserving subsequent FP32
outputs. `g1-vulkan.json` records passing G1 root, pose and decoder parity checks
with the new default qualified initialization. The initial attempt used an old
chain manifest missing capture hashes and failed before inference; the successful
checks use the newer provenance-complete manifests, still referencing the same
independent pinned source captures.


### Demo and source-asset completion audit

The current public Vulkan outputs exactly match the passing coexistence demo
captures: 72 comparisons, 1,111,048 values across raw features, corrected/world
transforms and contacts. Evidence is
`generated/soma/public-vulkan-sm120-chain-audit/demo-api-equivalence.json`.
The same directory contains a freshly rendered and visually inspected `demo.png`;
Chromium's comparison HTTP and headless interaction tests pass. The demo guide
now describes current passing captures and reproduction, while the numerical
investigation history remains here.

All eight pinned checkpoint/source assets were rehashed and their byte lengths
verified against `soma-reference.json` (`source-assets.json` in the audit
directory). All 22 SOMA reference-tool tests pass. The ongoing requirement audit
is recorded in `completion-audit.json`; it is explicitly incomplete until the
remaining final-build precision/backend and boundary coverage checks finish.


### Completed final-build audit

The final shared library passed 54 additional public chain comparisons: FP32 on
CPU and Vulkan, and calibrated mixed-FP16 on CPU, each across walking, sitting
and zombie motion in replay and independent feedback. Together with the 18
public Vulkan mixed-FP16 comparisons, this is 72 passing comparisons across
all four supported precision/backend profiles. Runtime binary hashes remained
unchanged during validation. Reports and the saved command driver are under
`generated/soma/final-public-matrix`.

The current build also passes 19 encoding/playback checks and a calibrated CPU
three-motion batch in both orders with zero motion/contact error. The final
Vulkan batch audit passes all 18 motion comparisons and eight raw-feature
comparisons, with exact batch/single boundaries and decisions; maximum reference
feature error is `8.9407e-8`. See `generated/soma/final-vulkan-batch-audit/summary.json`.

The requirement-by-requirement completion record is
`generated/soma/public-vulkan-sm120-chain-audit/completion-audit.json`. Its scope
is the pinned captures and documented stress fixtures, with native Vulkan mixed
precision qualified only for RTX 5070 Ti / NVIDIA Linux driver 595.71.05. CPU
mixed precision is calibrated arithmetic emulation. The upstream export is
mixed-FP16, not pure FP16 arithmetic. Public calls are single-request; internal
batching is independently checked. Chained playback does not include the
upstream asynchronous scheduler, foot locking or IK. No broader motion, hardware,
driver or source-revision parity is claimed.

Current-build G1 CPU root, pose and decoder checks also pass (3/3), alongside
the previously verified current Vulkan counterparts. Evidence is
`generated/soma/public-vulkan-sm120-chain-audit/g1-cpu-final.json`.

## SOMA interactive controller parity

Use the pinned standalone checkout and reference venv (including the upstream
app's headless dependencies). The capture imports the actual app runtime; it
does not load the native library. Model checkpoints are verified by the upstream
exporter's checkpoint loader. Assets and capture arrays carry SHA-256 hashes.

```sh
env -u PYTHONPATH .cache/soma-reference-venv/bin/python reference/capture_soma_controller.py \
  --upstream /tmp/motionbricks-compare/new \
  --checkpoints generated/soma/reference/soma \
  --output generated/soma/controller-reference-001

env -u PYTHONPATH .cache/soma-reference-venv/bin/python reference/check_soma_controller_inference.py \
  --capture generated/soma/controller-reference-001 \
  --library build/debug/libmotionbricks.so --bundle generated/soma/export-scales \
  --backend vulkan --output generated/soma/controller-reference-001/native-inference.json

nix develop --command env \
  MOTIONBRICKS_CONTROLLER_CAPTURE="$PWD/generated/soma/controller-reference-001" \
  MOTIONBRICKS_CONTROLLER_REPORT="$PWD/generated/soma/controller-reference-001/oracle-controller.json" \
  go test -C demo -run '^TestUpstreamController$' -count=1 -v

nix develop --command env \
  MOTIONBRICKS_CONTROLLER_CAPTURE="$PWD/generated/soma/controller-reference-001" \
  MOTIONBRICKS_CONTROLLER_NATIVE=1 \
  MOTIONBRICKS_LIB="$PWD/build/debug/libmotionbricks.so" \
  MOTIONBRICKS_SOMA_MODEL="$PWD/generated/soma/export-scales" \
  MOTIONBRICKS_CONTROLLER_REPORT="$PWD/generated/soma/controller-reference-001/native-controller.json" \
  go test -C demo -run '^TestUpstreamController$' -count=1 -v
```

Use a fresh capture directory. Deployment consumes `controller-assets.json` and
its adjacent binary target/idle arrays; it needs neither Python nor the reference
NPZ. Select this asset file with `-soma-assets`. The two Go checks isolate the
controller and then exercise native feedback; the Python check independently
replays all exact upstream generation inputs. See
[streaming validation](../docs/STREAMING.md#independent-soma-controller-capture)
for scope, measurements, and the distinct interactive acceptance contract.

### Interactive mixed-FP16 validation (2026-11-19)

The same capture script accepts `--onnx-fp16` with the official pinned graph.
It verifies the graph checksum, disables ONNX Runtime graph optimizations, and
requires the CUDA execution provider. The deployed demo remains FP32. The GPU
reference venv supplies ONNX Runtime/CUDA; the CPU reference venv supplies the
upstream app and Warp dependencies:

```sh
nix develop --command env -u PYTHONPATH .cache/soma-gpu-reference-venv/bin/python \
  -c 'import site,runpy;site.addsitedir(".cache/soma-reference-venv/lib/python3.12/site-packages");runpy.run_path("reference/capture_soma_controller.py",run_name="__main__")' \
  --upstream /tmp/motionbricks-compare/new \
  --checkpoints generated/soma/reference/soma \
  --onnx-fp16 generated/soma/reference/motionbricks/assets/onnx/soma/mmm_v6_soma_fp16.onnx \
  --output generated/soma/controller-reference-fp16-new
```

Run the inference and Go controller checks above with the FP16 capture directory.
Both infer precision from the capture manifest. Native validation selects the
qualified Vulkan SM120 profile (C precision enum 3), without changing live demo
precision. Existing interactive tolerances are unchanged.

Results in `generated/soma/controller-reference-fp16-002`:

- Independent upstream capture: 55 seconds, 2,750 frames, 82 generations.
- Isolated controller: all 82 generation ticks, durations and transition offsets
  match; target error at most 0.00024 mm and speed error zero. This exposed a
  missing [-1,1] clamp on directional blend coordinates during a transition to
  a slower mode. The clamp now matches upstream; FP32 regression still passes.
- Native inference replay: 80/82 events pass. All 82 selected durations match.
  Events 7 (tick 256) and 79 (tick 2507) fail pose/contact tolerances. Worst errors
  are 9.008 mm position, 3.897 degrees rotation and 0.05404 contact probability;
  event 7 changes contact classification. Repeated native results are identical,
  and every FP16 call preserves subsequent FP32 results exactly.
- Native feedback fails the existing 0.05-degree rotation limit at tick 62
  (1.24 seconds), reaching 0.050562 degrees. This is a fail-fast result after
  63 ticks, not qualification of the remaining sequence.

The machine-readable reports are `oracle-controller.json`,
`native-inference.json`, and `native-controller.json`. Interactive mixed-FP16
parity is **not qualified**; previous passing fixed-chain results do not extend
that claim to this input sequence. No tolerance was relaxed for this validation.

### FP16 divergence trace

`generated/soma/controller-fp16-divergence/diagnosis.json` records the first
mismatching projection for both failing replay events. The traced native outputs
are exactly equal to the public API outputs. All three encoded floating-point
inputs and the normalized FP16 projection inputs match upstream exactly.

| Event | First differing projection | Native FP32 sum | Native FP16 | Upstream FP16 |
|---|---|---|---|---|
| 7 | root `_proj_local_root_value`, [0,1,24] | 0.981201171875 | 0.9814453125 | 0.98095703125 |
| 79 | root `_proj_global_root_value`, [0,7,11] | -0.969482421875 | -0.9697265625 | -0.96923828125 |

Both native sums land exactly on an FP16 midpoint. The exact sums of the half
operands are 0.9812011122703552 and -0.9694823920726776, respectively, slightly
on the side chosen by upstream. The discrepancy starts in projection product
accumulation/rounding before bias, propagates through the root and pose models,
and changes one discrete pose code per event: [6,4] from 12 to 0 and [8,1] from
15 to 9 (indices are zero-based). Duration selection remains equal.

The diagnostic runner now accepts optional `fixture.heading_angle` and
`fixture.allowed_token_bits` metadata and `mixed-fp16-sm120`, which uses the
public Vulkan arithmetic scope. `export_soma_controller_event.py` creates these
fixtures from a controller capture. Named `.matmul_f32` and `.matmul_half` taps
on the two root projections distinguish accumulation from casting. Adding taps
was verified to preserve outputs exactly. `diagnose_soma_controller_fp16.py`
checks public API equivalence and writes the numeric report from these traces.

For closed-loop diagnosis, `MOTIONBRICKS_CONTROLLER_TRACE=DIR` saves native
contexts, targets and outputs in the test. The second generation already has
small encoded-input differences (up to 7.16e-6), including different raw half
roundings. Those raw-input comparisons demonstrate sensitivity; they do not
establish which normalized model value first differs in the feedback run.
The diagnosis above preserves the pre-fix evidence.

### FP16 projection fix

The qualified SM120 Vulkan path now sends the 4/5-wide root projections through
the existing padded matrix product instead of the vector reduction. Pose
`_proj_local_pose` also uses the remainder-first K grouping: fixing the root
projection exposed one differing half value there in event 79. These changes
are limited to the existing mixed-FP16 arithmetic scope; the API is unchanged.

`generated/soma/controller-reference-fp16-002/native-inference-fixed.json`
passes all 82 recorded generation events using
`build/debug/libmotionbricks-fp16-candidate.so`. Maximum position error is
5.90e-6 m and rotation error 3.20e-5 degrees; contacts, duration selections,
repeatability and FP32 scope restoration are exact. The previously qualified
walk, sit and zombie three-step chains also pass; their new reports are under
`generated/soma/controller-fp16-divergence/regression-fixed-{walk,sit,zombie}`.
The full FP32 native-feedback controller regression passes all 80 events.

This does not establish closed-loop FP16 parity. The native-feedback test still
fails at tick 62 (1.24 seconds), with rotation error 0.050562 degrees against
the unchanged 0.05-degree limit. Its report is
`generated/soma/controller-reference-fp16-002/native-controller-fixed.json`.
No tolerances were relaxed. The candidate library is separate from the live
demo library, and the demo remains on FP32.

### Unified opaque API migration

SOMA now loads through `mb_model_load` with runtime option `MB_MODEL_SOMA`, uses
`mb_inference_request` skeleton/transform setters and `mb_model_infer`, and returns
`mb_motion`. The former `mb_soma_*` exports/header are removed. The demo and Python
public-API chain harness use the shared entry points. See [SOMA usage](../docs/API-SOMA.md).

Validation artifacts:

- `controller-reference-fp16-002/unified-api-inference.json`: all 82 recorded
  generation events pass, with exact contacts/durations, repeatability and FP32
  restoration. Maximum position error 5.90e-6 m, rotation 3.20e-5 degrees.
- `controller-reference-001/unified-api-controller.json`: all 80 FP32 native
  feedback events pass, maximum playback position error .2443 mm and rotation
  .01273 degrees.
- `unified-api-chain-{walk,sit,zombie}/report.json`: existing Vulkan mixed-FP16
  replay/feedback chains pass through the new API.
- `controller-reference-fp16-002/unified-api-controller.json`: the previously
  identified closed-loop FP16 failure remains at tick 62, rotation .050562
  degrees against .05. This migration does not claim to fix it.

The C API tests cover rejected settings, transactional setters, bounded output
buffers, batch/single equality, output lifetime and cleanup after a batch error.
G1 inference/model/batch regressions and Go binding/demo tests pass. The shared
library exports no `mb_soma_*` symbols.

A backend-free ASan/UBSan libFuzzer run exercised 2,684,721 inputs in 61 seconds
without a sanitizer finding. This targets public API validation/state handling,
not arbitrary GGUF parsing or GPU kernels. Reproduce with:

```sh
cmake -S . -B build/retarget-fuzz -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DMOTIONBRICKS_ENABLE_GGML=OFF -DMOTIONBRICKS_ENABLE_VULKAN=OFF \
  -DMOTIONBRICKS_ENABLE_PHYSICS=OFF -DMOTIONBRICKS_DOWNLOAD_MODELS=OFF \
  -DMOTIONBRICKS_SANITIZE=ON -DMOTIONBRICKS_ENABLE_FUZZERS=ON
cmake --build build/retarget-fuzz --target motionbricks-retarget-fuzz motionbricks-retarget-api-test
./build/retarget-fuzz/tests/motionbricks-retarget-api-test
mkdir -p /tmp/motionbricks-retarget-corpus
./build/retarget-fuzz/tests/motionbricks-retarget-fuzz /tmp/motionbricks-retarget-corpus \
  -max_total_time=60 -max_len=4096 -timeout=10 -print_final_stats=1
```

With a GGML-enabled sanitizer build, `MOTIONBRICKS_FUZZ_SOMA_BUNDLE` optionally
loads a trusted CPU bundle to exercise model-specific inference/rejection paths.
That optional model-backed fuzz mode was not part of the run above.
