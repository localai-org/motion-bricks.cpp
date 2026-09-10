# SONIC: GGML versus ONNX Runtime CPU

Measured 2026-09-10 on the same Ryzen 9 7900 and physical CPU masks as
[the native CPU profile](SONIC-CPU-PROFILE.md). This compares the adopted
GGML AVX512/`-O3` F32 build with **ONNX Runtime 1.22.0 CPUExecutionProvider**
from the cached, pinned SONIC reference image. It is not a comparison with
TensorRT, GPU inference, or every ONNX Runtime version/configuration.

For matching G1-only work, GGML is **1.35x as fast on one core** and
**1.19x on two cores**: 25.7% / 16.0% lower pair latency. Against the original
multi-branch encoder, the ratios are 2.36x / 2.30x, partly because that graph
does extra work.

## Results

Encoder-plus-decoder latency in milliseconds. Each cell is the median of
the corresponding per-run statistic over three runs (not a pooled percentile).
All ONNX rows enable graph optimization; primary ONNX rows disable spinning.

| Runtime/graph | Cores | Encoder mean | Decoder mean | Pair mean | Pair P95 | Pair P99 |
|---|---:|---:|---:|---:|---:|---:|
| GGML AVX512 / O3, G1 | 1 | 0.272 | 0.657 | **0.932** | 1.041 | 1.113 |
| ONNX Runtime, G1 extraction | 1 | 0.385 | 0.867 | 1.254 | 1.381 | 1.493 |
| ONNX Runtime, original encoder | 1 | 1.233 | 0.965 | 2.203 | 2.431 | 2.598 |
| GGML AVX512 / O3, G1 | 2 | 0.185 | 0.430 | **0.617** | 0.690 | 0.726 |
| ONNX Runtime, G1 extraction | 2 | 0.231 | 0.503 | 0.735 | 0.818 | 0.873 |
| ONNX Runtime, original encoder | 2 | 0.804 | 0.612 | 1.419 | 1.579 | 1.677 |
| ONNX Runtime, G1 extraction, spinning enabled | 2 | 0.510 | 1.036 | 1.531 | 4.038 | 4.105 |

Run means for GGML span 0.912–0.946 ms / 0.612–0.617 ms; G1-only ONNX
without spinning spans 1.250–1.258 ms / 0.729–0.749 ms. None of the measured
pairs exceeds 20 ms. The spinning control's two session pools compete within
the same two-core budget, and its run means span 1.435–1.687 ms. We use the
faster non-spinning ONNX configuration for the headline comparison rather
than amplifying that scheduling disadvantage. A shared-pool or single-session
ONNX pipeline was not tested.

## What is being compared

The official ONNX encoder evaluates **three** five-layer MLP branches (G1,
teleop and SMPL), quantizes their outputs and then selects the requested
mode. The original graph contains 15 Gemm nodes and three FSQ paths. Native
SONIC implements only G1 mode 0, so an unmodified-model comparison includes
an advantage from specialization, not just the inference engine.

To compare the same branch, the harness also extracts the original G1
quantizer output `/quantizer/Cast_output_0` and its dependency graph. It adds
a final reshape to the original `[1, 64]` output interface. This retains the
original observation packing, five Gemm layers, weights and FSQ operations;
it does not reconstruct the MLP from the native implementation. Inputs are
required to select mode 0. The extracted encoder has 59 nodes and five Gemm
nodes. The original decoder is unchanged (48 nodes, seven MatMul nodes).

## Method

- Original ONNX hashes are checked before loading. The G1-only graph hash is
  `6d240804f3639bee7b35ecca27d638536ad980c0f7ad0fe763fbb2a886959ae5` with
  ONNX 1.18.0. Each JSON records the original policy hashes, serving graph
  hashes and runtime build information.
- ONNX Runtime uses `ORT_ENABLE_ALL`, sequential graph execution,
  `inter_op_num_threads=1`, and one/two intra-op threads per session.
  Encoder and decoder have separate session threadpools. The principal
  comparison disables worker spinning so idle encoder workers do not compete
  with decoder workers; a separate two-core G1 comparison enables spinning.
- CPU 2 for one core, CPUs 2 and 3 for two distinct physical cores. Docker
  also applies the same cpuset; no GPU is exposed. Worker affinity is checked
  after inference. With two cores, ORT has the caller plus two session workers,
  all confined to those two cores. GGML uses one shared backend worker team.
- ONNX Runtime runs in the existing
  `motionbricks-sonic-ggml-reference:onnx1.18` container, with networking
  disabled and a read-only repository mount. GGML runs on the host. Both
  execute on the same host CPU; container startup/loading are excluded.
- Three sequential fresh-process runs per configuration, reversing runtime
  and core-count order on the second repeat. Each run loads the same 2,205
  observations, warms up for 100 pairs, then times 2,205 encoder+decoder pairs.
  No concurrent benchmark/build jobs or changes to other user processes.
- ONNX uses I/O binding and preallocated F32 output arrays. Latency includes
  input rebinding, the Python/native entry point and copying newly encoded
  tokens into the decoder input. GGML includes its ctypes/C ABI entry point,
  validation, native packing and token copy. These are invocation-level
  serving measurements, not isolated matrix-kernel timings. Fixture reads,
  initialization and parity checks are excluded.
- The machine remains under its normal governor and background load; neither
  CPU reservation nor fixed frequency is assumed. This is burst inference,
  not a new 50 Hz or full physics measurement.

## Correctness

Every ONNX run first compares all 2,205 samples against the existing
independent fixture: exact FSQ tokens, captured-input decoder actions and
encoder-to-decoder composed actions. Graph outputs are not expanded to expose
layers, which would inhibit optimization and change the measured graphs.
Numerical outputs retain the existing absolute error `5e-4` and relative L2
`1e-5` limits. The native build separately passes its full layer/output suite.
All original, extracted and spinning-control ONNX runs pass: zero token
mismatches or action/composed-action failures; maximum action absolute error
`2.86102e-6` and relative L2 error `6.29111e-7` across all runs.

## Reproduce

Build native inference with `cmake --preset sonic-cpu-avx512` and
`cmake --build --preset sonic-cpu-avx512`. Use the normal native harness:

```sh
python reference/profile_sonic_cpu.py \
  --library build/sonic-cpu-avx512-o3/libmotionbricks.so \
  --model generated/sonic/ggml/sonic-g1.gguf \
  --fixture generated/sonic/ggml/all-layers.mbsonic \
  --cpus 2 --iterations 2205
```

Run the ONNX harness in the cached reference image:

```sh
docker run --rm --network none --cpuset-cpus 2 \
  --mount "type=bind,src=$PWD,dst=/work,readonly" --workdir /work \
  --entrypoint python motionbricks-sonic-ggml-reference:onnx1.18 \
  reference/profile_sonic_onnx.py \
  --policy generated/sonic/policy \
  --fixture generated/sonic/ggml/all-layers.mbsonic \
  --cpus 2 --iterations 2205 --encoder-scope g1 --spinning 0
```

Repeat with both CPU arguments set to `2,3`. Use `--encoder-scope original`
for the unmodified graph and `--spinning 1` for the spinning control. Select
the correct physical CPU IDs for another machine. A failed parity check
produces a report but returns nonzero, so inaccurate results cannot silently
pass the comparison runner.

Raw JSON and logs are in `generated/sonic/ggml/cpu-profile/onnx/`.
The [committed summary](benchmarks/sonic-cpu-2026-09-10.json) preserves
per-run burst statistics and the aggregate comparisons without large captures.
The native source/model and fixture hashes are recorded in the original
[CPU report](SONIC-CPU-PROFILE.md).
