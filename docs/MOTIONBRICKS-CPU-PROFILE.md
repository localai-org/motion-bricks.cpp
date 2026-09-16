# MotionBricks CPU profile and explicit batching

Measured 2026-09-16 at source revision `2727a45` plus the changes described
here. This profiles the MotionBricks planner, not SONIC.

## Method

- AMD Ryzen 9 7900, Linux 6.18.38, GGML Zen 4 backend.
- NVIDIA GeForce RTX 5070 Ti, driver 595.71.05, for the Vulkan follow-up.
- AMD Ryzen 9 7900 integrated Radeon, RADV/Mesa 26.1.8, for the iGPU follow-up.
- Released 183,148,382-parameter F32 G1 bundle and `walk.mbstyle`.
- CPU 0 for one-core tests; CPUs 0 and 1 for two-core tests.
- Model/style loading excluded; three warmups; normal policy selected 11 tokens
  and returned 44 frames.
- Four-request measurements use four agents, distinct seeds, and one model.

Build the two opt-in harnesses with:

```sh
nix develop -c cmake --preset sonic-cpu
nix develop -c cmake --build build/sonic-cpu \
  --target motionbricks-cpu-profile motionbricks-batch-profile -j2
```

Example invocations:

```sh
taskset -c 0 env OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OMP_DYNAMIC=FALSE \
  build/sonic-cpu/bin/motionbricks-cpu-profile \
  generated/g1-f32 generated/styles/walk.mbstyle 1 1 3 10

taskset -c 0,1 env OMP_NUM_THREADS=2 OMP_THREAD_LIMIT=2 OMP_DYNAMIC=FALSE \
  build/sonic-cpu/bin/motionbricks-batch-profile \
  generated/g1-f32 generated/styles/walk.mbstyle 2 4 3 8

GGML_VK_VISIBLE_DEVICES=0 build/vulkan-profile/bin/motionbricks-batch-profile \
  generated/g1-f32 generated/styles/walk.mbstyle 1 4 5 20 vulkan
```

## Profile and implemented optimizations

The original one-core `perf record` attributed 87.1% of sampled cycles to
`ggml_vec_dot_f32`, another 5.1% to surrounding matrix multiplication, 2.6%
to `memmove`, and 1.3% to F32 `im2col`. The pose transformer was the
largest stage, followed by root planning and VQ decode.

Three low-risk changes are now implemented:

1. Duration selection has its own small graph. The old path ran and discarded a
   complete six-token root prediction before an eleven-token root prediction.
   If only one duration is enabled, even the duration probe is skipped.
2. The fixed 2,560-float VQ codebook is copied once when the model loads rather
   than once per plan.
3. Public C and Go batch APIs accept 1--64 independent robots/requests under one
   shared model call. Equal-duration native work is bucketed, attention and
   convolution lanes stay isolated, and per-request seeds remain independent.

The duration change produced the useful CPU latency improvement:

| Configuration | Before | After | Change |
|---|---:|---:|---:|
| One core, one plan | 113.2 ms | 101.1 ms | -10.7% |
| Two cores, one plan | 70.7 ms | 53.9 ms | -23.8% |
| One-core root stage | 42.3 ms | 28.1 ms | -33.6% |
| Two-core root stage | 26.8 ms | 14.8 ms | -44.8% |

These before/after runs were made on the same host but at different times, so
whole-plan deltas include normal frequency variation. The removed root work is
also directly visible in the stage timings.

## Four-robot batch result

The native implementation genuinely combines dense projections across four
requests and keeps attention isolated per robot. On this CPU it preserves exact
output parity but is slower than four optimized executions:

| Cores | Four serial plans | Public B=4 call | Forced native B=4 | Native vs serial |
|---:|---:|---:|---:|---:|
| 1 | 360.8 ms | 360.7 ms | 386.4 ms | 0.934x |
| 2 | 251.1 ms | 252.8 ms | 276.0 ms | 0.910x |

The model's short 6--16-token matrices already enter GGML's small-matrix
kernels; increasing the request dimension does not offset the extra graph and
lane-handling work on one or two cores. The production CPU dispatcher therefore
uses the explicit batch API but executes its lanes independently through one
model. This preserves one model copy, one request boundary, ordered results and
all-or-nothing output ownership without accepting the 7--10% fused-graph
regression. The native path remains available internally for profiling and is
used for non-CPU backends, where the trade-off can differ.

On the RTX 5070 Ti, the fused graph does pay off:

| Vulkan workload | Round latency | Aggregate throughput |
|---|---:|---:|
| B=1 | 12.5 ms | 79.8 plans/s |
| Four separate plans | 49.7 ms | 80.4 plans/s |
| One public B=4 call | 27.0 ms | 148.1 plans/s |

That is 1.84x the throughput of four separate Vulkan calls. All four outputs
become available after 27.0 ms; 6.75 ms is the throughput-equivalent cost per
robot, not an individual robot's latency. Vulkan numerical differences versus
batch one were at most 7.72e-6 in the root stage and 6.50e-6 in final rotations.

The Ryzen integrated GPU also benefits from fusion, but is substantially slower:

| Integrated-GPU workload | Round latency | Aggregate throughput |
|---|---:|---:|
| B=1 | 58.9 ms | 17.0 plans/s |
| Four separate plans | 235.7 ms | 17.0 plans/s |
| One public B=4 call | 140.8 ms | 28.4 plans/s |

Its B=4 graph is 1.67x faster than four separate iGPU calls, 1.80x faster than
the measured two-core CPU B=4 call, and 5.21x slower than the RTX B=4 graph.
Final rotation differences versus batch one were at most 7.18e-6.

For all four CPU lanes, native versus batch-one comparisons measured zero maximum
absolute difference in root logits/results, pose logits, decoder output, final
root translations and final rotations. Mixed forced durations 6, 8, 11 and 16
also pass both public and forced-native batch parity tests.

This model is a one-shot encoder-style planner, not an autoregressive LLM.
There is no KV cache or token-by-token continuous batching. Its analogous
sharing opportunity is weight traversal in same-duration microbatches, and the
benchmark above shows that opportunity does not pay on this particular CPU.

## API behavior

- `mb_agent_plan_batch` takes parallel arrays of agents and commands.
- `mb_model_infer_batch` takes independent stateless requests.
- Go exposes `Model.PlanBatch`.
- Count is explicit, from 1 through 64; there is no hidden queue or wait window.
- Agents in a stateful batch must be unique and use the same model.
- Output order matches input order. Every output is NULL on C API failure.
- Calls sharing a model still require caller serialization.

Lower-precision weights were not adopted: the distributed bundle is F32, and a
new weight format needs conversion plus duration/token/motion parity evidence.
Persistent graph/work-buffer caching also remains a possible follow-up, but it
was not a leading sampled hotspot and would add substantially more lifecycle
complexity than the retained changes.

Raw baseline and post-change results are stored in
`docs/benchmarks/motionbricks-cpu-2026-09-16.json`.
