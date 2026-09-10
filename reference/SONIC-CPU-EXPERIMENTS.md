# SONIC CPU kernel and weight-storage experiments

Default builds use standard GGML, F32 weights and runtime CPU selection; see
[CPU backends](../docs/CPU-BACKENDS.md). To reproduce the single-variant
AVX512/`-O3` experiments and use the interposer below:

```sh
cmake --preset sonic-cpu-avx512
cmake --build --preset sonic-cpu-avx512
taskset -c 2,3 ctest --preset sonic-cpu-avx512
```

Use the repository development shell or equivalent CMake/Ninja/C++ tools.
The preset uses existing local model/fixture files; no downloads occur.
CPU IDs here match the profiled Ryzen 9 7900. Select appropriate physical
cores on another host. Runtime callers should explicitly set one/two threads;
the preset does not change the public API's default thread count. The test
preset caps OpenMP workers at two and runs tests sequentially. The API tests
retain assertions in optimized builds, and the parity tests cover both one
and two threads.

`sonic_cpu_experiment.cpp` is an optional Linux AVX512 diagnostic interposer.
It is never installed, linked into the production runtime, or loaded by the
normal test/build presets. It replaces selected SONIC matrix-vector graph
nodes with a CPU custom operation that computes four output rows together,
sharing each activation-vector load across those rows. A one-row tail handles
the final 29-output layer. The reduction uses GGML's four-accumulator F32
tree. Bias, SiLU, observations, FSQ and the public API stay in the existing
runtime.

Modes and scope:

| Environment variable | Values | Meaning |
|---|---|---|
| `SONIC_CPU_EXPERIMENT` | `f32`, `f16` | Keep weights F32, or round a cache to F16 and widen on load for F32 accumulation |
| `SONIC_CPU_EXPERIMENT_SCOPE` | `decoder` (default), `all` | Replace the seven decoder matrices, or all twelve matrices |

Only weights are rounded in F16 mode; activations, accumulation and biases
remain F32. Original F32 tensors remain allocated to preserve the normal
loader and graph ownership. The experiment tests a smaller **active matrix
working set**, not reduced process resident memory or a deployable F16 GGUF.
F16 caches and metadata live until process exit. Use one model per process
and CPU inference only; model-loading concurrency and Vulkan are unsupported.

Build the explicitly selected diagnostic target:

```sh
cmake --build build/sonic-cpu-avx512-o3 \
  --target motionbricks-sonic-cpu-experiment -j2
```

Check the entire accuracy fixture before interpreting any performance result:

```sh
taskset -c 2 env \
  LD_PRELOAD="$PWD/build/sonic-cpu-avx512-o3/libmotionbricks-sonic-cpu-experiment.so" \
  SONIC_CPU_EXPERIMENT=f32 SONIC_CPU_EXPERIMENT_SCOPE=decoder \
  OMP_THREAD_LIMIT=1 OMP_DYNAMIC=FALSE \
  build/sonic-cpu-avx512-o3/bin/motionbricks-sonic-parity \
  generated/sonic/ggml/sonic-g1.gguf \
  generated/sonic/ggml/all-layers.mbsonic cpu 1 --report-all
```

Repeat with CPUs `2,3`, thread limit 2 and parity argument `cpu 2`.
`--report-all` continues after numerical mismatches, reports how many samples
failed each comparison and counts individual token differences. It still
returns exit code 1 on failure and retains the existing absolute/relative
thresholds. Invalid fixtures and non-finite values remain fatal.

For latency, apply the same environment to `python reference/profile_sonic_cpu.py`
with the normal model/fixture, `--library build/sonic-cpu-avx512-o3/libmotionbricks.so`,
`--cpus 2` or `--cpus 2,3` and `--iterations 2205`. Baseline runs omit
`LD_PRELOAD`. Use a fresh process for each measurement. The harness enforces
and checks affinity before/after inference; set OpenMP environment variables
before process startup because the interposer preloads the CPU backend.

Observed results and adoption decisions are in
[the profiling report](../docs/SONIC-CPU-PROFILE.md). F16 fails the current
accuracy contract and must not be treated as an accepted optimization.
