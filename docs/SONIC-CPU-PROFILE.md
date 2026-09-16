# SONIC inference on one and two CPU cores

For a direct CPU comparison with the official ONNX model and a G1-only
extraction, see [GGML versus ONNX Runtime](SONIC-ONNX-COMPARISON.md).
The [committed benchmark summary](benchmarks/sonic-cpu-2026-09-10.json)
includes the individual burst-run statistics behind the comparison tables.

Measured 2026-09-10 on an AMD Ryzen 9 7900. Enabling GGML's AVX2/FMA
kernels is the largest demonstrated improvement: an encoder-plus-decoder
pair falls from 9.01 to 1.36 ms on one physical core, and from 4.49 to
0.71 ms on two. Both optimized configurations pass the independent ONNX
parity fixture, including exact FSQ tokens. These are inference timings;
MuJoCo, MotionBricks planning and browser rendering are excluded.

A same-day [AVX512 follow-up](#avx512-follow-up) improves burst latency
further to 1.13 ms / 0.66 ms on this machine, with independent parity passing.
The subsequent [`-O3` follow-up](#compiler-optimization-follow-up) reaches
0.95 ms / 0.62 ms and also passes parity.

The default distributable build now uses GGML runtime CPU selection. The
**`sonic-cpu` preset** builds all supported CPU variants for the target
architecture with `-O3` and profiling symbols:

```sh
cmake --preset sonic-cpu
cmake --build --preset sonic-cpu
taskset -c 2,3 ctest --preset sonic-cpu
```

It builds into `build/sonic-cpu`, disables downloads, and uses local weights
and fixtures. Tests include module discovery, API validation and parity at
one and two threads. Callers still set their own thread count; use the
profiling harness for strict physical-core affinity. See
[backend selection, installation and validation](CPU-BACKENDS.md).

The measurements below used a single AVX512 variant; the
`sonic-cpu-avx512` preset retains those exact ISA settings with runtime
variant selection explicitly disabled for historical reproduction and the
optional kernel interposer. No F16 or custom-kernel experiment was adopted.

## Method

- Source: `0c72c57b7f5021c3ba50f13cd2f2d9e24decbd19`; pinned GGML
  `8c63e70982c95ceb862e3a1073a2c1beef75d60a`. The profiling changes add a
  harness and configurable parity thread count; production inference code
  and weights are unchanged.
- GCC 15.2.0, Linux 6.18.38, OpenMP enabled. Optimized builds use
  `RelWithDebInfo` (`-O2 -g -DNDEBUG`), without fast-math. The existing
  `build/debug` has no GGML AVX, AVX2, FMA or SSE4.2 enabled. The generic
  optimized control keeps these disabled; the AVX2 build explicitly enables
  SSE4.2, BMI2, F16C, AVX, AVX2 and FMA.
- One core: logical CPU 2. Two cores: CPUs 2 and 3, distinct physical cores
  sharing a 32 MiB L3 cache. Their SMT siblings (14 and 15) are excluded
  from the inference process. All workers inherit affinity before backend
  initialization, and the harness verifies every surviving thread's mask.
  `mb_runtime_options_set_threads` and OpenMP's thread limit are set to 1/2.
- Governor: `powersave`, frequency boost enabled. No exclusive CPU
  reservation, fixed frequency or changes to other user processes. Results
  characterize this machine under its observed load, not a real-time bound.
- Original G1 mode-0 F32 GGUF, 14,415,453 parameters. Inputs cycle through
  all 2,205 samples in `all-layers.mbsonic`: recorded controller observations
  and 16 synthetic cases. The decoder receives the newly encoded tokens.
- Each burst run excludes loading and 100 warmup pairs, then measures 2,205
  pairs. Three runs per configuration, reversing configuration order for
  the second repetition. No concurrent benchmark or build jobs.
- Python `ctypes` calls the native C ABI. Per-stage wall time includes FFI,
  validation, locking and graph execution. Pair time also includes copying
  64 tokens into the decoder input. Fixture reads and input allocation are
  outside the timed interval; parity comparison is a separate executable.

Model SHA256:
`c3fe437837debffbd44fc85190330a255fc8fb3468c190113c6feae28a2d64e0`.
Fixture SHA256:
`6321db83d3ce6ae8d9a541a861e2dd2b39fdab1e6d9a385dc5e96c4f68ec64db`.

## Burst latency

Milliseconds; each column is the median of that statistic across three
runs. P95/P99 are per-run pair percentiles, not percentiles of pooled data.

| Build | Physical cores | Encoder mean | Decoder mean | Pair mean | Pair P95 | Pair P99 |
|---|---:|---:|---:|---:|---:|---:|
| Existing Debug, scalar | 1 | 2.637 | 6.370 | 9.012 | 10.029 | 11.743 |
| Existing Debug, scalar | 2 | 1.327 | 3.158 | 4.490 | 5.041 | 5.099 |
| Optimized, scalar | 1 | 2.463 | 5.959 | 8.425 | 9.597 | 9.904 |
| Optimized, scalar | 2 | 1.220 | 2.936 | 4.159 | 4.414 | 4.707 |
| Optimized, AVX2/FMA | 1 | 0.390 | 0.967 | **1.359** | 1.385 | 1.454 |
| Optimized, AVX2/FMA | 2 | 0.215 | 0.496 | **0.712** | 0.767 | 0.817 |

AVX2 run means ranged from 1.340–1.375 ms (one core) and
0.712–0.731 ms (two). Two cores give 1.91x the throughput of one.
Relative to the existing Debug build, the measured improvement is 6.63x
and 6.30x respectively; relative to the optimized scalar control, 6.20x
and 5.84x. Merely selecting an optimized build type does not fix disabled
SIMD kernels.

Across 6,615 measured pairs per configuration, neither AVX2 configuration
exceeded 20 ms; maxima were 1.874 ms and 1.037 ms. The existing two-core
Debug runs did include a 93.8 ms outlier. Its cause was not traced, and it
must not be hidden by the percentile summary. These are burst observations,
not a guarantee about future scheduler delays.

## AVX512 follow-up

The Ryzen 9 7900 exposes all five instruction subsets enabled by
`GGML_AVX512=ON`: AVX512F/CD/VL/DQ/BW. GGML already implements an F32
AVX512 dot-product path; disassembly of the resulting `ggml_vec_dot_f32`
confirms ZMM loads and FMA instructions. No weight conversion or production
source changes are needed. AVX512 VNNI, BF16 and VBMI options remain off.

Built `build/sonic-cpu-avx512` with the same optimized settings as AVX2,
adding only `-DGGML_AVX512=ON`. A fresh comparison repeats both builds
three times using the same 2,205-pair burst protocol, CPUs 2 / 2,3 and
default OpenMP waiting without individual worker binding. Configuration
order is reversed for the second repeat. Values are medians of per-run
statistics, in milliseconds; the older AVX2 table above is retained as the
original baseline.

| ISA | Cores | Encoder mean | Decoder mean | Pair mean | Pair P95 | Pair P99 |
|---|---:|---:|---:|---:|---:|---:|
| AVX2 | 1 | 0.391 | 0.946 | 1.339 | 1.393 | 1.439 |
| AVX512 | 1 | 0.313 | 0.818 | **1.134** | 1.240 | 1.342 |
| AVX2 | 2 | 0.213 | 0.494 | 0.708 | 0.756 | 0.792 |
| AVX512 | 2 | 0.195 | 0.462 | **0.661** | 0.723 | 0.776 |

AVX512 reduces pair latency by **15.3% on one core** and **6.6% on two**
relative to the fresh AVX2 measurements. Its run means range from
1.131–1.200 ms and 0.658–0.674 ms. Neither core configuration exceeds 20 ms
in 6,615 measured pairs; maxima are 1.469 and 1.021 ms.

Both AVX512 core counts pass all 2,205 independent parity observations,
including exact FSQ tokens, all layer preactivations, decoder actions and
composed actions. Maximum absolute error is `1.52588e-5`, and maximum
relative L2 error is `7.18279e-7`, within the existing limits. These values
differ from AVX2 because the vector reduction changes accumulation order.

An additional 50 Hz check used explicit worker binding and passive OpenMP
waiting for both ISAs, with one 500-pair run per configuration:

| ISA | Cores | Pair mean (ms) | Pair P99 (ms) | CPU use, one core = 100% |
|---|---:|---:|---:|---:|
| AVX2 | 1 | 1.691 | 1.909 | 8.52% |
| AVX512 | 1 | 1.447 | 1.955 | 7.31% |
| AVX2 | 2 | 1.300 | 1.709 | 11.29% |
| AVX512 | 2 | 1.346 | 4.371 | 11.05% |

No run missed its completion deadline. AVX512 improves the observed one-core
mean and CPU use, but **does not demonstrate a two-core paced-latency
improvement**: a longer tail (maximum 14.39 ms) raises its mean despite a
slightly lower median (1.250 vs 1.263 ms). These short cadence checks should
not be treated as definitive tail-latency comparisons. The repeated burst
result is stronger evidence of the kernel's speedup.

For this host, AVX512 is a useful incremental improvement over AVX2. This is
a hardware-specific build, not an automatic runtime fallback for CPUs that
lack the enabled instructions. Reproduce by using the optimized configure
command below with build directory `build/sonic-cpu-avx512` and adding
`-DGGML_AVX512=ON`; then use that directory's parity binary and library.
Follow-up logs, JSON and disassembly are saved separately under
`generated/sonic/ggml/cpu-profile/avx512/`.

## Compiler optimization follow-up

Inspection of the AVX512 `-O2` disassembly found a concrete inefficiency:
the four vector accumulators in `ggml_vec_dot_f32` are held on the stack,
with a load/store pair on each iteration of the four-way inner loop.
Compiling with `-O3` unrolls that loop and keeps all four accumulators in
ZMM registers. The main loop's accumulator stack accesses disappear.

Built a separate `build/sonic-cpu-avx512-o3`, changing only optimization
level from the prior AVX512 build. All source, weights, ISA flags and thread
settings remain the same. Debug symbols are retained and fast-math is off.
Fresh, alternating comparisons use the same three 2,205-pair burst runs
per configuration and CPUs 2 / 2,3. Milliseconds, median of each per-run
statistic:

| Optimization | Cores | Encoder mean | Decoder mean | Pair mean | Pair P95 | Pair P99 |
|---|---:|---:|---:|---:|---:|---:|
| AVX512 `-O2` | 1 | 0.312 | 0.801 | 1.117 | 1.168 | 1.232 |
| AVX512 `-O3` | 1 | 0.278 | 0.670 | **0.951** | 1.052 | 1.146 |
| AVX512 `-O2` | 2 | 0.194 | 0.455 | 0.650 | 0.703 | 0.752 |
| AVX512 `-O3` | 2 | 0.187 | 0.431 | **0.621** | 0.674 | 0.713 |

This is a further **14.8% / 4.4% latency reduction**. The `-O3` run means
span 0.944–0.952 ms and 0.613–0.624 ms. All 6,615 pairs per configuration
finish within 20 ms; maxima are 1.470 ms and 0.954 ms. This follow-up tests
burst inference; it does not establish new 50 Hz or physical-rollout results.
Because `-O3` was applied to the whole build, the end-to-end improvement
cannot be attributed exclusively to the accumulator change.

Both core counts pass all 2,205 independent parity cases, including exact
FSQ tokens. Maximum absolute error is `1.52588e-5`, relative L2
`7.5854e-7`. A separate 10,000-pair sampled profile collected 4,976 / 6,380
samples with zero loss. Dot products still take 90.99% / 87.36% of sampled
user cycles; matrix-multiply dispatch takes 3.60% / 3.29%. On two cores,
the two principal OpenMP barrier symbols total 4.41%. Startup and warmup
are included in these sampling percentages, as in the earlier profiles.

The next substantial experiments remain matrix-vector kernels and reduced
weight storage. Plan caching, pool bookkeeping and small-operation fusion
are lower-priority candidates: they do not account for most sampled cycles.
No additional speedup is claimed for those untested changes.

Reproduce the AVX512 configure command in a new directory, adding:

```sh
-DCMAKE_BUILD_TYPE=RelWithDebInfo \
'-DCMAKE_C_FLAGS_RELWITHDEBINFO=-O3 -g -DNDEBUG' \
'-DCMAKE_CXX_FLAGS_RELWITHDEBINFO=-O3 -g -DNDEBUG'
```

Alternatively, this toolchain's `Release` configuration uses `-O3 -DNDEBUG`;
keep the explicit AVX512/AVX2/FMA ISA options. The measured build retains
symbols to support profiling. Reports, parity logs and before/after assembly
are in `generated/sonic/ggml/cpu-profile/o3/` and the earlier `avx512/` folder.

## Llamafile SGEMM follow-up

Enabling GGML's Llamafile CPU matrix kernels benefits batched SONIC calls. On
the Ryzen 9 7900, a fused batch of four encoder/decoder pairs fell from 2.708
to 2.548 ms on one core (-5.9%) and from 1.428 to 1.315 ms on two cores (-7.9%).
The two-core result was noisier than the one-core result: individual reductions
were 2.2% and 13.6%.

Batch one remains a matrix-vector workload, which Llamafile deliberately
declines. A small vendored GGML guard keeps that rejected dispatch out of the
hot path. In matched alternating runs, the resulting build reduced batch-one
pair latency from 1.094 to 1.038 ms (-5.1%); that smaller improvement appears
to come from the changed fallback layout rather than the tinyBLAS kernel.

The expanded CPU harness accepts `--batch`; allocation and fixture assembly
remain outside the timed interval. SONIC API validation passes, including
exact batch encoder tokens and decoder agreement within the existing tolerance,
and the full 2,205-sample batch-one parity suite still passes on one and two
cores. Raw measurements are in
`benchmarks/llamafile-cpu-2026-09-16.json`.

## Decoder-kernel and reduced-storage experiments

An optional [diagnostic interposer](../reference/SONIC-CPU-EXPERIMENTS.md)
tested a four-row matrix-vector kernel that shares each activation load across
four output rows. It runs selected graph nodes as CPU custom operations,
keeping the public inference path, bias/SiLU and FSQ unchanged. A second
mode rounds only weights to an F16 cache and widens them into F32 registers;
activations and accumulation remain F32. The F16 cache is used for either
the decoder alone or both networks. Original F32 tensors remain allocated:
this tests the active weight working set, not a reduced-memory model loader.

The F32 custom kernel passes all 2,205 fixture observations on both core
counts, for both decoder-only and all-layer scope. This provides a control
for the kernel's indexing, partitioning, tail handling and reduction tree.
Full-fixture numerical results at both one and two threads:

| Experiment | FSQ tokens differing | Samples with decoder action failure | Samples with composed-action failure | Maximum absolute error across checked arrays | Maximum relative L2 |
|---|---:|---:|---:|---:|---:|
| Four-row F32 | 0 | 0 | 0 | 1.52588e-5 | 7.5854e-7 |
| F16 decoder weights | 0 | 2,205 | 2,205 | 0.00918198 | 0.000318715 |
| F16 encoder + decoder weights | 41 | 2,205 | 2,205 | 0.0628353 | 0.0652328 |

The 41 token differences occur in 41 samples. Decoder-only rounding leaves
encoder tokens unchanged, but still fails the existing action tolerance on
every sample. The F16 results fail the current contract and are not adopted;
their effect on closed-loop physics was not evaluated. `--report-all` on the
parity executable collects all mismatches while preserving failure exit status
and the original numerical thresholds.

Three new burst runs per configuration, 2,205 measured pairs per run, with
configuration order reversed on the second repeat. Baseline and experiments
all use the same AVX512/`-O3` runtime and physical CPU masks. Pair timings in
milliseconds; each cell is the median run mean, followed by the full range
of run means:

| Implementation | One core | Two cores | Decision |
|---|---:|---:|---|
| Standard GGML F32 | **0.930** (0.928–0.935) | **0.617** (0.614–0.619) | Adopted |
| Four-row F32 decoder | 1.373 (1.364–1.754) | 0.828 (0.796–1.387) | Slower, despite passing parity |
| F16 decoder cache | 1.255 (1.001–2.106) | 0.697 (0.548–0.711) | Fails accuracy |
| F16 encoder + decoder cache | 1.191 (0.885–2.049) | 1.179 (0.486–1.251) | Fails accuracy |

The F32 decoder prototype is slower even in its best runs. The F16 timing
spread is substantial, so the occasional faster run does not establish a
reliable performance win; accuracy independently disqualifies it. All-network
F16 also has eight pairs over 20 ms across its two-core runs, with a 118 ms
maximum. These measurements apply to this prototype and do not rule out
different tiling or mixed-precision designs. No new inference fast path or
relaxed weight validation is enabled in the adopted preset.

Artifacts are under `generated/sonic/ggml/cpu-profile/kernels/`. The optional
experiment target is excluded from normal builds and installation. See
[experiment setup and limitations](../reference/SONIC-CPU-EXPERIMENTS.md)
for exact environment settings and reproduction.

## Hotspots and improvement priorities

Separate `perf record` runs collected 6,911 and 7,518 user-cycle samples
over 10,000 pairs, with zero lost samples. The F32 dot-product kernel
`ggml_vec_dot_f32` accounts for **94.72%** of sampled cycles on one core
and **91.44%** on two. The two main OpenMP barrier symbols account for
another 2.66% on two cores. These are self-cycle shares across the sampled
process, including startup; the 1.5–1.7% memmove share includes model/fixture
loading and must not be interpreted as inference transfer overhead.

The decoder consumes approximately 70–71% of measured inference latency.
Its seven matrices contain 10,177,024 F32 weights (40.7 MB), versus
4,227,072 weights (16.9 MB) in the encoder. A pair performs approximately
28.8 million floating-point multiply/add operations and reads 57.6 MB of
matrix weights at the algorithm level. This is not a measurement of DRAM
traffic: cache reuse and prefetching affect actual traffic. The sampled
hotspot includes arithmetic and memory stalls, so it does not establish
which of those limits the vector kernel.

Priorities, distinguishing measured improvements from further experiments:

1. **Enable SIMD and `-O3` in an optimized CPU build.** Both improvements are
   demonstrated and pass parity on both core counts. Keep explicit ISA settings or add
   suitable CPU dispatch; a build requiring AVX2/FMA must run on compatible
   hardware. Existing cache entries can keep these options disabled even
   when changing the CMake build type.
2. **Choose threads and waiting policy for the actual cadence.** Two cores
   nearly halve burst latency, while one core already has substantial room
   within a 20 ms inference budget. Measure CPU use and deadline tails at
   50 Hz before choosing the default; see below.
3. **Concentrate kernel experiments on decoder matrix-vector products.**
   The four-row prototype above regresses performance; other tiling/prefetch
   choices remain unbenchmarked. The existing AVX512 path has been measured
   and adopted with `-O3`. The decoder's 2048x2048 matrix alone
   contains 16.8 MB of weights, making it a useful first target for per-layer
   instrumentation.
4. **Reduced weight storage requires a separate accuracy contract.**
   F16 storage would reduce all matrix weights to about 28.8 MB, small enough
   in principle for this core group's 32 MiB L3. Conversion costs, competing
   cache users and token sensitivity may offset the benefit. The current
   loader deliberately accepts only F32. The F16 experiment above fails
   exact tokens/action parity; it is rejected under the existing contract.
5. **Treat graph/pool reuse and bias/SiLU fusion as secondary work.** Graphs
   and tensor buffers are already built once in `src/sonic.cpp`. The CPU
   backend still replans each call and uses disposable GGML threadpool
   bookkeeping because no pool is attached in `src/neural_runtime.cpp`.
   OpenMP manages its own worker team; this does not mean a new OS worker is
   spawned for every inference. Reusing plans/pool bookkeeping or fusing
   small operations could reduce overhead, but the sampled dot-product
   dominance leaves limited headroom compared with kernel improvements.

## 50 Hz latency and CPU consumption

Two 500-pair runs per configuration, 20 ms scheduled arrival intervals,
100 warmup pairs. Ranges below show both run results. CPU percentage is
process CPU seconds / wall seconds x 100, where 100% means one fully busy
core; it includes OpenMP activity between ticks. Deadline misses mean a pair
finished after its scheduled arrival plus 20 ms, so catch-up pairs following
a long stall can also miss their deadlines.

| AVX2 configuration | Pair mean range (ms) | Pair P99 range (ms) | CPU use | Deadline misses across 1,000 pairs |
|---|---:|---:|---:|---:|
| One core | 1.682–1.694 | 1.851–1.988 | 8.48–8.53% | 0 |
| Two cores, default OpenMP waiting | 1.336–3.192 | 2.572–63.816 | 35.0–41.7% | 49 |
| Two cores, `OMP_WAIT_POLICY=PASSIVE` | 1.303–1.317 | 1.939–1.944 | 11.35–11.38% | 0 |
| Two cores, explicit thread binding, default waiting | 1.145–1.161 | 1.420–1.846 | 34.11–34.23% | 0 |
| Two cores, explicit thread binding, passive waiting | 1.287–1.289 | 1.573–1.745 | 11.16–11.25% | 0 |

The default two-core runs had 41.2/157.0 ms maximum pair latencies; passive
runs had 2.76/8.40 ms maxima. The cgroup and its ancestors had no CPU quota
or recorded throttling. The measurements establish a useful waiting-policy
comparison under observed conditions, but do not isolate the cause of every
outlier. CPU affinity restricts placement; it does not reserve CPUs against
other processes. Passive waiting reduced measured CPU consumption by
67–73%. With one core there is no second OpenMP worker to spin while the
caller sleeps.

The binding follow-up sets `OMP_PROC_BIND=TRUE OMP_PLACES='{2},{3}'` before
backend initialization. Affinity checks confirm the caller on CPU 2 and
the worker on CPU 3, rather than allowing either thread on either core.
Two further runs per waiting policy had zero deadline misses. Default
waiting maxima were 2.15/4.65 ms; passive maxima were 8.09/6.77 ms. This
supports explicit binding as a useful scheduling control; it does not
retroactively prove the cause of the earlier stalls. Bound default waiting
has slightly lower ordinary latency, while bound passive waiting uses about
one third as much CPU.

For a single 50 Hz controller on this machine, one optimized core is the
lowest-CPU measured option. Use two cores when the lower burst latency is
valuable; explicitly bind the workers and use passive waiting when idle CPU
consumption matters. Validate those settings with the actual workload.
This budget covers SONIC inference alone; physics and planning need their
own allowance.

## Correctness

The updated `motionbricks-sonic-parity` accepts an optional thread count.
Both optimized builds passed all 2,205 observations at both one and two
threads: every MLP preactivation, captured-input decoder actions, composed
actions, and exact FSQ tokens. AVX2 maximum absolute error was
`1.23978e-5` and maximum relative L2 error `8.16604e-7`, within the existing
`5e-4` and `1e-5` limits. SIMD changes the accumulation path, so this
validation matters even with F32 weights.

## Reproduce

Use the repository development environment (`nix develop`, or equivalent
local CMake/Ninja/C++/Python tools). All model and fixture files must already
exist; no downloads are required. The existing demo build is left intact.

```sh
cmake -S . -B build/sonic-cpu-avx2 -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DMOTIONBRICKS_ENABLE_VULKAN=OFF \
  -DMOTIONBRICKS_ENABLE_PHYSICS=OFF \
  -DMOTIONBRICKS_DOWNLOAD_MODELS=OFF \
  -DMOTIONBRICKS_BUILD_TESTS=OFF \
  -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
  -DGGML_SSE42=ON -DGGML_BMI2=ON
cmake --build build/sonic-cpu-avx2 --target motionbricks-sonic-parity -j2

taskset -c 2 build/sonic-cpu-avx2/bin/motionbricks-sonic-parity \
  generated/sonic/ggml/sonic-g1.gguf \
  generated/sonic/ggml/all-layers.mbsonic cpu 1
taskset -c 2,3 build/sonic-cpu-avx2/bin/motionbricks-sonic-parity \
  generated/sonic/ggml/sonic-g1.gguf \
  generated/sonic/ggml/all-layers.mbsonic cpu 2

python reference/profile_sonic_cpu.py \
  --library build/sonic-cpu-avx2/libmotionbricks.so \
  --model generated/sonic/ggml/sonic-g1.gguf \
  --fixture generated/sonic/ggml/all-layers.mbsonic \
  --cpus 2 --iterations 2205
```

Repeat the harness three times with `--cpus 2` and `--cpus 2,3`. Adapt
CPU IDs to the local topology; the harness rejects SMT siblings, more than
two cores, or CPUs outside the current allowed mask. For the optimized
scalar control, use a separate build directory and set all six ISA options
above to `OFF`. To measure the existing Debug library, change `--library`.

For controller cadence, add `--period-ms 20 --iterations 500`. Compare
the default OpenMP environment with `OMP_WAIT_POLICY=PASSIVE` set before
starting Python. The JSON includes process CPU seconds, wall time,
start-lateness percentiles and completion deadline misses. Latency excludes
sleep; CPU time includes worker activity between ticks.
For the bound two-core experiment, prefix the harness with
`env OMP_PROC_BIND=TRUE OMP_PLACES='{2},{3}'`, optionally adding
`OMP_WAIT_POLICY=PASSIVE`, and pass `--cpus 2,3`.

For sampled hotspots, prefix a separate burst run with:

```sh
perf record -e cycles:u -F 499 -g --call-graph dwarf,8192 \
  -o perf-sonic.data -- python reference/profile_sonic_cpu.py \
  --library build/sonic-cpu-avx2/libmotionbricks.so \
  --model generated/sonic/ggml/sonic-g1.gguf \
  --fixture generated/sonic/ggml/all-layers.mbsonic \
  --cpus 2 --iterations 10000
perf report -i perf-sonic.data --stdio --no-children -g none --sort symbol
```

Profiling runs are separate from latency results because sampling changes
timing. Sampling includes process startup, loading and warmup; the latency
JSON excludes them. Raw local JSON, parity logs, build logs, counter results
and perf captures are under `generated/sonic/ggml/cpu-profile/` (gitignored).
The harness and this report are tracked; weights and large captures are not.
