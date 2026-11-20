# tinyBLAS cache, tile, and accumulator experiments

Date: 2026-09-16

## Result

None of the four experiments improves Motion Bricks planning on this CPU. The
upstream `4 x 6` tinyBLAS tile remains the fastest choice. At the representative
10-token/40-frame workload, the experiments change median one-core latency by:

| experiment | 1-core latency | change | 2-core latency | change |
|---|---:|---:|---:|---:|
| upstream control | 57.710 ms | - | 33.870 ms | - |
| 2 x N tile | 80.322 ms | +39.2% | 43.823 ms | +29.4% |
| prefetch, distance 2 | 59.977 ms | +3.9% | 34.788 ms | +2.7% |
| prefetch, distance 4 | 59.395 ms | +2.9% | 34.841 ms | +2.9% |
| prefetch, distance 8 | 59.744 ms | +3.5% | 34.782 ms | +2.7% |
| four-row packed weights | 66.040 ms | +14.4% | 37.470 ms | +10.6% |
| K-blocked accumulators, K=256 | 61.415 ms | +6.4% | 35.411 ms | +4.5% |

The retired-FP counter is essentially constant at 258.3-258.7 billion
operations for 60 measured plans. These transformations do not reduce the
mathematical work; they only change scheduling and data movement.

## Experiments

The code is archived in
[the experiment patch](../../patches/ggml/experiments/tinyblas-cache-experiments.patch).
It is not present in the normal GGML source or applied by normal builds. See
[isolated reproduction instructions](../../patches/ggml/README.md).
In a patched export, the compile-time selector is
`src/ggml-cpu/llamafile/sgemm-experiment.h`; zero selects the upstream control.

1. **2 x N tile**: replace the AVX-512 `RM=4, RN<=6` tile with `RM=2,
   RN<=10`. This selects 2x6, 2x10, and two 2x8 tiles for 6, 10, and 16 token
   columns respectively.
2. **Software prefetch**: prefetch each A/weight row 2, 4, or 8 vector steps
   ahead. A vector step is 64 bytes, so the tested lead distances are 128, 256,
   and 512 bytes. B was not explicitly prefetched.
3. **Persistent four-row weight packing**: duplicate F32 weights into a layout
   that places the four 16-float row blocks used by one microkernel iteration
   consecutively. Packing happens on first use and is excluded from latency and
   steady-state counter regions.
4. **K blocking**: process K in blocks of 256 while keeping partial SIMD
   accumulators for all output columns in a stack array between blocks. This is
   the proposed “bring weights in before use, spill partial accumulators to L1”
   tradeoff.

All saved variants passed the pose, root, and decoder parity tests. The control
library and every experiment library were loaded through their ELF SONAME using
`LD_LIBRARY_PATH`; `LD_DEBUG=libs` was used to verify the selected file. This is
necessary because the profiling executable is directly linked to
`libggml-cpu.so.0`; the runtime backend-directory option does not override that
already-loaded library.

## Test system and method

- AMD Ryzen 9 7900, Zen 4, 12 physical cores / 24 threads
- Linux 6.18.38
- GCC 15.2.0, `RelWithDebInfo`, `-O3 -g -DNDEBUG`
- GGML Llamafile enabled; Zen 4-only CPU build; all-variants dispatch disabled
- repository `587ba93dae2a5e5d8066baac91036822a4612a7a`
- GGML `0a249dd4d2f4dfcd962123a3d881f73f93588ccf`
- one core pinned to CPU 2; two physical cores pinned to CPUs 2 and 3
- model `generated/g1-f32`, style `generated/styles/walk.mbstyle`
- forced 10-token plans unless otherwise stated; one independent request
- latency: five warmups followed by 50 rounds; table values are the median of
  three independently started process means in rotated variant order
- 6- and 16-token latency: five warmups, 30 rounds, three process means
- counters: ten warmups followed by 60 measured plans

The profiler's `MOTIONBRICKS_PROFILE_PERF_CONTROL` hook writes `enable` and
`disable` to a perf control FIFO immediately around measured rounds. Counter
runs used `perf stat -D -1 --control fifo:...`, so model loading, warmup, first
use packing, and the three post-benchmark trace plans are excluded.

The basic, load/store-stall, FP-resource-stall, L2, and fill-source groups were
run separately and each event was scheduled 100% of the time. The top-down
metrics were multiplexed (16% for most inputs and 33% for the SSE/AVX check), so
their percentages are directional. Stall events overlap: their percentages
must not be added together.

## Latency across token-column widths

One-core latency, median of three process means:

| variant | 6 tokens / 24 frames | change | 10 tokens / 40 frames | change | 16 tokens / 64 frames | change |
|---|---:|---:|---:|---:|---:|---:|
| upstream | 44.496 ms | - | 57.710 ms | - | 74.753 ms | - |
| 2 x N | 53.876 ms | +21.1% | 80.322 ms | +39.2% | 94.468 ms | +26.4% |
| prefetch 2 | 45.561 ms | +2.4% | 59.977 ms | +3.9% | 75.274 ms | +0.7% |
| prefetch 4 | 46.857 ms | +5.3% | 59.395 ms | +2.9% | 76.136 ms | +1.9% |
| prefetch 8 | 46.356 ms | +4.2% | 59.744 ms | +3.5% | 75.851 ms | +1.5% |
| packed | 52.701 ms | +18.4% | 66.040 ms | +14.4% | 84.919 ms | +13.6% |
| K-block 256 | 46.497 ms | +4.5% | 61.415 ms | +6.4% | 77.968 ms | +4.3% |

There is no hidden win at 6 or 16 columns. Prefetch distance 2 becomes nearly
neutral at 16 columns, but remains slower and does more work.

## One-core counters

### Execution efficiency

Counts cover 60 measured 10-token plans. Cycle counts in later tables differ
slightly because each counter family was collected in a separate run.

| variant | cycles (B) | cycle change | instructions (B) | instruction change | IPC |
|---|---:|---:|---:|---:|---:|
| upstream | 17.759 | - | 21.529 | - | 1.212 |
| 2 x N | 24.878 | +40.1% | 22.908 | +6.4% | 0.921 |
| prefetch 2 | 18.260 | +2.8% | 25.940 | +20.5% | 1.421 |
| prefetch 4 | 18.542 | +4.4% | 25.888 | +20.2% | 1.396 |
| prefetch 8 | 18.275 | +2.9% | 25.795 | +19.8% | 1.412 |
| packed | 20.631 | +16.2% | 21.827 | +1.4% | 1.058 |
| K-block 256 | 18.740 | +5.5% | 22.825 | +6.0% | 1.218 |

Prefetch's higher IPC is not a speedup: the core retires many additional
prefetch-loop/address/branch instructions while total cycles still rise.

### Resource stalls

Each cell is the raw event count in billions followed by event count divided by
cycles from the same run. The ratios describe event pressure, not exclusive
fractions of time.

| variant | load not complete | load queue | store queue | FP register file | FP scheduler |
|---|---:|---:|---:|---:|---:|
| upstream | 8.131 (46.1%) | 1.050 (5.9%) | 0.316 (1.8%) | 11.462 (64.6%) | 0.344 (1.9%) |
| 2 x N | 15.970 (64.5%) | 1.709 (6.9%) | 0.392 (1.6%) | 17.933 (72.9%) | 0.132 (0.5%) |
| prefetch 2 | 8.986 (48.5%) | 1.225 (6.6%) | 0.330 (1.8%) | 11.092 (61.0%) | 0.383 (2.1%) |
| prefetch 4 | 9.220 (49.2%) | 1.244 (6.6%) | 0.365 (1.9%) | 10.991 (60.5%) | 0.377 (2.1%) |
| prefetch 8 | 8.961 (48.2%) | 1.346 (7.2%) | 0.348 (1.9%) | 11.031 (60.2%) | 0.396 (2.2%) |
| packed | 10.376 (51.2%) | 1.035 (5.1%) | 0.348 (1.7%) | 13.172 (66.5%) | 0.671 (3.4%) |
| K-block 256 | 9.141 (47.7%) | 1.074 (5.6%) | 0.699 (3.6%) | 11.715 (61.9%) | 0.225 (1.2%) |

Absolute change from upstream makes the tradeoffs especially clear:

| variant | load-not-complete | load queue | store queue | FP-register stall | FP-scheduler stall |
|---|---:|---:|---:|---:|---:|
| 2 x N | +96.4% | +62.9% | +24.1% | +56.5% | -61.5% |
| prefetch 2 | +10.5% | +16.7% | +4.5% | -3.2% | +11.2% |
| prefetch 4 | +13.4% | +18.5% | +15.5% | -4.1% | +9.7% |
| prefetch 8 | +10.2% | +28.2% | +10.0% | -3.8% | +15.2% |
| packed | +27.6% | -1.4% | +10.0% | +14.9% | +94.9% |
| K-block 256 | +12.4% | +2.3% | +121.1% | +2.2% | -34.7% |

### Cache requests and fill sources

`ls_rd_blk_c` exactly tracks the difference between L2 data accesses and hits
to counter rounding, so it is shown as the L2 miss count. Fill-source events are
AMD raw-event counts, not byte counts.

| variant | L2 data accesses (B) | L2 hit rate | L2 misses (M) | local-L2 fills (B) | DRAM-I/O fills (M) |
|---|---:|---:|---:|---:|---:|
| upstream | 3.554 | 97.60% | 85.2 | 3.469 | 314.1 |
| 2 x N | 4.710 | 98.23% | 83.5 | 4.618 | 243.1 |
| prefetch 2 | 3.598 | 96.85% | 113.2 | 3.478 | 339.1 |
| prefetch 4 | 3.620 | 96.36% | 131.7 | 3.470 | 334.7 |
| prefetch 8 | 3.607 | 96.42% | 129.1 | 3.459 | 342.7 |
| packed | 3.360 | 97.40% | 87.3 | 3.260 | 535.1 |
| K-block 256 | 3.600 | 97.66% | 84.2 | 3.497 | 247.7 |

Relative to upstream:

| variant | L2 accesses | L2 misses | DRAM-I/O fills |
|---|---:|---:|---:|
| 2 x N | +32.5% | -2.0% | -22.6% |
| prefetch 2 | +1.3% | +32.8% | +8.0% |
| prefetch 4 | +1.9% | +54.5% | +6.5% |
| prefetch 8 | +1.5% | +51.5% | +9.1% |
| packed | -5.5% | +2.4% | +70.3% |
| K-block 256 | +1.3% | -1.2% | -21.2% |

### AMD top-down attribution

| variant | retiring | backend | backend: memory | backend: CPU | frontend | bad speculation |
|---|---:|---:|---:|---:|---:|---:|
| upstream | 20.6% | 76.3% | 50.0% | 26.3% | 2.6% | 0.4% |
| 2 x N | 15.7% | 82.0% | 65.2% | 16.8% | 1.8% | 0.3% |
| prefetch 2 | 23.2% | 73.5% | 51.5% | 22.0% | 2.5% | 0.4% |
| prefetch 4 | 23.0% | 73.9% | 52.3% | 21.6% | 2.5% | 0.4% |
| prefetch 8 | 22.8% | 74.1% | 52.2% | 22.0% | 2.5% | 0.4% |
| packed | 17.8% | 79.0% | 56.1% | 22.9% | 2.3% | 0.3% |
| K-block 256 | 20.6% | 75.4% | 51.4% | 24.1% | 3.4% | 0.3% |

No variant records meaningful SSE/AVX transition stalls.

## Two-core counters

The same behavior survives on two physical cores. Percent columns again show
event count divided by cycles from that event family's run.

| variant | cycles (B) | change | IPC | load not complete | load queue | store queue | FP register | FP scheduler |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| upstream | 20.273 | - | 1.067 | 49.7% | 5.3% | 2.6% | 63.9% | 1.7% |
| 2 x N | 26.575 | +31.1% | 0.866 | 64.0% | 5.9% | 2.0% | 70.5% | 0.5% |
| prefetch 2 | 21.185 | +4.5% | 1.229 | 51.7% | 6.3% | 2.5% | 60.5% | 1.7% |
| prefetch 4 | 21.239 | +4.8% | 1.223 | 51.2% | 6.8% | 2.6% | 60.6% | 1.7% |
| prefetch 8 | 21.189 | +4.5% | 1.222 | 51.4% | 7.2% | 2.5% | 59.5% | 1.7% |
| packed | 22.231 | +9.7% | 0.986 | 53.6% | 4.7% | 2.3% | 66.0% | 3.0% |
| K-block 256 | 21.294 | +5.0% | 1.076 | 49.7% | 5.2% | 4.2% | 62.3% | 1.1% |

## Interpretation by experiment

### 1. Smaller row tile

This is decisively worse. Although `2 x 10` has 20 accumulators versus 24 for
`4 x 6`, it also has ten live B operands and halves reuse of each B vector
across output rows. L2 hit traffic rises 32.5%, load-not-complete events nearly
double, IPC falls from 1.212 to 0.921, and FP-register-file stalls rise rather
than fall. The limiting quantity is the complete live set and load/reuse
schedule, not accumulator count alone.

### 2. Software prefetch

Prefetch slightly lowers FP-register stalls, but pays roughly 20% more retired
instructions and increases outstanding-load pressure. The load-queue penalty
grows with distance (+16.7%, +18.5%, +28.2%), while L2 misses grow 33-55%.
Hardware prefetch and out-of-order execution are already doing a better job for
this access stream. Explicit prefetch issues work too early or redundantly and
does not hide enough critical latency to repay its instruction cost.

### 3. Interleaved persistent weights

Packing does achieve one local goal: L2 access traffic falls 5.5% and load-queue
stall events fall 1.4%. It loses overall because DRAM-source fills rise 70.3%,
FP-register stalls rise 14.9%, and FP-scheduler stalls nearly double. The new
layout disrupts the original row-stream locality and duplicates a large working
set. Peak RSS rises from 747,216 KiB to 1,372,416 KiB: +625,200 KiB (+83.7%).

### 4. K-blocked partial accumulators

K blocking is the cleanest demonstration of the proposed cache trade. It cuts
DRAM-source fills 21.2%, L2 misses 1.2%, and FP-scheduler stalls 34.7%. However,
storing/reloading partial accumulator vectors more than doubles store-queue
stalls (+121.1%), adds 6.0% instructions, and still adds 5.5% cycles. The
baseline already consumes each weight stream effectively enough that forcing
partial sums through L1 costs more than the avoided long-latency fills.

## Decision

Keep upstream tinyBLAS `4 x 6`, do not add explicit A prefetch, do not persist
the interleaved weight copy, and do not K-block through an L1 accumulator array
for these 6-16-column Motion Bricks matrices on Zen 4.

The useful next direction is not another blind reduction in tile size. A viable
candidate must preserve the baseline's B reuse and avoid partial-sum stores,
for example an assembly-tuned `4 x N` schedule that shortens selected live
ranges without increasing the number of B loads. It should first demonstrate a
lower FP-register stall count *and* no increase in load-not-complete/L2 traffic
before a full latency sweep.

## Archive and normal-build verification — 2026-09-22

The patch was applied to a clean export of the pinned GGML revision and checked
to reproduce both experimental files byte-for-byte before restoring the
submodule. Normal builds do not apply this archive. The GGML worktree is clean;
the optional measured-region FIFO hook remains confined to the profiling tool.

A fresh stock-Debian CPU build using the restored submodule, with Llamafile
explicitly enabled, built the library and profiler. The 2,205-case G1 and
64-case SMPL neural parity fixtures both passed with zero token mismatches.
Compilation and execution were limited to eight cores. Local verification log:
`/tmp/mb-tinyblas-clean-check.log`. Existing experiment binaries are retained
as local artifacts, not selected by normal build configuration.
