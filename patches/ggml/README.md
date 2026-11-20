# GGML patches

Normal builds use the pinned `ggml/` submodule without applying patches. The
current pin is `0a249dd4d2f4dfcd962123a3d881f73f93588ccf`.

## Archived experiments

[experiments/tinyblas-cache-experiments.patch](experiments/tinyblas-cache-experiments.patch)
preserves the September 16 tinyBLAS tile, prefetch, packed-weight and K-block
experiments, including the selector header. None improved the tested workload;
see the [results](../../docs/benchmarks/TINYBLAS-CACHE-EXPERIMENTS-2026-09-16.md)
and [raw measurements](../../docs/benchmarks/tinyblas-cache-experiments-2026-09-16.json).
This patch is **not applied by CMake or any normal build script**.

To reproduce an experiment, apply it only to a separate export and use a separate
build directory. From the repository root:

```sh
experiment_source=$(mktemp -d /tmp/motionbricks-tinyblas.XXXXXX)
experiment_patch="$PWD/patches/ggml/experiments/tinyblas-cache-experiments.patch"
git -C ggml archive 0a249dd4d2f4dfcd962123a3d881f73f93588ccf | tar -x -C "$experiment_source"
(cd "$experiment_source" && git apply --check "$experiment_patch" && git apply "$experiment_patch")
# Edit $experiment_source/src/ggml-cpu/llamafile/sgemm-experiment.h:
# 0=control, 1=2xN tile, 2=prefetch, 3=packed weights, 4=K-blocking.
# Prefetch experiments additionally set distance to 2, 4 or 8.
cmake -S . -B build/tinyblas-experiment -DCMAKE_BUILD_TYPE=Release \
  -DMOTIONBRICKS_GGML_SOURCE_DIR="$experiment_source" \
  -DMOTIONBRICKS_DOWNLOAD_MODELS=OFF -DGGML_LLAMAFILE=ON
cmake --build build/tinyblas-experiment --target motionbricks-cpu-profile -j8
```

The patch's default selector is zero. Compiler/ISA, CPU affinity, warmup and
measurement settings needed to reproduce the recorded results are in the
experiment report. In particular, its tile/packing variants target AVX-512.
The packed-weight cache is diagnostic only: it assumes immutable allocations
and retains them by address; do not use it in a normal model-loading service.

The optional `MOTIONBRICKS_PROFILE_PERF_CONTROL` FIFO hook is confined to the
explicitly built `motionbricks-cpu-profile` executable (`EXCLUDE_FROM_ALL`).
It does nothing unless that environment variable is set and does not affect
the library, demo or inference API.

Old experimental binaries remain local artifacts; do not point production
`LD_LIBRARY_PATH` or backend paths at them. Restore/rebuild normal binaries from
the clean submodule if reusing a build previously used for experiments.
