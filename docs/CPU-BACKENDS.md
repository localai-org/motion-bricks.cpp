# CPU backend selection and packaging

Normal builds enable `MOTIONBRICKS_CPU_ALL_VARIANTS=ON`. This sets GGML's
`GGML_CPU_ALL_VARIANTS=ON`, `GGML_BACKEND_DL=ON`, `BUILD_SHARED_LIBS=ON` and
`GGML_NATIVE=OFF`. GGML builds the variants it supports for the **target
architecture**; this is not a cross-architecture universal binary. On x86-64
with GCC, the pinned revision builds 14 variants, including baseline x64,
Haswell/AVX2, Skylake-X/AVX512, Zen 4 and Sapphire Rapids. Other architectures
use GGML's own supported variant lists; unsupported targets can opt out.

At first model load, GGML scores the packaged modules against CPU features
and loads the highest-scoring compatible backend. MotionBricks initializes
CPU inference through GGML's device registry and sets the requested thread
count through the selected backend's function table. `MB_DEVICE_AUTO` still
selects CPU. Vulkan also initializes through the registry, with the existing
F32 parity settings applied before discovery.

Discovery searches beside the loaded `libggml` library (DLL directory on
Windows). This works for C/C++ executables, a statically linked MotionBricks
archive using shared GGML, and Python/Go loading the shared library from an
unrelated working directory. `mb_runtime_options_set_backend_directory` can
supply an explicit directory **before the first model load** instead.
Selection is process-wide and serialized across concurrent loads; changing
this option after successful discovery does not replace an active backend.
A missing/incompatible CPU module returns `MB_BACKEND_UNAVAILABLE`; discovery
can be retried with a corrected path.

## Build and install

For CPU-only inference with profiling symbols and `-O3`:

```sh
cmake --preset sonic-cpu
cmake --build --preset sonic-cpu
ctest --preset sonic-cpu
cmake --install build/sonic-cpu --prefix "$PWD/dist/motionbricks"
```

This preset uses local SONIC weights and fixtures and never downloads them.
The normal `debug` and `release` presets also enable runtime selection;
Release uses CMake's optimized compiler flags (`-O3` with GCC/Clang), while
Debug retains debug optimization settings.

Distribute the installed tree, including `libmotionbricks`, `libggml`,
`libggml-base` and the `libggml-cpu-*` modules, plus enabled GPU modules.
On Unix, libraries and modules install together under the configured
`CMAKE_INSTALL_LIBDIR`; executables are under `CMAKE_INSTALL_BINDIR`.
Relative runtime library search paths allow the default install layout to
be relocated. Windows DLLs and modules install under `bin`. The installed
MotionBricks static archive still requires its GGML dependencies; it is
not a standalone archive.

For a deliberately single-variant build, configure
`-DMOTIONBRICKS_CPU_ALL_VARIANTS=OFF` and explicit GGML ISA options. The
`sonic-cpu-avx512` preset does this to reproduce the historical measurements
and support the optional diagnostic interposer. It requires an AVX512 CPU.
The interposer is not part of the default build or installed runtime.

## Validation

The Linux `motionbricks-cpu-backend-loading` CTest uses the public C ABI in
fresh Python processes without third-party packages. It checks highest-score
selection, concurrent and repeated model loads, missing-directory recovery,
and an explicitly packaged baseline x64 fallback. It runs from a temporary
working directory. The same test can validate an installed tree:

```sh
python3 tests/backend_loading_test.py \
  --library dist/motionbricks/lib/libmotionbricks.so \
  --model generated/sonic/ggml/sonic-g1.gguf
```

Use `lib64` in this command if that is the configured install library directory.

SONIC parity tests additionally compare all 2,205 captured samples, including
intermediate layers, exact FSQ tokens and actions, at one and two threads.

### Local results: 2026-09-10

On the Ryzen 9 7900, the default build selects `ggml-cpu-zen4`. A fresh
comparison against the single-variant AVX512 build preserves the measured
performance (median of three independent run means):

| Build | One physical core | Two physical cores |
|---|---:|---:|
| Runtime selection, Zen 4 | 0.924 ms | 0.616 ms |
| Single AVX512 variant | 0.925 ms | 0.617 ms |

Each run measures 2,205 encoder+decoder pairs after 100 warmups, with CPU
affinity restricted to logical CPU 2 or CPUs 2,3 (distinct physical cores).
Both builds use GCC 15.2.0, F32 and `-O3`; model loading and backend discovery
are outside the timing interval. These differences are within run-to-run
variation. No pairs exceeded 20 ms. Full per-run statistics and asset hashes
are in [the dispatch benchmark](benchmarks/sonic-cpu-dispatch-2026-09-10.json).
Use `reference/profile_sonic_cpu.py` with each preset's library to reproduce;
the [profiling report](SONIC-CPU-PROFILE.md) describes the measurement limits.

Validation passed:

- All five `sonic-cpu` preset tests. Both optimized CPU parity tests report
  maximum absolute error `1.52588e-5`, relative L2 `7.5854e-7`, and zero token
  mismatches over all 2,205 captures.
- 27 debug tests, including MotionBricks CPU/Vulkan inference and parity,
  SONIC CPU/Vulkan parity, physics API and backend loading. Command:
  `ctest --preset debug --exclude-regex 'purego|go-demo|open-loop' --parallel 1`,
  with CPU affinity 2,3 and `OMP_THREAD_LIMIT=2`. The local debug configuration
  enables physics and hardware-tagged Vulkan parity. Go/browser and long
  open-loop replay tests were excluded from this focused validation.
- All four `sonic-cpu-avx512` preset tests with variant selection disabled.
- Installation to a temporary prefix followed by relocation: the Python
  backend-loading test and a separately linked consumer of the installed
  MotionBricks static archive both locate and use the packaged Zen 4 backend
  from an unrelated working directory.

These checks ran on Linux x86-64. Baseline fallback was exercised on the same
host; other CPU hardware, Windows and macOS were not available for testing.
