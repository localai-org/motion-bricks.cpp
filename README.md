# motion-bricks.cpp

A C++23/GGML port of NVIDIA MotionBricks for CPU and Vulkan inference, with a
stable C ABI suitable for PureGo.

The released batch-one G1 inference path is implemented end to end: strict
GGUF loading, root/duration planning, pose-token prediction, VQ decoding,
418/414/413 feature conversion, style alignment, and skeletal animation
output. CPU and Vulkan use the same public API and preserve the same duration
and pose-token decisions in the reference suite.

The demo also offers optional native MuJoCo physics driven by **GGML SONIC**,
including Kimodo animation playback, with target/reference/physical skeletons.
The original G1 mode-0 policy passes independent CPU/Vulkan layer parity.
See [SONIC setup, results and limitations](docs/SONIC-GGML.md) and the
[MuJoCo 3.12 upgrade notes](docs/MUJOCO-UPGRADE.md).
The browser sends WebSocket commands and renders buffered server-owned motion;
see [streaming and client QA](docs/STREAMING.md).

## Build

The normal build uses CMake and does not depend on Nix:

```sh
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

Configuration downloads and SHA-256-verifies the published 0.73 GB G1 F32
GGUF and style bundles into `generated/` when they are not already present.
The same operation can be run explicitly:

```sh
python scripts/download_gguf_weights.py
```

For an offline or source-only build, preserve an existing local bundle or use
`cmake --preset debug -DMOTIONBRICKS_DOWNLOAD_MODELS=OFF`. The repository and
revision are configurable with `MOTIONBRICKS_MODEL_REPOSITORY` and
`MOTIONBRICKS_MODEL_REVISION`.

On NixOS, enter the reproducible development shell first:

```sh
nix develop
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

When the pinned `ggml/` submodule is present, it is included automatically.
The non-neural ABI and validation subset can also be built without GGML:

```sh
cmake -S . -B build/debug -G Ninja -DMOTIONBRICKS_ENABLE_GGML=OFF
```

The Go binding and demo use PureGo to load `libmotionbricks` at runtime; they
do not use cgo or `import "C"`. Once the native shared library has been built,
the Go components therefore need no C compiler and can be built with cgo
explicitly disabled:

```sh
cd demo
CGO_ENABLED=0 go build -o ../build/debug/bin/motionbricks-demo .
```

`CGO_ENABLED=0` is optional but recommended for making this property explicit
in builds and CI. It affects only the Go build—the native C++ library is still
built separately with CMake.

The sanitizer lane is:

```sh
cmake --preset asan-ubsan
cmake --build --preset asan-ubsan
ctest --preset asan-ubsan
```

## Current ABI

Start with the [API selection guide](docs/API.md): choose stateless MotionBricks
inference, the optional animation controller, standalone SONIC inference, or
the SONIC/MuJoCo physics controller. The [inference-only C example](examples/inference.c)
uses caller-supplied poses without styles or an agent.

The installed C API uses only fixed-width scalars, pointers, and opaque heap
handles. Callers never reproduce a C or C++ structure layout. All constructors
have matching free functions, and no C++ exception crosses the ABI boundary.

The current CLI can report ABI information:

```sh
./build/debug/bin/motionbricks-cli abi
```

After producing the trusted safetensors intermediates described in
`reference/README.md`, build and inspect an F32 runtime bundle with:

```sh
python scripts/convert_to_gguf.py \
  --safe-directory generated/safe \
  --output generated/g1-f32
./build/debug/bin/motionbricks-cli inspect generated/g1-f32
```

The released G1 inference path contains exactly **183,148,382 learned F32
parameters**. The bundle loader validates the upstream revision, source
checkpoint identities, component roles, tensor counts, parameter counts,
anchor shapes, and the 34-joint parent topology before accepting a model.

Convert the 15 original demo styles with:

```sh
python scripts/convert_styles.py \
  --safe-directory generated/safe \
  --output generated/styles
```

At runtime the high-level flow is: load one immutable model, load one or more
`.mbstyle` assets, create an agent, reset it from an initial style (or supply
at least four frames of G1 context), set movement/facing/style on a command,
then call `mb_agent_plan`. The returned motion owns row-major F32 root
translations `[frames,3]` and local XYZW rotations `[frames,34,4]`. Call
`mb_agent_advance` as playback progresses so replanning uses the generated
motion as its next context.

The current implementation covers original preprocessed G1 styles. Direct
Kimodo GLB-to-`.mbstyle` conversion remains subsequent integration work.

## Weights

Ready-to-run native weights and all 15 upstream style primitives are published
as [MotionBricks-G1-GGML](https://huggingface.co/LocalAI-io/MotionBricks-G1-GGML)
under the Hugging Face `LocalAI-io` organisation. NVIDIA currently distributes
MotionBricks checkpoints through Git LFS in
[`NVlabs/GR00T-WholeBodyControl`](https://github.com/NVlabs/GR00T-WholeBodyControl/tree/a0732b642c0333077e127a2f56ab0014c196bca4/motionbricks),
not a separate Hugging Face model repository, so the model card links to that
pinned upstream revision. The downloader verifies the version-controlled
distribution manifest before accepting any file.

The default build is pinned to Hugging Face commit
`cc2a47603dbc203a4f18f35dd06ed3611833f506` rather than the mutable `main`
branch.

## Interactive demo

The initial Go/Three.js demo renders the model's 34-joint skeleton alongside
the four actual placed target-keyframe ghosts. It lets you steer relative to
the current camera yaw with W/A/S/D,
turn facing with the arrow keys, orbit/zoom the camera, and switch among the
converted upstream styles. It uses the reusable PureGo binding and the same
stateful native agent as other applications. See the
[demo guide](docs/DEMO.md) for build, run, architecture, and headless-Chromium
test instructions.

Pose tokens now use upstream-style Gumbel sampling by default. An explicit
argmax mode remains available for diagnostics; [sampling validation](docs/SAMPLING.md)
uses shared random draws rather than assuming identical framework RNG seeds.

The observational-parity tools can also turn the accepted upstream session
into one immutable `.mbreplay` file consumed by the C++ CLI and the Three.js
replay-only mode, plus a 345-frame MuJoCo reference MP4. See the
[reference guide](reference/README.md#target-trace-and-cross-runtime-visual-replay).

The parity gate converts all 14 captured replans into an independent open-loop
fixture, runs each through the real planner, writes aggregate, per-plan, and
optional neural-boundary reports, and presents an upstream/native overlay with
a joint-error scrubber. Strict CPU parity now passes all 14 plans against an
upstream PyTorch CPU replay of the exact accepted sparse inputs. The original
CUDA capture remains a separate cross-device diagnostic because near-tie pose
argmax choices can differ across CPU and CUDA arithmetic. See
[open-loop native parity](reference/README.md#open-loop-native-parity).

The opt-in CPU/Vulkan gate also passes all 14 plans on the recorded NVIDIA RTX
5070 Ti configuration. It uses exact durations and bounded public animation
errors rather than requiring bit-identical floating-point intermediates.

## Design

- [Human-led design](docs/motions-bricks.md)
- [Implementation sketch and plan](docs/IMPLEMENTATION.md)
- [Versioned formats](docs/FORMATS.md)
- [Go/Three.js demo](docs/DEMO.md)
- [Observational motion QA](docs/MOTION_QA.md)
- [Pinned upstream reference](reference/README.md)

## License

motion-bricks.cpp source code is licensed under the
[Apache License 2.0](LICENSE). NVIDIA's original model weights and converted
GGUF/style distributions remain under the NVIDIA Open Model License reproduced
with the published model. It permits derivative models and redistribution with
conditions including retention of the agreement and attribution, Trustworthy
AI terms, and trade compliance. Bundled third-party components retain their own
licenses; the vendored Three.js files are covered by
`demo/web/vendor/THREE-LICENSE.txt`.
