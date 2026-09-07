# Choose your API

The library separates animation generation, control policy inference and
physics. None requires the web demo, Go, Kimodo or Python at runtime. You can
use MotionBricks without SONIC, SONIC without MotionBricks, or combine them.

## Models and abstraction levels

| Component | Level | Public header / entry point | Library does | You supply/manage |
|---|---|---|---|---|
| MotionBricks G1 | Stateless model pipeline | [`inference.h`](../include/motionbricks/inference.h), `mb_model_infer` | Normalization, root/duration network, pose sampling, VQ decoding, motion-representation decoding | Explicit source/target constraints, duration mask, seed; your controller, style choice, placement and seams |
| MotionBricks G1 | Optional input adapter | `inference.h`, `mb_inference_request_set_boundary_poses` | FK and conversion of local skeletal poses to model features | Two already-placed four-frame boundaries in a common canonical frame |
| MotionBricks G1 | Stateful animation controller | [`motionbricks.h`](../include/motionbricks/motionbricks.h), `mb_agent_*` | Context history, style sampling, spring targets, canonicalization, inference, world restoration and seam blending | Style, direction/facing, speed/target, seed, when to plan and advance |
| SONIC G1 mode 0 | Policy inference | [`sonic.h`](../include/motionbricks/sonic.h), `mb_sonic_encode/decode` | Reference encoder + FSQ tokens; observation decoder → actions | Reference observations, measured-state history, timing, motor scaling, controller and simulator |
| MuJoCo + SONIC | Stateful physics controller | [`physics.h`](../include/motionbricks/physics.h), `mb_physics_*` | Native-pose adaptation, observation history, SONIC calls, motor targets/PD torques, physics steps | SONIC handle, scene/config, reference motion and reference time |

The stateless MotionBricks API is **composed inference**, not an API for arbitrary
individual transformer tensors. Root/pose/VQ graph functions remain internal.
SONIC's encoder and decoder can be invoked separately. Its layer-output API is
diagnostic, not an alternate controller.

### Typical combinations

- Your animation controller, no physics: explicit constraints → `mb_model_infer`
  → render/export `mb_motion`. No style assets or agent needed.
- Our animation controller, no physics: `mb_style` + `mb_command` + `mb_agent`
  → `mb_motion`.
- Your simulator/controller: construct observations → `mb_sonic_encode/decode`
  → scale actions and apply your motor controller. No `mb_physics` needed.
- Our simulation: any compatible G1 motion, including imported animation →
  `mb_physics_step`. MotionBricks itself need not be loaded.
- Full stack: either MotionBricks interface → G1 motion → our SONIC/MuJoCo
  session. The web app is one consumer of this combination.

These are composable interfaces in **one library**, not separate link-time
packages. Each model is loaded only when you call its load function.

## Detailed guides

- [MotionBricks stateless inference, feature layouts and C example](API-INFERENCE.md)
- [MotionBricks stateful controller and common ownership rules](#stateful-motionbricks)
- [SONIC inference observations and optional physics](API-SONIC-PHYSICS.md)
- [Model/style formats](FORMATS.md), [sampling semantics](SAMPLING.md)

## Build and link only what you need

Use the [normal Linux build](../README.md#build). Physics is off by default:

```sh
cmake --preset release -DMOTIONBRICKS_ENABLE_PHYSICS=OFF
cmake --build --preset release
cmake --install build/release --prefix /your/install/prefix
```

Headers are installed under `include/motionbricks`; link `libmotionbricks`.
For example, using an installation whose library directory is `lib`:

```sh
cc consumer.c -I/your/install/prefix/include -L/your/install/prefix/lib \
  -Wl,-rpath,/your/install/prefix/lib -lmotionbricks -o consumer
```

Use your platform's actual library directory (`lib` or `lib64`). GGML/backend
shared libraries must also be discoverable by the runtime linker/backend
loader; `mb_runtime_options_set_backend_directory` specifies a backend directory.
With an in-tree build you can use `-Iinclude -Lbuild/debug` and an absolute rpath
to `build/debug`. Static consumers must link the C++ runtime and transitive
GGML dependencies too.

CPU and Vulkan use the same APIs; select with opaque `mb_runtime_options`
setters before loading each model. For a CPU-only build additionally set
`MOTIONBRICKS_ENABLE_VULKAN=OFF`. For SONIC-only use, set
`MOTIONBRICKS_DOWNLOAD_MODELS=OFF` to avoid the default MotionBricks bundle
download, then provide the separately converted SONIC GGUF.

Enable `MOTIONBRICKS_ENABLE_PHYSICS` only if using our MuJoCo controller;
[supply the SDK/scene/config as documented](SONIC-GGML.md#build-and-prepare-assets).
In a physics-enabled build MuJoCo is a native library dependency even if you
don't create a session. In disabled builds physics functions remain exported
but return `MB_BACKEND_UNAVAILABLE`. SONIC inference still works.

## Common C/FFI contract

- ABI version 1. New functions are additive; older binaries with the same ABI
  version may lack newer symbols. Require a library revision exporting the APIs
  you use; ABI version alone is not feature discovery.
- Opaque heap handles; no by-value structs, compiler-sized enums or C++ objects
  cross the ABI. Booleans/enum-like selectors are `uint32_t`, lengths `uint64_t`.
  This is suitable for PureGo and other FFIs without struct-layout knowledge.
- Calls return `mb_status`. Supply writable `char error[1024]` (or another
  capacity). Messages truncate safely and are NUL-terminated when capacity is
  nonzero. Success clears the buffer. `NULL,0` discards diagnostics; there is no
  global/thread-local `last_error`. Copy a message before reusing your buffer.
- C++ exceptions are caught at the C boundary. Null pointers, dimensions and
  numerical values are checked, but the caller must provide valid live handles
  and enough allocated memory for the declared capacity. Dangling pointers,
  double frees and lying about allocation sizes are not detectable safely.
- All constructors have matching `*_free`; freeing NULL is allowed. Do not free
  objects while calls are in flight. Output-motion buffers are borrowed until
  `mb_motion_free`; new results do not invalidate old motion handles.
- Motion layout: Y-up, metres, 30 FPS; roots `[frames,3]`, local XYZW rotations
  `[frames,34,4]`, frame-major. Use `mb_model_get_joint_*` for names/hierarchy and
  `mb_model_get_neutral_joint_position` for rest positions. G1Skeleton34 is not
  an arbitrary humanoid, SMPL-X or SOMA skeleton.
- Serialize operations sharing a MotionBricks model, including different
  agents and stateless requests using that model. Serialize mutable request,
  command, style and agent access. SONIC serializes inference internally;
  physics sessions require caller serialization. Use separate model handles
  for independently scheduled MotionBricks inference workers.

The existing [Go bindings](../bindings/go) are convenience wrappers for the
controller/physics demo, not an exhaustive wrapper of every exported C function.
The new stateless API can be bound directly with PureGo using pointer/scalar
arguments; it does not require cgo or copying any native struct layout.

## Stateful MotionBricks

Use this when you want the supplied animation-controller behavior:

1. `mb_model_load(bundle, options, ...)`.
2. `mb_style_load(model, style_path, ...)` and `mb_agent_create(model, ...)`.
3. `mb_agent_reset(agent, initial_style, ...)` to seed history, or supply your
   own history with `mb_agent_set_context` (at least four frames, 34 joints).
4. Create `mb_command`; set its style, movement/facing, target speed or world
   target, seed and optional diagnostic argmax.
5. `mb_agent_plan` returns an owned `mb_motion`; read its borrowed buffers.
6. Call `mb_agent_advance` as your playback advances, then plan again when needed.

The model must outlive its agents. A command borrows its style; keep styles
alive while referenced by commands/agents. Runtime options are copied during
model loading and may be freed afterwards. Motions own their data independently.
Target getters on controller-produced motions expose the four placed style
target poses. Negative command speed selects the style's configured speed.

Replacing context or commands does **not** turn this into inference-only mode:
the controller still constructs targets and applies its seam. Use `inference.h`
to own those decisions. See [demo integration](DEMO.md) for scheduling examples;
its WebSocket protocol is an application interface, not the C ABI.
