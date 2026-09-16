# SONIC inference and optional physics

See [API selection](API.md) for the full stack. SONIC is a motion-tracking
policy, not the MotionBricks animation generator. You can load SONIC without
loading MotionBricks or using any of our animation/controller code.

## Level 1: SONIC policy inference only

Header: [`motionbricks/sonic.h`](../include/motionbricks/sonic.h).

1. Configure opaque `mb_runtime_options` (CPU/Vulkan, threads, backend directory).
2. `mb_sonic_load(gguf, options, &sonic, error, capacity)`.
3. `mb_sonic_encode(sonic, observations, 1762, tokens, 64, ...)`.
4. Put the 64 tokens at the start of your decoder observation, followed by
   measured-state/action history, and call
   `mb_sonic_decode(sonic, observations, 994, actions, 29, ...)`.
5. `mb_sonic_free` when no calls or dependent physics sessions remain.

For independent requests, callers can use `mb_sonic_encode_batch` and
`mb_sonic_decode_batch`. The caller supplies an explicit batch size from 1 to
`MB_SONIC_MAX_BATCH` (64); there is no internal queue or batching delay. Input
and output buffers are tightly packed in request-major order, and their counts
are the total float counts across the batch. For example, a four-request encode
uses 7,048 observation floats and 256 token floats. A four-request decode uses
3,976 observation floats and 116 action floats. Graph scratch is created lazily
and cached for each used batch size, while the model weights remain single-copy.
Calls sharing a SONIC handle remain serialized, so submit one populated batch
instead of four concurrent single-request calls when throughput is the goal.

These are synchronous flat-F32 calls with caller-owned buffers. There is no
simulation, control timer, motor scaling, PD controller, history update or
MotionBricks request inside them. Decode does not implicitly reuse the last
encoder result: **you supply the tokens in the decoder observation**. You can
also supply compatible tokens from elsewhere and call only decode.

Each item follows the pinned original-release **G1 mode 0**, F32 contract; the
batch API only aggregates independent items for execution. Other encoder modes
and incompatible model identities are rejected. Encoder observations must have
element 0 equal to zero. All inputs, including ignored encoder elements, must be
finite with magnitude <=1e6; invalid data is rejected, not clamped.

### Encoder observation: 1,762 floats

Offsets below are zero-based, half-open. Initialize unused fields to zero.

| Offsets | Shape | Mode-0 meaning |
|---|---|---|
| `[0,1)` | 1 | Mode selector, exactly 0 |
| `[4,294)` | `[10,29]` | Absolute reference hinge positions, radians, Isaac ordering |
| `[294,584)` | `[10,29]` | Reference hinge velocities, radians/sec, same ordering |
| `[601,661)` | `[10,6]` | Reference base orientation relative to current physical base |
| All other offsets | — | Unused by the converted mode-0 branch |

Orientation is the first two columns of a 3x3 rotation matrix, **row-interleaved**:
`[R00,R01,R10,R11,R20,R21]`. This differs from MotionBricks' column-concatenated
6D representation. Use the upstream reference/physical heading-alignment
convention, not arbitrary world rotations.

The released export packs each encoder row from 58 consecutive values starting
at `4 + 58*t`, followed by six values at `601 + 6*t`, for t=0..9. This grouping
is intentionally preserved even though the public position and velocity blocks
are separately frame-major. **Do not “correct” the observation by interleaving
q/dq per frame**; provide the above public layout. The encoder then runs the MLP
and FSQ32, returning 64 quantized float token values (not integer codebook IDs).

Our physical adapter samples ten reference poses at current reference time plus
`0,.1,...,.9` seconds and computes velocities using a .02-second forward step.
Past the reference end it holds the final pose and supplies zero velocities.
The encoder itself neither samples motion nor computes these observations.

### Decoder observation: 994 floats

| Offsets | Shape | Meaning |
|---|---|---|
| `[0,64)` | 64 | FSQ encoder token values |
| `[64,94)` | `[10,3]` | Measured base angular-velocity history |
| `[94,384)` | `[10,29]` | Joint position minus configured default position |
| `[384,674)` | `[10,29]` | Joint velocity |
| `[674,964)` | `[10,29]` | Previous unscaled decoder actions |
| `[964,994)` | `[10,3]` | Gravity projected into the base frame |

Each history block is oldest to newest, ten 50 Hz controller samples, flattened
frame-major. The newest sample includes current measured state and the previous
tick's action. Rotations/angular velocities/gravity use the physical model's
coordinate conventions, not the browser's Y-up skeletal coordinates. Match
upstream startup padding if reproducing its controller; our adapter initializes
q/dq/angular/actions to zero and gravity to `[0,0,1]`, then appends observations.

The 29-dimensional action output uses Isaac joint ordering. It is neither a
torque nor an absolute joint angle. Our adapter computes
`target = default_position + action_scale * action`, reorders to hardware
joints, then applies configured PD control and torque limits. Consumers using
their own controller must supply these model-specific constants and mappings.

### Joint ordering

The Isaac-order indices 0..28 map to hardware-order indices:

```text
0,6,12,1,7,13,2,8,14,3,9,15,22,4,10,16,23,5,11,17,24,18,25,19,26,20,27,21,28
```

Hardware order is:

```text
left:  hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
right: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
waist: yaw, roll, pitch
left:  shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw
right: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw
```

This is not MotionBricks' 34-joint tree order. Do not directly place skeletal
quaternions in SONIC's 29 hinge-coordinate buffers. Our optional physics layer
performs that adaptation with the verified robot configuration; a custom
controller must implement the equivalent conversion.

`mb_sonic_layer` copies cached preactivations from the last successful inference:
encoder layers 0..4 or decoder layers 0..6. NULL/zero queries the required float
count. Batched traces are request-major and include the batch dimension in that
count. It is intended for parity/debugging and does not advance any controller.
Inference calls are serialized per SONIC model, but a multi-call encode/decode/
trace sequence is not an atomic transaction: serialize at the caller if you
need traces from a particular request.

See [SONIC validation and limitations](SONIC-GGML.md) and
[`src/sonic_physics.cpp`](../src/sonic_physics.cpp) for the tested observation,
heading-alignment and motor implementation. These are model-specific contracts;
substituting a robot, materials or controller does not inherit tracking parity.

## Level 2: SONIC + MuJoCo controller

Header: [`motionbricks/physics.h`](../include/motionbricks/physics.h).
Optional native MuJoCo 3.12.0 dependency, enabled at build time. A build against
3.3.5 is retained for reference comparisons; see [SDK setup](MUJOCO-UPGRADE.md).
`mb_physics_engine_version(buffer, capacity, error, error_capacity)` copies the
loaded engine's NUL-terminated version into a caller-owned buffer (32 bytes is
sufficient). It returns backend-unavailable without physics, and rejects a
header/runtime ABI mismatch before any model structures are accessed. The Go
binding exposes `Physics.EngineVersion()`; `/api/stream` reports `mujoco_version`.

1. Load SONIC, then `mb_physics_create(sonic, scene_xml, config, &session, ...)`.
   The config contains the verified skeleton/motor mapping and controller data;
   use the scene/config pair prepared by the reference tools, not arbitrary XML.
2. `mb_physics_start` accepts one G1Skeleton34 root `[3]` and local XYZW pose
   `[34,4]`. This initializes state once. Replacing reference motion later does
   not reset physical state.
3. `mb_physics_step` takes a full source clip (roots `frames*3`, rotations
   `frames*136`) plus source time in seconds. Each call advances one 50 Hz
   controller tick and four 200 Hz physics substeps, irrespective of wall time.
4. Read physical/reference positions, status, optional collision shapes/traces.
5. `mb_physics_reset` explicitly clears state/history; call start again.
6. Free the session before its SONIC model.

Input motion is 30 FPS, Y-up, metres, with local XYZW rotations. It can come from
either MotionBricks API or any correctly converted G1 animation. Source time
chooses the reference pose; physics time advances independently with calls.
At source end the final reference pose is held. Your scheduler decides whether
to hold, replace the reference, or continue another animation/controller.

Step output buffers each contain 90 floats: 30 physical-body joint XYZ positions
in Y-up IsaacLab body order. `mb_physics_skeleton` returns their 30 parents.
They are **not** 34 local skeletal rotations and cannot be passed straight back
as MotionBricks context. Reference output is computed using physical FK; actual
output comes from the simulated bodies.

`mb_physics_status` returns simulation time, detected contact count and sticky
fall status. Falling does not pause the simulation. Numerical safety failures
return errors; callers decide their UI/recovery policy. Continue ticking after
a fall if you want to observe the physical result, or explicitly reset.

`mb_physics_trace` copies observation, action, motor and torque buffers from the
last completed tick. Field IDs/layouts are documented in the header. NULL/zero
queries sizes. This is a diagnostics API, not a torque-input stepping API.

Collision APIs return definitions once and actual world transforms after
stepping. See [collision geometry](COLLISION-GEOMETRY.md) for local/world bases,
convex hulls, foot soles and interpolation. No renderer is required.

Each session owns MuJoCo state/history and borrows its SONIC model. Callers must
serialize session operations and destruction. The physics interface is a
**bundled controller**, not general-purpose MuJoCo bindings: there is no public
function to inject arbitrary torques, change PD gains in-place or replace its
observation builder. To own those decisions, use level-1 SONIC inference and
your own simulation loop.
