# MotionBricks: inference without our controller

See [API selection](API.md). Public contract:
[`motionbricks/inference.h`](../include/motionbricks/inference.h).

`mb_model_infer` is a stateless wrapper of the existing composed inference path.
It performs normalization, root/duration inference, pose-token inference and
sampling, VQ decoding, and conversion to local joint rotations/root translations.
It does not select a style, position targets with springs, maintain history,
blend seams, remove overlap frames, or place a clip back into your world.

## Pose-level inputs

Create `mb_inference_request`, then call `mb_inference_request_set_boundary_poses`
twice: boundary 0 for four source frames and boundary 1 for four target frames.
Each accepts exactly 12 root floats and 544 local XYZW quaternion floats.
The four target frames describe the **end** of the generated clip, not frames
4–7 immediately after the source. All frames are sampled at 30 FPS.

You choose which poses to use, their placement and their heading. Both boundaries
must use the same canonical frame. To follow the controller's convention:

1. Save the first source root's world X/Z and yaw.
2. Translate X/Z by that origin and rotate both boundaries about Y by negative
   yaw; leave world height unchanged. Premultiply each root quaternion by that
   inverse-yaw rotation. Leave non-root local quaternions unchanged.
3. Encode/infer in that common frame.
4. Apply the saved yaw and X/Z translation to output roots and root rotations
   when placing the result in your world. The API does not pin output poses
   exactly to constraints or apply the controller's additional seam alignment.

The helper performs FK using the loaded G1 skeleton. As upstream does for
MotionBricks context, virtual hand/toe endpoints retain their FK positions but
have global-identity conditioning orientations. It does not alter your input
arrays. Source outgoing velocity at slot 3 is unknown, so that local-root mask
is disabled. The target's final velocity repeats its preceding velocity.
Calling the helper resets the selected half's feature masks; apply custom masks
**afterwards**. Duration and sampling settings are unchanged.

Use already-compatible G1Skeleton34 poses, not arbitrary GLB joint orders.
Finite values and quaternion norms are validated. Physical feasibility and
in-distribution motion quality are not guaranteed by input validation.

## Raw feature inputs

For full control, supply all three fields with
`mb_inference_request_set_features`. No pose helper or style is needed.
Getters copy values back; `data=NULL, capacity=0` queries the required count.
Counts are elements, not bytes. Setters require exact counts; getters permit
larger buffers. Failed setters leave the request unchanged.

All fields contain eight boundary slots: source frames 0–3, followed by target
frames 0–3. During inference those target slots are placed at output frames
`N-4 .. N-1`, where N is the selected duration.

| Field | Float count | Per-slot layout (raw, unnormalized) |
|---|---:|---|
| `MB_INFERENCE_GLOBAL_ROOT` | 40 | `[x, height, z, cos(yaw), sin(yaw)]` |
| `MB_INFERENCE_LOCAL_ROOT` | 32 | `[yaw_velocity, x_velocity, z_velocity, height]` |
| `MB_INFERENCE_POSE` | 2424 | 99 position values followed by 204 orientation values |

Yaw is radians about +Y; +Z is forward at yaw zero. Velocities are radians/sec
or metres/sec, using 30 FPS forward differences within each four-frame block.
Despite the name “local root”, X/Z velocity is in the common canonical axes,
not a separate body-heading frame.

The 303-value pose row is:

- 33 non-root joints, in model order, each `[joint.x-root.x, joint.y,
  joint.z-root.z]`. **Y is absolute height, not root-relative.**
- All 34 joints (root first), each global rotation's first two columns:
  `[R00,R10,R20,R01,R11,R21]`. These are global orientations, not local
  quaternions and not row-interleaved matrix columns. Endpoint convention is
  as described above for context; raw inputs are not silently rewritten.

The pipeline normalizes features using the bundle's support statistics. Do not
normalize them yourself. It subtracts first-source X/Z for root prediction and
adds that translation back during reconstruction; it does not rotate away yaw.
Use canonical inputs as described above for the released model's convention.

All fields must be supplied even if some slots are masked out; fill unused
values with finite zeros. Values with magnitude above 1e4 are rejected, not
clamped. Enabled global-root slots require unit cosine/sine pairs (squared
norm tolerance .02). Slot zero must be enabled as the reconstruction anchor.
These bounds prevent pathological numerical inputs; they are not quality limits.

## Masks, duration and sampling

`set_mask` / `get_mask` use flat `uint32_t` 0/1 arrays:

- Feature fields: eight entries each. Defaults are all enabled except
  local-root slot 3.
- `MB_INFERENCE_DURATIONS`: eleven entries. Index i allows `24 + 4*i` output
  frames (6+i model tokens); defaults all enabled. At least one is required.

To request exactly 40 frames, enable only duration index 4. Otherwise the root
model chooses the allowed duration with the largest logit. Duration choice is
always argmax; the sampling option applies to pose tokens only.

`set_seed` / `get_seed` manage a uint64 seed. `set_sampling_argmax(...,1)` enables
diagnostic argmax; zero (default) uses Gumbel temperature 1. The RNG restarts
from the request seed on every call. Repeated identical inputs on the same
backend reproduce results; seed equality does not mean matching PyTorch's RNG
or guarantee identical decisions across different hardware. See [sampling
validation](SAMPLING.md) for stored-noise and numerical parity details.

## Output and ownership

`mb_model_infer(model, request, &motion, error, capacity)` returns `mb_motion`.
It leaves the request unchanged; no agent, command, style or physics handle
is consulted. Both request and model may be freed once the call completes.
The returned motion remains valid until `mb_motion_free`.

Use the existing motion getters for frame count, roots and local XYZW rotations.
Output is 24–64 frames in multiples of four, including source/target boundary
regions. Choose overlap removal, blending, world placement and scheduling in
your controller. Constraints condition the model; they are not guaranteed exact
pose pins. Controller target-metadata getters return zero frames/empty buffers
for these results; keep your own requested target poses for display.

There is no hidden playback state or retained random stream. Calls sharing a
model must still be serialized because its backend is shared. This API is
stateless with respect to animation/control, not a promise of thread safety.

## C example

[`examples/inference.c`](../examples/inference.c) implements a checked, reusable
`infer_between` function using only installed C headers. It accepts an already
loaded model and eight canonical input frames: four source then four target.
It demonstrates a fixed 40-frame duration, ownership and cleanup. No style or
agent is involved. The example is compiled by the normal test build.

```c
#include <motionbricks/inference.h>
/* After creating/configuring runtime options and loading your model: */
mb_inference_request *request = NULL;
mb_motion *motion = NULL;
char error[1024];
/* Check every status in real code; see the complete example. */
mb_inference_request_create(&request, error, sizeof error);
mb_inference_request_set_boundary_poses(request, model, 0,
    source_roots, 12, source_local_xyzw, 544, error, sizeof error);
mb_inference_request_set_boundary_poses(request, model, 1,
    target_roots, 12, target_local_xyzw, 544, error, sizeof error);
mb_model_infer(model, request, &motion, error, sizeof error);
mb_inference_request_free(request);
/* Read borrowed motion buffers; render, export, or pass them to physics. */
mb_motion_free(motion);
```

## Validation

`motionbricks-inference-api` is a pure-C, weight-free ABI rejection/ownership
test. With a local reference bundle/style available,
`motionbricks-inference-model` tests both sampling modes, min/max durations,
sparse masks, repeatability, pose conversion and result lifetime. The Vulkan
variant is enabled with the existing Vulkan-test option.

Exact equality against the pre-existing internal `run_transition` boundary
guards the wrapper against changing inference semantics. This is **not a new
independent upstream parity measurement**: independent neural/controller parity
remains in the existing reference suites. No planner arithmetic is replaced.

`motionbricks-inference-fuzz` exercises the new C input boundary under
ASan/UBSan. Build with Clang, `MOTIONBRICKS_SANITIZE=ON` and
`MOTIONBRICKS_ENABLE_FUZZERS=ON`. GGUF loading is excluded; optionally provide
`MOTIONBRICKS_FUZZ_BUNDLE` to load a trusted local model once before fuzzing
inference/pose inputs. Normal tests never download weights themselves.
