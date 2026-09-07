# Collision geometry overlay

Enable **Live physics**, then **Show collision geometry**. The optional overlay
uses surfaces at 60% opacity with subtle outlines. It follows the physical
robot, including its uncontrolled hand bodies, and continues after a fall.
Turning physics off hides it. It does not change simulation or motor control.

The geometry comes from the compiled MuJoCo model, not from skeleton joints or
decorative visual meshes. The current G1 scene has 47 robot collision geoms:
39 convex mesh hulls plus primitive shapes, including the two 17 × 6 × 1 cm
foot-sole boxes. World/floor geoms and visual-only geoms are excluded. Contact
margins are not expanded into additional surfaces.

Mesh faces use MuJoCo's compiled convex graph, following the layout used by
[MuJoCo 3.3.5's own convex-hull renderer](https://github.com/google-deepmind/mujoco/blob/3.3.5/src/render/render_context.c#L271).
Triangles are losslessly indexed before transmission; no simplification or
skeleton-based approximation is applied. Shapes are sent at connection setup
and identified by a content hash so reconnecting does not rebuild unchanged
render geometry. Each physical frame adds XYZ/XYZW transforms from actual
`geom_xpos`/`geom_xmat`. The viewer interpolates them at the same timestamps as
the physical skeleton.

The current scene's 88,604 hull triangles are drawn in two batched passes
(surfaces and outlines), with no per-frame geometry rebuilding. This exact
overlay is more expensive than the skeleton alone, particularly in a browser
using software WebGL; leave the checkbox off when it is not needed.

Local mesh vertices and primitive sizes remain in the compiled geom basis.
The streamed rotation includes the change to viewer Y-up world coordinates;
applying another axis conversion to local vertices would be incorrect.

## C API and checks

`motionbricks/physics.h` adds four caller-buffer functions on the opaque session:
`mb_physics_collision_count`, `mb_physics_collision_shape`,
`mb_physics_collision_triangles` and `mb_physics_collision_transforms`.
Definitions remain stable until the session is destroyed. Mesh sizes can be
queried before allocation; invalid indices/counts/capacities return errors.
The APIs expose no struct layout or borrowed pointer.

An independent MuJoCo instance checks every exported geom's origin and axes,
both at initialization and after replaying one native tick's captured motor
torques. Maximum observed world-frame error is 1.30e-7; both foot-sole dimensions
are checked explicitly. The same tests pass under ASan/UBSan. The new API
boundaries are included in the inference/physics fuzzer, excluding model loading.

Headless browser tests check the checkbox, transparency, foot dimensions and
basis conversion, indexed hulls, invalid transforms, and playback with the
overlay enabled. Screenshots are saved under `generated/sonic/ggml/stream-qa/`.
