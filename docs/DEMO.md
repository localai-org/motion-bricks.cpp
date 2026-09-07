# Go and Three.js demo

The initial demo is a local Go application that calls `libmotionbricks`
through the PureGo binding and serves an embedded Three.js viewer. It renders
the released 34-joint G1 hierarchy directly from MotionBricks root
translations and local XYZW joint rotations; it does not require MuJoCo or a
skinned mesh. Solid cyan cylinders and round joints identify the generated
character. An orange diamond-jointed ghost skeleton identifies a selected
placed style-pose constraint supplied to the planner; all four constraints can
be overlaid for inspection.

The native viewer sends commands over WebSocket and interpolates timestamped
poses from the server-owned simulation. See [streaming and client QA](STREAMING.md)
for playback buffering, reconnect/ownership and numerical safety checks.

## Build and run

For optional actuator-driven simulation, enable **Live physics · GGML SONIC**
after following [the SONIC setup](SONIC-GGML.md). It overlays the physical robot
and reference, works during Kimodo playback, and keeps physical state across
plans and clip transitions. Physics remains optional for the kinematic demo.

First build the native project and create the model/style assets described in
the main README. Then build the Go application:

```sh
cmake --build --preset debug
cd demo
CGO_ENABLED=0 go build -o ../build/debug/bin/motionbricks-demo .
cd ..
```

The demo and reusable Go binding call the native shared library through
PureGo. They contain no cgo bridge and do not require a C compiler during the
Go build, so cgo can be disabled. This does not remove the separate C++/CMake
build that produces `libmotionbricks`.

Run it from the repository root:

```sh
./build/debug/bin/motionbricks-demo \
  -listen 127.0.0.1:8080 \
  -library ./build/debug/libmotionbricks.so \
  -model ./generated/g1-f32 \
  -styles ./generated/styles \
  -device cpu
```

To expose Kimodo-authored G1 sequences alongside live planning, add a directory
containing Kimodo's node-only `animation.glb` exports. It is scanned
recursively; exports for other skeletons are ignored.

```sh
./build/debug/bin/motionbricks-demo \
  -listen 0.0.0.0:8080 \
  -library ./build/debug/libmotionbricks.so \
  -model ./generated/g1-f32 \
  -styles ./generated/styles \
  -device vulkan \
  -kimodo-dir ../kimodo.cpp/demo-output
```

`MOTIONBRICKS_KIMODO_DIR` is the equivalent environment variable. The GLBs
remain external runtime inputs and are not copied into this repository.

You can also use **Upload animation GLB** in the control pane without configuring
`-kimodo-dir`. It accepts Kimodo G1 (`g1skel34`) skeleton-animation GLBs at 30 FPS,
up to 16 MiB and 10 minutes. Other skeletons/mesh-only GLBs are rejected; this
does not perform retargeting. A validated upload is selected automatically;
press Play to start it. Uploads survive page refresh but are held in bounded
server memory and cleared on restart. Original local files are not modified.

The initial stitching policy is deliberately bounded and deterministic:

- starting a clip advances the MotionBricks context to the visible frame,
  aligns the clip's first root position and heading to that pose, and prepends
  an eight-frame smooth quaternion/root blend;
- every authored source frame then plays exactly once and cannot be
  interrupted; the server rejects normal plan requests while the browser
  disables movement, style, jump, and clip controls;
- a visible progress bar measures the authored sequence, excluding the short
  entry blend;
- on completion, the final four aligned Kimodo frames replace the agent's
  context. The next plan restores the pre-clip style, world-space movement and
  facing, using MotionBricks' normal first-four-frame context blend. Walking
  therefore continues instead of becoming stationary. Space/Escape or tapping
  the active pad direction stops it; new directional input replaces it.

This first pass does not feed arbitrary Kimodo poses into the model as target
keyframes. The controller currently exposes locomotion/style targets, and
earlier out-of-distribution full-pose target experiments were unstable. Using
the opening pose for an explicit transition and the closing frames for the
supported context handoff keeps the seam inspectable while preserving the
complete authored animation.

Open `http://127.0.0.1:8080/`. Hold physical W/A/S/D keys to move relative to
the current camera yaw, or tap an on-screen direction to latch the same
camera-relative control; tap the active pad direction again to stop.
Space or Escape also stops movement. Left/right arrow keys rotate the facing
direction without changing the current travel vector. The selector switches
among all `.mbstyle` files found in the style directory, including the 15
converted upstream styles. Drag over the viewport to orbit, use the wheel to
zoom, and use **Reset camera** to restore the automatically framed view. The
camera follows only the animated skeleton, so target placement never pulls the
view away from the character. The T0–T3 slider selects one fully visible target
pose. **Overlay all four consecutive poses** reveals the complete constraint
window. These are adjacent 30 FPS constraint frames rather than four distant
waypoints, so their exact world positions are intentionally close together.

Camera following uses a critically damped pelvis anchor rather than the whole
skeleton's bounding box. Position and look-at share the filtered anchor, so
arm swings cannot jerk the aim independently of the camera. Vertical movement
is damped more heavily and capped at 0.8 m/s; horizontal following is capped
at 6 m/s per axis. Tracking state persists across replans and authored/live
handoffs. Reset camera explicitly recentres it; this does not alter any
animation or conceal motion errors in the QA data.

Pose-token sampling now defaults to upstream-style **Gumbel sampling** at
temperature 1. `-sampling argmax` restores the deterministic diagnostic mode.
Equal command inputs and seeds reproduce native draws; PyTorch's same integer
seed is not expected to produce the same random stream. See
[sampling parity](SAMPLING.md) for shared-noise validation and its limits.

Press **Jump** or `J` to request one ordinary MotionBricks transition with a
temporary 5.0 m/s target speed. This uses the current movement direction, or
the facing direction when stationary. The higher speed makes the existing
spring controller place the selected style's four standard keyframes farther
away with a higher implied root velocity. It does not impose a vertical arc,
switch styles, supply custom poses, or post-process the generated root.
The complete one-shot transition plays before ordinary locomotion resumes;
the normal 16-frame walking replan must not cut off its airborne phase.
In the fixed CPU walking regression, this produces about 16 cm of both-foot
clearance (versus 1 cm at 2 m/s), peaking at frame 22. This is a short generated
hop, not a guaranteed high jump for every style or input pose. The regression
checks actual FK foot clearance, and the browser test checks that the jump
survives frame 16 and returns to normal planning afterwards.

`-device` accepts `cpu`, `vulkan`, or `auto`. The server deliberately binds to
localhost by default. Model inference is serialized while sessions keep
independent agent/context state.

## Runtime shape

The browser creates a session, receives a 30 FPS animation chunk, and asks for
a replacement chunk when controls change or after every 16 played frames while
movement remains requested (about 0.53 seconds). This matches the default
upstream interactive-controller cadence and keeps a moving character on a
receding horizon instead of letting it reach and slow at the end of each target
trajectory.
Each request contains movement, facing, style, seed, and the number of frames
already consumed. The Go server advances that session's native agent and
returns owned animation data and target constraints as JSON:

- root translations: `[frames, 3]`;
- local joint rotations: `[frames, 34, 4]`, XYZW;
- placed target roots: `[4, 3]`;
- placed target local rotations: `[4, 34, 4]`, XYZW;
- G1 joint names, parent indices, and neutral positions from the loaded model.

After every inference the native controller applies upstream's default seam
filter to the first four frames: generated contributions of 0.3, 0.4333333,
0.5666667, and 0.7 for root translation and the 29 physical joint coordinates.
Generated root orientation is retained, matching upstream MuJoCo qpos behavior.

The native target data is captured after style-frame sampling, spring-based
world placement, and heading correction. The ghosts therefore visualize the
actual planner inputs. The viewer does not force them in front: forward travel
normally places them ahead, while stops and turns can make them overlap the
character or move sideways.

The browser builds `THREE.Bone` objects from the returned hierarchy and draws
solid cylinders/spheres for the generated skeleton and translucent
cylinders/diamonds for target poses. It also draws generated and target root
paths on the floor. Animation chunks are immutable in JavaScript; a later
version can replace JSON with a binary streaming protocol without changing
the native API.

Three.js r180 is vendored under `demo/web/vendor` so the demo has no runtime
CDN dependency.

## Captured-session replay

The same application has a replay-only mode for direct comparison with the
upstream MuJoCo MP4. It does not load `libmotionbricks`, GGUF weights, styles,
or an agent:

```sh
./build/debug/bin/motionbricks-demo \
  -listen 127.0.0.1:8080 \
  -replay ./generated/session-replay/session.mbreplay
```

The browser parses the versioned binary directly, draws the animated physical
G1 hierarchy in mint, and draws T0--T3 in distinct amber, orange, red, and
magenta. Play/pause and frame scrubbing use the artifact's 30 FPS timeline.
The plan/mode display and both root paths come from that same immutable file;
the camera bounds are computed from the animated skeleton only.

## Open-loop parity viewer

The demo can instead open a JSON report produced by `motionbricks-parity`:

```sh
./build/debug/bin/motionbricks-demo \
  -listen 127.0.0.1:8080 \
  -comparison ./generated/open-loop-parity/cpu-report.json
```

This mode loads no model or native library. By default it plays a 176-frame
showcase assembled from the accepted capture's forward-walk, right-turn, and
zombie-walk replans. Each showcased slice stops where the next replan begins,
so it does not replay overlapping predictions twice. Select any independently
evaluated replan from the same menu to loop and scrub it in isolation. The
solid mint rig is the upstream unblended plan;
the blue diamond-jointed rig is native output. Red line segments connect each
corresponding joint, while the panel reports per-frame maximum/RMS joint error,
aggregate plan metrics, style, and exact/mismatched duration. Solid and dashed
floor paths show upstream and native roots. The comparison is deliberately
open loop: at each replan boundary the native branch starts again from the
recorded upstream context, so an early error cannot contaminate later plans.
The showcase is therefore a visual comparison playlist, not evidence of
closed-loop native playback.

## Tests

With the generated assets present, CTest registers `motionbricks-go-demo` when
Go and Chromium are available. The test starts an in-process HTTP server,
loads the real native model, plans an initial `walk` chunk, then uses headless
Chromium to select `walk_zombie`, turn right, plan another chunk, render the
34-joint generated hierarchy plus the target inspector, and capture initial,
forward-motion, and style-and-turn screenshots. It uses real Chrome click and
keyboard events and asserts forward-pad movement, pad stop, keyboard movement,
keyboard stop, camera-relative direction before and after a real viewport
orbit, animated-skeleton camera anchoring, individual target selection,
and the four-pose overlay before the visual self-test.

When `generated/session-replay/session.mbreplay` exists at CMake configuration
time, the suite also starts a replay-only server and asks headless Chromium to
jump to a turning replan. It asserts 345 frames, 30 physical joints, 14
plans, four visible target ghosts, byte-identical HTTP delivery, and an
animated-only camera anchor, then captures a replay screenshot.

Setting `MOTIONBRICKS_COMPARISON` adds HTTP validation and a headless Chromium
test for the parity viewer. It selects a turning plan, scrubs it, verifies two
34-joint rigs and 34 error vectors, requires a green report-level parity
verdict, and captures a screenshot:

```sh
cd demo
MOTIONBRICKS_COMPARISON=../generated/open-loop-parity/cpu-report.json \
MOTIONBRICKS_COMPARISON_SCREENSHOT=../generated/open-loop-parity/threejs-comparison.png \
MOTIONBRICKS_CHROME="$(command -v chromium)" CGO_ENABLED=0 go test -v ./...
```

The Go tests can also be run directly:

```sh
cd demo
MOTIONBRICKS_LIB=../build/debug/libmotionbricks.so \
MOTIONBRICKS_MODEL=../generated/g1-f32 \
MOTIONBRICKS_STYLES=../generated/styles \
MOTIONBRICKS_CHROME="$(command -v chromium)" \
CGO_ENABLED=0 go test -v ./...
```

### Observational motion QA

Changes that affect skeleton decoding, root placement, context, blending, clip
alignment, or controller cadence must also run the observational motion QA:

```sh
nix develop -c ./scripts/run_motion_qa.sh ../kimodo.cpp/demo-output
```

The default run selects the shortest compatible G1 sequence. Set
`MOTIONBRICKS_MOTION_QA_CLIP` to a relative clip ID such as
`6cef070244b9d11e`, and `MOTIONBRICKS_MOTION_QA_DEVICE` to `cpu` or `vulkan`,
to exercise a particular sequence/backend. Artifacts go to
`generated/motion-qa` unless a second script argument supplies another output
directory.

The headless browser pauses and samples exact frames at the entry start,
middle, and boundary; throughout the authored sequence; and at the first,
last blended, and settled exit frames. Every snapshot records the root, all 34
local XYZW rotations, all 34 rendered world-joint positions, playback state,
and a PNG. The JSON report also scans every frame for:

- non-finite values and quaternion norm drift;
- root displacement, speed, acceleration, and total excursion;
- per-joint angular steps and rendered world-space joint speed;
- entry bounds derived from the original MotionBricks and Kimodo components;
- absolute anti-teleport/limb-explosion limits and strict entry/exit seam
  limits.

This is a regression gate, not a baseline generator: a failed limit should be
investigated visually and numerically rather than relaxed to match a newly
broken output. Keep the report and screenshots together when reviewing a
motion-affecting change.

Without the native asset environment variables, the parser test still runs
and the native/browser integration cases are skipped.

## Initial limitations

- G1 is the only skeleton supported by the released model.
- The viewer intentionally shows a bone skeleton, not a skinned avatar.
- HTTP JSON carries whole planned chunks; binary streaming is future work.
- Sessions are in-memory and intended for a trusted local demo, not an
  internet-facing multi-user service.
