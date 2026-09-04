# Go and Three.js demo

The initial demo is a local Go application that calls `libmotionbricks`
through the PureGo binding and serves an embedded Three.js viewer. It renders
the released 34-joint G1 hierarchy directly from MotionBricks root
translations and local XYZW joint rotations; it does not require MuJoCo or a
skinned mesh. Solid cyan cylinders and round joints identify the generated
character. An orange diamond-jointed ghost skeleton identifies a selected
placed style-pose constraint supplied to the planner; all four constraints can
be overlaid for inspection.

## Build and run

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

`-device` accepts `cpu`, `vulkan`, or `auto`. The server deliberately binds to
localhost by default. Model inference is serialized while sessions keep
independent agent/context state.

## Runtime shape

The browser creates a session, receives a 30 FPS animation chunk, and asks for
a replacement chunk when controls change or playback approaches the end.
Each request contains movement, facing, style, seed, and the number of frames
already consumed. The Go server advances that session's native agent and
returns owned animation data and target constraints as JSON:

- root translations: `[frames, 3]`;
- local joint rotations: `[frames, 34, 4]`, XYZW;
- placed target roots: `[4, 3]`;
- placed target local rotations: `[4, 34, 4]`, XYZW;
- G1 joint names, parent indices, and neutral positions from the loaded model.

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

Without the native asset environment variables, the parser test still runs
and the native/browser integration cases are skipped.

## Initial limitations

- G1 is the only skeleton supported by the released model.
- The viewer intentionally shows a bone skeleton, not a skinned avatar.
- HTTP JSON carries whole planned chunks; binary streaming and client-side
  overlap blending are future work.
- Sessions are in-memory and intended for a trusted local demo, not an
  internet-facing multi-user service.
