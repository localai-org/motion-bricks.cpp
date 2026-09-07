# Server-owned motion streaming

The native demo uses WebSocket commands and timestamped poses. The browser
renders an interpolated stream; it does not schedule inference, advance the
agent or step physics. Uploads, metadata and recorded playback remain HTTP.

An optional [collision geometry overlay](COLLISION-GEOMETRY.md) sends compiled
shape definitions at connection setup and actual geom transforms with poses.

## Scheduling and playback

One server event loop owns the reference timeline and physical state. A separate
bounded worker plans from a four-frame snapshot at a future reference boundary.
Stale revisions and late plans are discarded. Planning runs ahead of playback;
normal replanning cannot replace the past.

If a cold/slow plan exceeds its lookahead, both simulation clocks wait before
its reserved boundary. The UI reports **Waiting for motion planner** and the
same plan can then be accepted, instead of repeatedly discarding it as late
until the reference runs out. Ordinary warmed-up playback remains real time.

The server advances an integer 50 Hz clock. With physics enabled, every tick
runs SONIC and four 200 Hz MuJoCo substeps. Network writes never block this loop.
The robot model is compiled once at startup; toggling physics clears state without
recompiling it. At most four catch-up ticks run per loop; severe computation lag
slows simulation and increments `slow_ticks`, never increases the physics
timestep or skips physical steps.

Reference replacement uses an eight-frame eased crossfade, starting exactly
at the old reference pose. This is an explicit **deployment smoothing policy**,
not an upstream inference/parity claim. Kimodo clips play in full, with the exit
planned from their final pose and previous movement intent retained.

No directional input requests the idle pose style (when available) and explicit
zero target speed. The completed stop plan's last pose is then held until input
changes; the 50 Hz clock and optional physical simulation continue. This demo
policy avoids repeatedly feeding the native planner's small forward fallback
back into idle. The native planner and upstream parity semantics are unchanged.

The browser buffers about 150 ms, interpolates roots/physical joints and SLERPs
local quaternions. Its render clock preserves fractional elapsed time, with at
most 2% correction for clock drift. Missing frames cause an explicit hold and
rebuffer, not extrapolation. Playback starts at the first displayed pose. A new
session epoch flushes old frames; older epochs cannot resurrect old motion.
The queue is limited to 128 poses; long hidden-tab stalls resynchronize.

## Protocol version 1

`GET /api/stream` describes availability. Connect to `/api/stream/socket` using
WebSocket (WSS behind HTTPS). Version 1 uses JSON and complete poses, avoiding
delta-chain recovery problems. Reverse proxies must forward WebSocket upgrades.

The first connection controls one shared simulation; up to seven more viewers
are read-only. `hello` returns ownership, a private reconnect token, last command
sequence, epoch, physics availability and playback delay. The browser stores the
token in session storage and reconnects with `?resume=TOKEN`. Disconnect pauses
the world without resetting it. After 30 seconds disconnected, another connection
may claim ownership. This is demo ownership, not authentication: use an
authenticated reverse proxy on untrusted networks.

Client messages use increasing positive safe-integer `seq` values:

| Type | Additional fields |
|---|---|
| `control` | `style`, `move: [x,z]`, `facing: [x,z]` |
| `jump` | none |
| `play_clip` | `clip` asset ID |
| `physics` | `enabled` boolean |
| `pause`, `resume`, `reset`, `ping` | none |

Acknowledgements report `received`, `scheduled`, `applied`, `superseded`,
`duplicate` or `rejected`. Continuous intent is coalesced into future plans.
Discrete clips/jumps are not replayed on reconnect. `targets` events publish
planned target keyframes. `frame` messages contain epoch, tick, simulation time,
reference root and 34 XYZW rotations; physics adds 30 world XYZ joints, parents
and contacts. Frames also report pause/error, action progress, control revision
and slow ticks. Reset alone starts a new epoch.

Production rejects old mutating HTTP batch endpoints with 410. They remain in
isolated tests to preserve regression/profile baselines.

## Bounds and failure handling

Same-origin browser upgrades only; bounded 4 KiB commands, 120 commands/sec,
eight viewers, 64 queued commands, one planning job, 32 reliable events per
viewer, and one coalescing outgoing pose. Slow viewers disconnect after bounded
write time; they cannot stall simulation. WebSocket Ping/Pong and browser JSON
keepalives are supported. Malformed/unknown fields and invalid control vectors
are rejected before planning.

Reference shapes, unit quaternions, targets and finite positions are checked
before use. Coordinates beyond 10 km, reference per-axis jumps over 0.7 m per
30 Hz sample, or physical joint jumps over 1 m per 50 Hz tick stop the session.
These are conservative corruption guards, **not biomechanical plausibility
proofs**. Errors identify the producing stage; bad poses are not clamped into
apparently successful output. A detected fall is a warning only: the policy,
physics substeps and animation continue without resetting. Numerical safety
failures still stop the simulation; disabling physics or resetting allows recovery.
The UI labels numerical failures as **Simulation stopped**, displays the reason and recovery
instructions, and disables Resume while recovery is required. Server logs record
the epoch, tick, action and error, distinguishing a safety pause from a crash.

## Verification

Model-free tests exercise framing, origin rejection, Ping/Pong, input limits,
malformed commands, timeline continuity and deterministic delivery jitter in
headless Chromium. Playback tests check monotonicity, bounded queues, packet-loss
holds, epochs, pause and NaN/Inf/invalid quaternion rejection.

Run ordinary tests with `cd demo && CGO_ENABLED=0 go test ./...`. Full native
Vulkan/physics browser QA uses prepared local assets without downloading weights:

```sh
MOTIONBRICKS_STREAM_TEST_ROOT="$PWD" \
MOTIONBRICKS_STREAM_KIMODO="$KIMODO_OUTPUT" \
  sh -c 'cd demo && CGO_ENABLED=0 go test -run "^TestStream" -v -count=1'
```

The test measures actual interpolated client poses: real-time progression,
root/joint displacement, finite values, buffering, pause/reconnect/reset, full
Kimodo playback and return to walking. Physical Kimodo playback separately
checks that any fall is reported without automatic reset or numerical explosion.
Successful tracking of arbitrary clips is not guaranteed by neural parity.

Debug Vulkan measurements (2026-09-07): 1.005x simulation speed locally and
1.004x with 100 ms added message-delivery delay, about 56 rendered FPS in
headless Chromium. Both eight-second walking windows had zero playback underruns
or invalid values; largest rendered root step was 4.4 cm, reference joint step
12.5 cm and physical joint step 11.0 cm. The full Kimodo handoff's largest root
step was 5.2 cm and walking
resumed. Physical Kimodo reported a fall without resetting or invalid values;
maximum rendered physical joint step was 16.9 cm. Artificial delivery delay is
not full WAN bandwidth/loss emulation. Reports and screenshots are written to
`generated/sonic/ggml/stream-qa/`.

See [the previous HTTP profile](SONIC-PERFORMANCE.md) for the bottleneck and
[SONIC parity and sanitizer coverage](SONIC-GGML.md) for inference checks.

The final native regression run passed 21 tests, including CPU/Vulkan parity.
Four ASan/UBSan SONIC tests passed, and a further 73,873 inference/physics ABI
fuzz cases (including the reusable reset) completed without sanitizer findings.
Socket lifecycle/ownership tests also passed Go's race detector.

Idle regression QA additionally checks startup and walk-to-stop: after settling,
the reference has exactly zero displacement over two seconds while simulation
time advances. Tests cover movement resuming from the hold, a deliberately slow
planner and visible physics-fall status. The post-fall regression requires the
actual native physics clock to advance at least two more seconds after falling,
without pause, reset or invalid output.
The warning-only fall run advanced native physics another 4.64 seconds after
the detected fall, with no pause, reset or invalid values (largest rendered
physical joint step 14.6 cm).
