# Live physics performance profile

This is the **pre-WebSocket HTTP baseline**. Production now uses
[server-owned streaming](STREAMING.md), with subsequent measurements there.

Measured 2026-09-07 using the current debug F32 NVIDIA Vulkan build, original
G1 SONIC, MuJoCo 3.3.5 and headless Chromium. The user's running demo was not
restarted or used as the test's physics worker. Measurements use an isolated
server with the same production handlers and frontend; other user processes
were left running. No runtime scheduling or inference arithmetic was changed.

## Finding

Rendering is not the bottleneck. The browser drains a five-tick (100 ms)
physics batch **before requesting another batch**, then freezes animation
while awaiting that request. Replanning also blocks new physics requests.
The GPU consequently spends much of its time waiting for the next request.

Eight-second walking measurements, after physics startup:

| Additional simulated request latency | Render FPS | Simulation seconds per wall second | P95 gap between physics updates | Largest gap |
|---|---:|---:|---:|---:|
| None: local HTTP | 60.0 | 0.670 | 66.8 ms | 216.7 ms |
| +50 ms | 60.2 | 0.462 | 133.4 ms | 333.4 ms |
| +100 ms | 60.1 | 0.357 | 200.0 ms | 416.7 ms |

The kinematic/render-only control was 60.2 FPS. Added latency is a delay in
the diagnostic fetch wrapper, not a measurement of the user's remote network
or a complete bandwidth/loss emulation. All measured requests returned 200;
none encountered worker-busy or ownership errors.

Locally, five-tick physics handlers averaged 10.3 ms (median 7.65 ms), while
the browser saw 20.4 ms per request. Planning handlers averaged 39.6 ms; the
browser saw 50.0 ms. The initial physical-scene setup took about one second,
but that was excluded from the steady-state browser interval. It does not
explain persistent slow animation.

In the local eight-second interval, simulation advanced only 5.36 seconds.
Non-overlapped physics request times totalled 1.08 seconds and plan request
times 0.45 seconds. These totals do not explain every lost millisecond:
animation-frame scheduling and the batch-clock reset also consume time.
Requests repeatedly upload about 61.6 KB of largely overlapping JSON motion
data (roughly 0.62 MB/s at the intended ten batches per second).

## Native breakdown

A diagnostic symbol interposer timed 498 controller ticks of the verified
walk. It excludes model/scene initialization and records actual graph calls,
buffer transfers and MuJoCo substeps. This is a burst-throughput measurement,
not a claim that every intermittently scheduled web request takes this time.
Two repeated runs took 0.423/0.424 wall seconds for 9.96 simulation seconds.

| Stage | Mean per 20 ms controller tick |
|---|---:|
| Complete native step | 0.824 ms |
| SONIC encoder | 0.126 ms |
| SONIC decoder | 0.166 ms |
| Four MuJoCo substeps | 0.329 ms |
| Remaining preparation, validation, diagnostics and FK | 0.203 ms |

Within the encoder/decoder time, the two graph calls total 0.229 ms, input
uploads 0.00084 ms, and output downloads/synchronization 0.0473 ms. These are
nested measurements and must not be added to the complete step again. Only
6,536 bytes are uploaded and 372 downloaded per tick; transfer bandwidth is
not the throughput limit. These wall timings include backend waits; they are
not GPU timestamp measurements separating kernel execution from driver work.

The HTTP worker uses `TryLock`, so it rejects contention rather than waiting
on its mutex. The measured browser received no such rejection. The standalone
model has one caller; encoder/decoder timings include their uncontended locks.
No claim is made about a heavily contended multi-client workload.

## Exact scheduling causes

In `demo/web/app.js`:

- `requestLiveBatch` refuses to run while `l.queue.length` is nonzero, or while
  a plan is pending. There is no overlap between playback and fetching.
- `advanceLivePhysics` starts the next request only after the queue empties.
  Awaiting compute/network therefore adds directly to playback time.
- Each response resets `l.elapsed` to zero, discarding residual fractional
  playback time instead of maintaining a continuous clock.
- MotionBricks replans every 16 reference frames; the live path stalls for
  that request too, even when the preceding plan still has usable frames.

## Recommended implementation, not applied by this profiling task

1. Keep a bounded reference/physics lookahead buffer and request ahead at a
   low-water mark. Maintain separate submitted, simulated and displayed
   cursors, allowing at most one physical request in flight.
2. Preserve the playback clock and interpolate timestamped results. A genuine
   underrun may hold the last pose, but normal batch completion must not
   discard time or intentionally empty the queue.
3. Plan before the reference horizon runs out. Tag plans with generations and
   commit at an explicit seam; never rewind an already integrated physical
   state to accept a late plan or Kimodo transition.
4. Upload each reference segment once and send IDs/timestamps thereafter.
   For higher-latency links, a bounded server-side control loop with streaming
   physical results avoids a round trip per 100 ms altogether.
5. Add performance assertions alongside correctness QA: simulation near 1x,
   measured underruns, no clock loss, and bounded memory/latency. Test walking,
   replanning, Kimodo entry/exit, reset, fall and ownership changes.

## Reproduction

Run `MOTIONBRICKS_PROFILE_ROOT="$PWD" go -C demo test -run
'^TestLivePhysicsProfile$' -count=1 -v` after building the native library and
preparing local model/scene assets. The diagnostic creates ignored JSON reports
under `generated/sonic/ggml/performance/browser-*.json`.

`reference/sonic_profile.cpp` is a Linux, single-caller diagnostic interposer.
Build it as a shared library with the project, GGML and MuJoCo headers, link
against `libmotionbricks` with `--no-as-needed` and a build-tree rpath, then
pass it as `--library` to `reference/run_native_sonic.py`. Its dependencies
supply the normal ABI; wrappers time the real implementations through
`RTLD_NEXT`. Set `MOTIONBRICKS_PROFILE_OUTPUT` to a new filename to save the
timing JSON. Outputs use exclusive creation, never overwrite an earlier run.
The interposer is not part of the installed runtime or demo build.
