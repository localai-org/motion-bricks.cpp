# Kimodo → MotionBricks transition diagnosis

2026-09-06. CPU native inference and pinned upstream PyTorch 2.7 CPU replay.
The endpoint-conditioning fix is now implemented; see the validation below.
The original diagnosis is retained here for comparison. This is **not a
complete motion-quality fix**: the real browser QA still flags seam rotations
and entry blending, although the explosive root jumps are gone.

## Implemented correction and validation

`encode_context` now replaces only the four virtual endpoint global rotation
matrices with identity, after computing FK in the canonical input frame.
It leaves their parent-derived positions, physical joint rotations, and the
caller's authored animation buffers unchanged. No root clipping or longer
blending was introduced.

The new `motionbricks-context-endpoints` regression uses rotated parents and
root, independently checks analytic endpoint positions and global identity
features, checks physical orientations, verifies input buffers are unchanged,
and proves that changing endpoint-local rotations cannot change conditioning.
It also covers taking the last four frames from a longer context.

Post-fix CPU browser measurements, using the same clips and QA thresholds:

| Check | Compound before → after | Hand-walking before → after |
| --- | ---: | ---: |
| Exit root gap | 5.319 → 0.0094 m | 2.936 → 0.0112 m |
| Peak exit root speed | 133.54 → 0.8304 m/s | 276.75 → 0.1793 m/s |
| First decoded root height | −17.594 → 0.1670 m | 10.574 → 0.7538 m |
| First blended root height | −5.184 → 0.1445 m | 3.705 → 0.7592 m |

Compound Vulkan QA agrees: 0.0095 m exit gap and 0.8328 m/s peak exit root
speed. Its target is visible in the post-fix exit screenshot.

Completed checks:

- 11 selected debug CPU regression, neural parity, and ABI tests passed.
- 11 ASan/UBSan native/reference-tool tests passed, with no sanitizer findings.
- All 14 strict upstream CPU-reference transition plans passed under
  ASan/UBSan: maximum root error 0.000232 m, FK error 0.000261 m, and rotation
  error 0.0136 degrees. This is the existing upstream-context fixture, not a
  newly established arbitrary-Kimodo full-pipeline parity claim.
- Three Vulkan component parity tests, CPU/Vulkan agent parity, and the
  standard Go/headless-browser demo suite passed.

The observational clip QA remains **failed**, with thresholds unchanged.
Exit seam maxima (124.86° compound, 175.97° hand-walking) occur at virtual
endpoints switching from authored rotations to the model's placeholder
convention. Physical-joint exit seam maxima are 4.87° and 8.81°, respectively.
Leaf orientation changes do not move skeleton joints, but may matter to an
attached mesh. Hand-walking additionally fails the opening single-frame
rotation gate (135.71°); both clips retain entry rotation-blend failures and
hand-walking retains its entry world-joint-step failure. These are not waived
or claimed fixed by the conditioning correction.

Artifacts are under `generated/transition-debug/fixed-compound/`,
`fixed-handstand/`, `fixed-compound-vulkan/`, and
`fixed-cpu-strict-parity.json`. The source fix is built; no persistent demo
server was restarted as part of these test runs.

## Original diagnosis: first bad boundary

The Kimodo GLB importer verifies the shared G1 34-joint topology but does not
adapt its virtual endpoint rotations to MotionBricks' conditioning convention.
`finishKimodo` passes the final four authored poses to `SetContext`.
`encode_context` computes every joint's global rotation from that hierarchy,
including virtual endpoints 7, 14, 25 and 33. `run_transition` then normalises
those rotations with the released MotionBricks statistics.

Upstream's `helper/mujoco_helper.py:406–411`, at revision
`a0732b642c0333077e127a2f56ab0014c196bca4`, instead inserts **global identity**
rotations for these endpoints after converting the canonical MuJoCo context.
This is also already documented in our `reference/build_plan_parity.py`.
The default scheme is `dummy`, not `parent`.

For compound clip `4b38c30f37a44024`, the largest normalised source feature is
frame 3, internal pose feature 254: left virtual hand joint 25, cont6d component
4 (matrix YY). Its dual-statistics index is 262:

- raw value approximately −0.06256; training mean 1;
- training standard deviation 5.1735e−8;
- effective denominator `sqrt(std² + 1e−5)` = approximately 0.0031623;
- normalised result **−336.0105**.

The normalisation formula matches upstream `motionlib/core/utils/stats.py`.
It is not a missing epsilon or a numerical divide-by-zero. The mismatch is
in the meaning of the input channel before normalisation. Finite quaternions
and matching joint names are insufficient to validate that convention.

## Propagation and target check

Compound-clip exit, measured in the actual native agent path:

| Boundary | Observation |
| --- | --- |
| Raw source pose features | −0.997 to +1.071 |
| Normalised source poses | −336.01 to +305.10 |
| Root planner output, normalised | −58.65 to +67.29 |
| Derived local-root conditioning, normalised | −120.75 to +123.38 |
| VQ decoder output, normalised | −3616.20 to +4864.00 |
| First decoded root height | −17.5937 m |
| After world transform | −17.5937 m |
| After four-frame opening blend, first height | −5.1840 m |

The final authored root is approximately `[-0.3765, 0.1349, 6.0470]`.
The first target root is `[-0.3785, 0.7794, 6.1527]`: about 0.106 m away
horizontally, with an ordinary standing height. The target is not placed at
the teleported location. World restoration preserves the already bad height;
the blend reduces it rather than introducing it.

The UI deliberately hides targets during authored playback and restores the
returned targets after finish. With the generated character underground and
the camera following it, losing sight of a sensible target is unsurprising;
this does not establish an independent target-rendering bug.

## Upstream replay and controlled ablation

`reference/replay_transition_debug.py` verifies the pinned upstream revision,
tracked model-source cleanliness, and safetensors hashes. It independently
replays the native root and decoder inputs through upstream modules. This is
not a claim of full-pipeline parity: the intervening pose-token selection is
held at the captured native output, and duration is fixed to the captured 11
tokens. Root and decoder agreement isolate those stages.

| Measurement | Compound clip | Hand-walking clip `6cef070244b9d11e` |
| --- | ---: | ---: |
| Root replay max-absolute error | 0.000107 | 0.000143 |
| Root replay relative L2 | 1.44e−6 | 7.59e−6 |
| Decoder replay max-absolute error | 0.00704 | 0.00309 |
| Decoder replay relative L2 | 1.80e−6 | 2.33e−6 |
| Upstream first decoded height | −17.5937 m | +10.5738 m |
| Endpoint-only corrected first height | +0.4661 m | +0.7343 m |
| Corrected decoder height range | 0.4001–0.8468 m | 0.3632–0.7343 m |

The diagnostic ablation changes only source endpoint rotation features to
normalised global identity. All other decoder inputs, including the already
bad root conditioning and chosen tokens, remain fixed. Separately, the same
correction reduces root planner output range to −4.6434…1.4206 (compound) and
−0.1968…1.4314 (hand-walking). This strongly isolates the endpoint mismatch
as a cause of the explosion; it does **not** prove the corrected animation is
acceptable or that all physical-joint conditioning conventions match.

After endpoint correction, the hand-walking source still has normalised
features as large as 28.38. Physical joint axes, limits, and upstream MuJoCo
round-tripping need review before claiming general support for arbitrary
Kimodo poses. An ordinary pose and an inverted pose may need different
recovery behavior even once the representation is correct.

## Wall time

Native CPU stage times in milliseconds; single observed runs, not a benchmark:

| Stage | Compound exit | Hand-walking exit |
| --- | ---: | ---: |
| Root planning (including duration selection rerun) | 214.95 | 163.36 |
| Pose planning | 157.99 | 167.87 |
| VQ decoding | 38.39 | 35.80 |
| Motion representation decoding | 0.10 | 0.10 |
| Sum | 411.42 | 367.12 |

These clocks exclude model loading, HTTP/browser latency, trace serialization,
and some inter-stage preparation. No waiting/locking explanation is needed
for the numerical discontinuity: bad tensors are present inside inference.

## Reproduce

Build the native shared library and install the normal demo QA prerequisites
(Go and Chromium). Nix is optional. Use a new directory for each process;
the diagnostic filenames start at `transition-0.json` per process.

```sh
cmake --build build/debug --target motionbricks_shared -j4
mkdir -p generated/transition-debug/run
MOTIONBRICKS_TRANSITION_TRACE_DIR="$PWD/generated/transition-debug/run" \
MOTIONBRICKS_MOTION_QA_CLIP=4b38c30f37a44024 \
  ./scripts/run_motion_qa.sh /path/to/kimodo/demo-output
```

The QA command currently exits nonzero as expected. Trace 0 is ordinary
opening planning; trace 1 is the failing Kimodo exit. Traces include complete
context, normalisation statistics, intermediate boundaries, targets, and root
positions before and after world restoration/blending. They contain motion
data and remain in the ignored `generated/` directory. Tracing is opt-in via
the environment variable and does not alter inference tensors or the C ABI.

Using the existing trusted reference image and local upstream checkout:

```sh
docker run --rm --user "$(id -u):$(id -g)" --entrypoint python \
  -v "$PWD:/work" -v /path/to/GR00T-WholeBodyControl:/upstream:ro \
  motionbricks-reference-session:torch2.7 \
  /work/reference/replay_transition_debug.py \
  --upstream-root /upstream --safe-directory /work/generated/safe \
  --trace /work/generated/transition-debug/run/transition-1.json \
  --output /work/generated/transition-debug/run/upstream-replay.json
```

## Follow-up implementation boundaries (original diagnosis)

1. Match upstream's virtual endpoint convention in **canonical model context**.
   Do not replace the authored playback rotations, and do not simply set local
   quaternions to identity: identity local rotation inherits the parent and is
   not identity global rotation.
2. Add a regression with non-identity parent rotations, nonzero world heading,
   and the actual Kimodo boundary. Assert encoded endpoint channels match
   upstream, then run full transition and browser QA without weakening bounds.
3. Inspect physical joint representation and recovery quality after this fix.
   Do not conceal invalid generated motion by clamping root height.
4. Resolve the secondary seam-timing issue: the exit starts from the first of
   the last four context frames after the UI has already played the clip to
   completion. This can replay three frames, but cannot explain metre-scale
   vertical jumps. Review opening blend quality separately.
