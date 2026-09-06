# Observational motion QA

Run this gate after changes to skeleton decoding, root transforms, pose
conversion, context management, target placement, blending, controller
cadence, or authored-sequence stitching:

```sh
nix develop -c ./scripts/run_motion_qa.sh ../kimodo.cpp/demo-output
```

The command is expected to exit non-zero when it detects an implausible
motion. Do not update limits simply to make a changed result pass.

## Procedure

The test starts the real native model and demo server, opens the UI in headless
Chromium, selects a real external Kimodo G1 GLB, and activates it through the
same Play button as an operator. Playback is paused only through an opt-in QA
hook so exact frames can be inspected.

It takes numerical snapshots and PNGs at:

1. entry start, middle, and final blended frame;
2. authored first, quarter, middle, three-quarter, and penultimate frame;
3. regenerated MotionBricks frame zero, final context-blended frame, and a
   later settled frame.

Each numerical snapshot contains the root, 34 local XYZW rotations, 34
rendered world-joint positions, control-lock state, and timeline metadata. A
full-sequence pass catches spikes between the sampled screenshots.

The report combines limits relative to the unmodified opening/closing
components with conservative absolute safety limits:

| Check | Default limit |
| --- | ---: |
| Quaternion norm error | 0.001 |
| Root speed | 15 m/s |
| Root acceleration | 120 m/s² |
| Rendered joint speed | 25 m/s |
| Local joint step | 120°/frame |
| Entry local-joint step | 3× component opening, minimum 12°/frame |
| Entry world-joint step | 3× component opening, minimum 0.20 m/frame |
| Entry root gap | 0.05 m |
| Exit root gap | 4× Kimodo tail step, minimum 0.25 m |
| Exit pose gap | 3× stable component step, minimum 60° |

Root excursion must also stay within 0.5 m of the Kimodo source excursion.
The absolute thresholds are intentionally generous: they are designed to
catch map-scale teleports, explosive acceleration, and single-frame limb
folding without pretending to be a dynamics or contact-validity proof.

Artifacts default to `generated/motion-qa/`. Select another loaded clip with
`MOTIONBRICKS_MOTION_QA_CLIP=<relative-id>`, another backend with
`MOTIONBRICKS_MOTION_QA_DEVICE=cpu|vulkan`, or provide an output directory as
the script's second argument.

## Review rule

A passing numerical report still requires looking at the entry, boundary, and
exit screenshots for self-intersection, ground penetration, foot sliding,
camera loss, and motion that is semantically inconsistent with the source.
Conversely, a failed report is not accepted merely because one still image
looks plausible. Review the first failing frame and compare its values with
both source-component statistics in the JSON.

## Current finding: root explosion fixed; seam QA still fails

After matching upstream's virtual endpoint conditioning convention, the CPU
exit root gaps are 0.0094 m (compound) and 0.0112 m (hand-walking), down from
5.319 m and 2.936 m. The compound Vulkan test agrees. All root and world-joint
exit speed/acceleration checks now pass on these runs.

The overall clip gate still fails: entry blending remains imperfect, and the
all-joint exit rotation check counts large changes in the virtual endpoints'
orientation convention. Hand-walking also fails its opening single-frame
rotation check. Thresholds remain unchanged. See the
[post-fix validation](KIMODO_TRANSITION_DEBUG.md#implemented-correction-and-validation)
for exact results and the distinction between physical and virtual joints.

### Original failure

The 2026-09-06 CPU run fails on both the 90-frame hand-walking clip
`6cef070244b9d11e` and the 240-frame compound clip `4b38c30f37a44024`.

For the hand-walking clip, the first MotionBricks root is 2.936 m from the
closing Kimodo root. The following opening reaches 276.75 m/s root speed,
6900.80 m/s² root acceleration, 281.54 m/s rendered-joint speed, and a
173.42° single-frame local-joint change. The entry also fails for this extreme
pose: its eight-frame morph reaches 29.52° and 0.438 m per frame.

The longer compound clip confirms that the exit defect is not specific to a
handstand: its first regenerated root is 5.319 m away, with 133.55 m/s root
speed and a 164.11° seam pose gap.

The subsequent stage trace and upstream replay found a concrete handoff
representation mismatch: animated virtual hand/toe rotations are supplied in
channels where upstream inserts global identity rotations. Normalisation is
the first numerical amplification; upstream PyTorch reproduces the ensuing
decoder failure. See [transition diagnosis](KIMODO_TRANSITION_DEBUG.md) for
measurements and controlled endpoint-only ablations. A generic claim about
arbitrary poses being out of distribution was premature.

The four-frame blend cannot repair the bad generated root because it retains
a generated contribution. Correct the input convention first, then re-run
this gate unchanged before deciding whether an additional exit bridge is
necessary. The existing entry-blend failures are a separate issue.
