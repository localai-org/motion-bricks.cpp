# Versioned format registry

A reader must reject an unknown major version rather than guessing.

| Format | Version | Status |
|---|---:|---|
| Installed C ABI | 1 | [Model inference, optional controllers, SONIC and physics](API.md) implemented |
| GGUF model-bundle manifest | 1 | F32 inference components implemented |
| MotionBricks style asset | 1 | Upstream G1 exemplar schema implemented |
| Reference fixture manifest | 1 | Root, pose, and decoder fixtures implemented |
| Portable observational replay | 1 | C++, Go, and Three.js readers implemented |
| Open-loop plan parity fixture | 1 | Python writer and checked C++ reader implemented |
| Open-loop parity report | 1 | C++ writer and Go/Three.js viewer implemented |
| Native backend comparison | 1 | CPU/Vulkan behavioral gate and neural diagnostics implemented |
| WebSocket animation protocol | 1 | [Server-owned streaming](STREAMING.md) implemented |

Minor, backwards-compatible additions are represented by optional keys or by
new API functions. Existing function signatures and field meanings do not
change within one C ABI version.

## GGUF model bundle v1

A bundle is a directory containing `manifest.json` and four GGUF v3 files:

| Component | Runtime contents | Tensors | Learned parameters |
|---|---|---:|---:|
| `pose.gguf` | Pose-token planner | 209 | 136,588,272 |
| `root.gguf` | Root-trajectory and duration planner | 150 | 34,122,833 |
| `vq-decoder.gguf` | Pose codebook and convolutional decoder | 51 | 12,437,277 |
| `support.gguf` | G1 skeleton, parents, mean, and standard deviation | 4 | 0 |

The learned inference total is 183,148,382 F32 parameters. The support file
contains 972 non-learned scalar values. Training-only VQ encoder tensors,
codebook EMA state, and initialization flags are deliberately omitted.

Every component carries `general.architecture=motionbricks`, format version,
component role, `g1skel34` skeleton identity, pinned upstream revision, source
safetensors hash, and exact scalar count. PyTorch tensor dimensions are stored
in reversed GGML order. Names at or above GGML's 64-byte limit are compacted
deterministically and collision-checked by the converter.

## MotionBricks style asset v1

A `.mbstyle` is a GGUF v3 file with `component=style`, the `g1skel34`
skeleton identity, a source SHA-256, name, configured speed, and frame count.
It contains checked F32 tensors for global joint positions
`[frames,34,3]`, flattened global rotation matrices `[frames,34,9]`, root
positions `[frames,3]`, and headings `[frames]`, plus an I32 allowed-duration
mask `[11]` corresponding to 6--16 tokens. The runtime validates all metadata,
shapes, finite values, frame limits, and the binary mask before accepting it.

The source identity is not restricted to NVIDIA's clip archive, so the same
schema can hold a future checked Kimodo/G1 conversion. Skeleton and coordinate
compatibility must still be established by the converter.

## Portable observational replay v1

`.mbreplay` is a bounded, immutable little-endian binary used only to replay a
captured upstream session. It contains no executable or pickle-bearing data and
does not invoke either planner. The 40-byte header is `MBRPLY1\0` followed by
eight U32 values: version (`1`), FPS, frame count, joint count, qpos count, plan
count, target frames per plan (`4`), and flags (`0`). The contiguous payload is:

| Field | Type | Shape |
|---|---|---|
| parents | I32 | `[joints]` |
| modes | I32 | `[frames]` |
| active plan | I32 | `[frames]` (`-1` means pre-roll) |
| upstream MuJoCo qpos | F32 | `[frames,qpos]` |
| animated joint world positions | F32 | `[frames,joints,3]` |
| first playback frame | U32 | `[plans]` |
| plan mode | I32 | `[plans]` |
| generated plan length | U32 | `[plans]` |
| target joint world positions | F32 | `[plans,4,joints,3]` |

Coordinates are right-handed, Y-up, and Z-forward. The current G1 replay uses
the root plus 29 physical MuJoCo bodies. MotionBricks' four virtual hand/toe
endpoints (motion-skeleton indices 7, 14, 25, and 33) are intentionally omitted
instead of being approximated. Readers reject unsupported headers, unsafe
dimensions, bad parent/plan indices, non-finite floats, truncation, and trailing
bytes.

## Open-loop plan parity fixture v1

`.mbparity` is a bounded little-endian binary containing independent planning
events. Its 36-byte header is `MBPARI1\0` followed by seven U32 values: version
(`1`), FPS, plan count, joints (`34`), context frames (`4`), target frames
(`4`), and flags (`0`). The header is followed by I32 parents `[34]`, F32
neutral joints `[34,3]`, and one variable-length record per plan.

Each record starts with command frame, expected frame count, mode, reserved
zero, U64 seed, movement `[3]`, and facing `[3]`. It then stores the four-frame
public context (roots and local XYZW rotations), expected upstream roots/local
rotations/FK positions for the variable plan length, and expected placed-target
roots/local rotations/FK positions for four frames. All animation data is
right-handed, Y-up, and Z-forward. Expected output is reconstructed from
upstream's unblended 414-value motion representation, not its optional
four-frame MuJoCo playback blend. The sidecar manifest records whether those
features came directly from the accepted CUDA observation or from the verified
upstream CPU replay; the strict CPU gate uses the latter.

The JSON report format `motionbricks-open-loop-report-v1` contains immutable
per-plan upstream/native world positions for visual comparison, native local
rotations and target transforms for backend comparison, scalar error metrics,
exact duration results, aggregate worst cases, skeleton topology, device,
movement/facing commands, and explicit acceptance tolerances. The added native
rotation/target arrays are optional v1 keys so older viewers remain compatible.
JSON is used here because it is a diagnostic artifact and lets the browser
consume the exact machine report.
Pose-token IDs remain absent from this public-output report. Optional separate
neural trace JSON records expose root, pose-token, VQ, and composition
boundaries without changing the C API or the `.mbparity` layout.

`motionbricks-native-backend-open-loop-comparison-v1` compares two reports at
the public animation boundary. It records source hashes, hardware and driver,
per-plan and worst-case root/FK/local-rotation/target metrics, duration results,
and explicit CPU-to-accelerator tolerances. When paired native trace directories
are supplied it also records duration-token and pose-token disagreement plus
continuous root/pose/decoder maxima. Neural values are diagnostic; acceptance
is based on the observable animation and exact duration.
