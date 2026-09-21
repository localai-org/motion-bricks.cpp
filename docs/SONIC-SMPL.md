# Native SONIC human-pose encoder

The original-release SONIC **SMPL encoder (mode 2)** runs in GGML alongside the
existing G1 encoder (mode 0), sharing the same decoder. There is no ZMQ, Python
or ONNX Runtime dependency at inference time. CPU and Vulkan are supported.

## Prepare the model

Use the existing hash-pinned official encoder/decoder ONNX files described in
[SONIC setup](SONIC-GGML.md). In the conversion environment:

```sh
python reference/convert_sonic_gguf.py \
  --policy generated/sonic/policy \
  --output generated/sonic/ggml/sonic-g1-smpl.gguf --include-smpl
```

The combined archive contains 34 F32 tensors and 19,056,285 values. It preserves
all G1/decoder tensors and adds five SMPL weight/bias pairs. Its architecture tag
is `g1-smpl-mode02-mlp-fsq32-v1`. The default conversion still produces the original
24-tensor G1 archive, byte-identical to the previous conversion. Existing G1 files
are accepted; requesting mode 2 on them fails explicitly.

Source SHA-256 pins:

- Encoder: `013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3`
- Decoder: `c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed`

This does not convert the low-latency or v1.1 checkpoints. The converter rejects
other ONNX identities before extracting weights. Weights remain generated local
artifacts; this change does not publish a new model release.

## Call the encoder

Load with `mb_sonic_load`, select mode 2 in observation element 0, and call
`mb_sonic_encode` or `mb_sonic_encode_batch`. Input remains 1,762 floats per
request; output remains 64 FSQ token floats. See the exact
[SMPL observation layout](API-SONIC-PHYSICS.md#human-pose-encoder-observation-mode-2-1762-floats).
Batch sizes 1–64 are supported, with one mode per batch. Calls can switch modes;
mode-specific graphs/caches stay separate. Layer traces follow the last successful
encoder call. The same original-release decoder consumes either encoder's tokens.

The caller supplies ten SMPL reference samples, their base-relative orientations
and six wrist angles per sample. The network is `840 → 2048 → 1024 → 512 → 512 → 64`,
with SiLU between layers and the original ties-to-even FSQ32 quantizer afterward.

GEM-X's SOMA-to-SMPL adapter, timestamped reference sampling, physical-base heading
alignment and LocalAI HTTP/WebSocket session integration remain separate work.
The current native MuJoCo clip adapter still constructs mode-0 observations.
Neural encoder support alone does not establish closed-loop human-motion tracking.

## Validation

An independent fixture runs the full hash-verified upstream ONNX with graph
optimizations disabled, selecting mode 2 and exposing its five preactivations.
It covers 64 deterministic synthetic cases: zero input, smooth plausible human
reference windows, and random observations. It checks all encoder/decoder layers,
exact FSQ tokens, decoder actions, and composed encoder-to-decoder actions.
These are inference tests, not camera/robot trials or a demographic benchmark.

```sh
python reference/capture_sonic_neural.py --policy generated/sonic/policy \
  --mode 2 --output generated/sonic/ggml/smpl-layers.mbsonic
cmake --preset sonic-cpu
cmake --build --preset sonic-cpu -j8
ctest --test-dir build/sonic-cpu -R 'motionbricks-sonic-' --output-on-failure
```

Both scripts refuse to overwrite existing outputs. CTest enables model-backed
checks only when the corresponding files exist. Vulkan parity tests additionally
require `MOTIONBRICKS_ENABLE_VULKAN_PARITY_TESTS=ON` and a Vulkan-enabled build.

The release checks use a clean export of the pinned GGML commit, independent of
local CPU experiment changes, with at most eight CPU cores. Detailed measurements
and scope are recorded in [the validation record](benchmarks/sonic-smpl-2026-09-21.json).

SMPL CPU (one and two threads) and NVIDIA Vulkan checks passed with **zero FSQ
token mismatches**. Maximum absolute layer/output error was `1.14441e-5`; maximum
relative L2 error was `8.29357e-7` on CPU and `8.01558e-7` on Vulkan. Existing
limits remain `5e-4` absolute and `1e-5` relative, with exact token equality.
The existing 2,205-case G1 suite also passed on both backends with the combined
model. CPU/Vulkan batch checks cover sizes 1, 4 and 64 and mode switching.

ASan/UBSan API checks and a seeded 60-second input-fuzz campaign passed:
500,043 executions, no findings. The campaign instruments project code, not
external GGML/driver binaries, and does not fuzz model-file parsing. Five
separate deterministic loading checks reject invalid architecture/source IDs,
missing SMPL tensors and truncated archives. These bounded checks are not a
claim of exhaustive input coverage.
