# Pose-token sampling and parity

The released upstream inference defaults to one pose-model pass, then
`gumbel_sample(pose_logits, temperature=1.0)` unless
`pose_token_sampling_use_argmax` is explicitly enabled. Native planning now
matches that default. Root-duration selection is unchanged; this sampler only
selects the eight VQ pose code heads per token position.

For each candidate code the sampler computes:

```
log_safe(x) = log(max(x, 1e-20))
gumbel(u)   = -log_safe(-log_safe(u))
score      = logit + gumbel(u)
token      = first argmax(score)
```

This follows NVIDIA's
`motionbricks/motionbricks/motion_backbone/models/sampling.py` at
`a0732b642c0333077e127a2f56ab0014c196bca4`. Both clamps are retained, including
the edge behavior at uniforms 0 and 1. Inputs are F32; no softmax is required
to sample. A separate 50,000-draw test checks the resulting categorical
frequencies against softmax probabilities with a 1 percentage-point tolerance.

## API and reproducibility

`mb_command_set_sampling_argmax(command, 0, error, capacity)` selects the default
Gumbel path; `1` selects argmax. Other integers and null handles are rejected.
`mb_command_get_sampling_argmax` exposes the setting without struct layout
assumptions. Go exposes `Command.SetSamplingArgmax(bool)` and the demo accepts
`-sampling gumbel|argmax` (default `gumbel`).

Each plan starts a private SplitMix64 stream at the command seed and maps its
high 24 bits to an exactly representable F32 uniform in `[0,1)`. No global RNG
is shared between agents or requests. Repeating a context, command, seed and
backend repeats its draws. Change the command seed for subsequent independent
draws. The existing seed-based target-style frame selection is unchanged.

PyTorch uses a different random generator. The same integer seed across
frameworks therefore does **not** imply the same tokens or motion. Even with
identical uniforms, tiny CPU/GPU logit differences can flip near-tied choices;
exact cross-backend token equality is a measured property of a fixture, not a
universal promise.

## Verified boundaries

- `tests/fixtures/upstream-gumbel.mbsampling` stores 512 rows of ten logits,
  uniforms, upstream noise/scores and expected token IDs. Its companion JSON
  records upstream revision, source hash, PyTorch version and fixture hash.
  `reference/capture_sampling.py` imports the actual pinned upstream module
  and replaces only its uniform draw. The ~84 KB fixture contains no weights
  and runs in ordinary CTest without Python or downloads. Every token must
  match exactly; maximum noise/score absolute error must be below `2e-5`.
- Native CPU and Vulkan agent traces now include the exact sampling uniforms.
  `reference/replay_transition_debug.py` replays upstream root, pose and
  decoder modules against these boundaries, including the actual upstream
  Gumbel function fed the recorded uniforms. For the tested 44-frame walking
  transition, **88/88 sampled tokens match** for both native backends; 27 of
  those choices differ from argmax. This checks real stochastic choices, not
  an accidentally unchanged argmax path. Codebook lookup from the upstream
  sampled tokens must also exactly match the native decoder's input.
- For that same transition, native CPU/Vulkan final roots differ by at most
  `2.57e-5 m` and local quaternion components by `1.11e-4`. The upstream pose
  logit relative-L2 differences are `1.58e-6` (CPU) and `1.64e-4` (Vulkan);
  the larger Vulkan drift did not change a selected token in this test.
- The existing 14-plan strict upstream CPU-reference suite still passes with
  **explicit argmax**. Its fixtures were captured in that mode; it is not
  relabelled as stochastic full-pipeline parity.

The neural replay is boundary-by-boundary: it feeds each upstream component
the recorded native conditioning and fixes the recorded duration. It does not
claim identical autonomous long-rollout behavior or identical framework RNGs.
The same Gumbel formula plus distribution checks establish sampler semantics;
shared-noise fixtures make operation-level parity deterministic.

## Regenerate and test

Use the existing trusted reference image (no unpickling is needed):

```sh
docker run --rm --user "$(id -u):$(id -g)" --entrypoint python \
  -v "$PWD:/work" -v /path/to/GR00T-WholeBodyControl:/upstream:ro \
  motionbricks-reference-session:torch2.7 \
  /work/reference/capture_sampling.py --upstream-root /upstream \
  --output /work/tests/fixtures/upstream-gumbel.mbsampling
cmake --build --preset debug
ctest --test-dir build/debug -R sampling --output-on-failure
```

The sampling/API fuzzer excludes GGUF loading:

```sh
cmake --preset asan-ubsan -DMOTIONBRICKS_ENABLE_FUZZERS=ON
cmake --build --preset asan-ubsan --target motionbricks-sampling-fuzz
./build/asan-ubsan/tests/motionbricks-sampling-fuzz -max_total_time=60
```

The initial ASan/UBSan smoke run completed 6,758,647 executions in 21 seconds
with no findings; this is not an exhaustive fuzzing claim.

Do not infer motion quality from passing sampling parity: it faithfully
implements the upstream distribution but cannot guarantee plausible recovery
from every external pose. Existing Kimodo seam/entry QA findings remain open.
