# SOMA Vulkan arithmetic shaders

These shaders adapt the pinned GGML normalization and softmax implementations to
reproduce the captured SM120 CUDA arithmetic used by the official SOMA mixed-FP16
ONNX export. Their ordinary operation paths retain the upstream calculations.

Build with `MOTIONBRICKS_ENABLE_SOMA_VULKAN_SHADERS=ON` (requires GGML and Vulkan).
`cmake/SomaVulkan.cmake` verifies the original shader and backend hashes and constructs a
private GGML source overlay under the build directory. The dependency checkout
is never patched. Source changes are CMake configure dependencies; the generated
backend commit label includes `soma-shaders`.

The overlay also retains scalar FP32 matrix pipelines on cooperative-matrix-2
hardware. Contiguous FP32 × FP32 operands use those pipelines instead of being
silently staged as FP16. Explicit half operands retain the cooperative path.
Non-contiguous staging and other operation families retain upstream behavior;
this is validated against SOMA fixtures, not a general precision guarantee for
arbitrary GGML graphs. `cmake/SomaVulkanBackend.cmake` applies these changes only
to the pinned backend hash and includes `soma_f32_pipelines.inc` in shader setup.

The private registry procedure `motionbricks_soma_vulkan_info_v1` reports eight
32-bit values: overlay protocol, PCI vendor ID, PCI device ID, Vulkan driver
version, enabled FP16, enabled cooperative-matrix-2, subgroup size, and Vulkan
driver ID. It rejects null arguments and an incorrect output count. Stock GGML
does not expose this procedure. The host qualification check requires protocol
1, RTX 5070 Ti (`10de:2c05`), NVIDIA proprietary driver 595.71.05
(`0x94d1c140`, driver ID 4), subgroup size 32, and both arithmetic capabilities.
This narrow hardware scope reflects the captured parity evidence. Qualification
is recorded in runner output metadata and gates the explicit public Vulkan mode.
A separate pre-initialization query reads physical-device properties without
creating a logical device, so CPU-first loading can preserve Vulkan capabilities.

The internal operation protocols are:

- Normalization: negative epsilon selects packed Welford normalization. The input
  is flattened into rows followed by one scale row and one bias row. Only data
  rows are consumed by the caller. Positive epsilon retains ordinary GGML norm.
- Softmax: `max_bias == -1` selects SM120 range reduction and corrected division.
  SOMA does not use ALiBi; ordinary `max_bias >= 0` operations retain upstream
  exponential/division arithmetic and ALiBi handling. The marker is passed per
  operation, not read from process environment by the shader.

The host selects these protocols only for the explicit SOMA arithmetic path.
The scoped softmax trial requires this build feature and rejects unsupported
builds. Public API calls select and restore the appropriate arithmetic settings
explicitly. The qualified public mixed mode needs no diagnostic flags. The scoped build has passed
all three captured chains and internal feature/full-motion batches, alongside
ordinary CPU/Vulkan FP32 regression checks. Hardware/driver qualification
must accompany any supported public mode; compiling the shaders does not imply
parity on arbitrary Vulkan devices.

The original GGML shaders and dependency are MIT licensed; see `ggml/LICENSE`.
Numerical fixtures, captured runtime identities and validation scope are documented
in `reference/README.md`.
