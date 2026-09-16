#!/usr/bin/env python3
"""Convert verified MotionBricks safetensors into inference-only GGUF v3.

This converter deliberately rejects PyTorch checkpoint files.  The pinned
reference container must first extract the upstream pickle archives into the
safe, deterministic safetensors lane documented in reference/README.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from safetensors import safe_open


GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
ALIGNMENT = 32

GGML_TYPE_F32 = 0
# ggml v0.20.2 / GGUF v3 enum value (the submodule is pinned with the bundle).
GGML_TYPE_I32 = 26
GGML_TYPE_BF16 = 30
GGML_FTYPE_ALL_F32 = 0
GGML_FTYPE_MOSTLY_BF16 = 24
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_STRING = 8
GGUF_TYPE_UINT64 = 10

UPSTREAM_REVISION = "a0732b642c0333077e127a2f56ab0014c196bca4"
SAFE_FORMAT = "motionbricks-safe-manifest-v1"
BUNDLE_FORMAT = "motionbricks-gguf-bundle-v1"
SAFE_HASHES = {
    "pose": "01327768be7413111dc927947a95cfb3e9ee7c52acfa35a054e8c2f3b838b888",
    "root": "d1529a8c9da915cb7dd0499272baba61db480660c0aae92b67fbe69828e83c5a",
    "vqvae": "544782e605ed96d60bf999243ef8f44640ea021a5655ef20af2d2853c3e18b5b",
    "support": "229b764411652b2ab0f824481d6daf897f701a97223029759444fc5bc241ea22",
}

JOINT_NAMES = (
    "pelvis_skel",
    "left_hip_pitch_skel", "left_hip_roll_skel", "left_hip_yaw_skel",
    "left_knee_skel", "left_ankle_pitch_skel", "left_ankle_roll_skel", "left_toe_base",
    "right_hip_pitch_skel", "right_hip_roll_skel", "right_hip_yaw_skel",
    "right_knee_skel", "right_ankle_pitch_skel", "right_ankle_roll_skel", "right_toe_base",
    "waist_yaw_skel", "waist_roll_skel", "waist_pitch_skel",
    "left_shoulder_pitch_skel", "left_shoulder_roll_skel", "left_shoulder_yaw_skel",
    "left_elbow_skel", "left_wrist_roll_skel", "left_wrist_pitch_skel",
    "left_wrist_yaw_skel", "left_hand_roll_skel",
    "right_shoulder_pitch_skel", "right_shoulder_roll_skel", "right_shoulder_yaw_skel",
    "right_elbow_skel", "right_wrist_roll_skel", "right_wrist_pitch_skel",
    "right_wrist_yaw_skel", "right_hand_roll_skel",
)


def encoded(value: str) -> bytes:
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def kv_string(key: str, value: str) -> bytes:
    return encoded(key) + struct.pack("<I", GGUF_TYPE_STRING) + encoded(value)


def kv_u32(key: str, value: int) -> bytes:
    return encoded(key) + struct.pack("<II", GGUF_TYPE_UINT32, value)


def kv_f32(key: str, value: float) -> bytes:
    return encoded(key) + struct.pack("<If", GGUF_TYPE_FLOAT32, value)


def kv_u64(key: str, value: int) -> bytes:
    return encoded(key) + struct.pack("<IQ", GGUF_TYPE_UINT64, value)


def aligned(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def compact_name(name: str) -> str:
    """Keep upstream names readable while respecting GGML_MAX_NAME (64)."""
    if len(name.encode("utf-8")) < 64:
        return name
    replacements = (
        ("_root_token_transformer_encoder.layers.", "root.l."),
        ("_shared_transformer_encoder.layers.", "shared.l."),
        ("_transformer_encoder.layers.", "transformer.l."),
        ("self_attn.", "attn."),
        ("in_proj_weight", "qkv.weight"),
        ("in_proj_bias", "qkv.bias"),
        ("external_cond_blocks.", "xcond."),
        ("target_cond_blocks.", "tcond."),
    )
    output = name
    for old, new in replacements:
        output = output.replace(old, new)
    if len(output.encode("utf-8")) >= 64:
        suffix = hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]
        output = output.encode("utf-8")[:48].decode("utf-8", "ignore") + "." + suffix
    return output


@dataclass
class Tensor:
    name: str
    value: np.ndarray
    kind: int
    dimensions: tuple[int, ...]
    offset: int

    @property
    def size(self) -> int:
        element_size = 2 if self.kind == GGML_TYPE_BF16 else 4
        return int(self.value.size) * element_size

    def bytes(self) -> bytes:
        if self.kind == GGML_TYPE_BF16:
            # Match ggml_compute_fp32_to_bf16: round to nearest even and
            # force NaNs to quiet NaNs.
            floats = np.asarray(self.value, dtype="<f4", order="C")
            bits = floats.view(np.uint32)
            rounded = (bits + (0x7FFF + ((bits >> 16) & 1))).astype(np.uint32)
            bf16 = (rounded >> 16).astype("<u2")
            nan = (bits & 0x7FFFFFFF) > 0x7F800000
            bf16[nan] = ((bits[nan] >> 16) | 64).astype("<u2")
            return bf16.tobytes(order="C")
        dtype = "<f4" if self.kind == GGML_TYPE_F32 else "<i4"
        return np.asarray(self.value, dtype=dtype, order="C").tobytes(order="C")


def load_selected(path: Path, select: Callable[[str], bool], strip: str) -> list[tuple[str, np.ndarray]]:
    output: list[tuple[str, np.ndarray]] = []
    with safe_open(path, framework="numpy") as source:
        for original in source.keys():
            if not select(original):
                continue
            name = original.removeprefix(strip)
            value = source.get_tensor(original)
            output.append((name, value))
    if not output:
        raise ValueError(f"selection produced no tensors from {path}")
    return output


def tensor_kind(component: str, name: str, value: np.ndarray, pose_bf16: bool = False) -> int:
    if np.issubdtype(value.dtype, np.floating):
        if (pose_bf16 and component == "pose" and value.ndim == 2
                and name not in {"_pose_token_emb.weight", "_proj_num_valid_positions.weight"}):
            return GGML_TYPE_BF16
        return GGML_TYPE_F32
    if np.issubdtype(value.dtype, np.integer) or np.issubdtype(value.dtype, np.bool_):
        return GGML_TYPE_I32
    raise ValueError(f"unsupported tensor dtype for {component}:{name}: {value.dtype}")


def write_component(path: Path, component: str, values: list[tuple[str, np.ndarray]],
                    source_hash: str, *, general_name: str = "NVIDIA MotionBricks G1",
                    extra_metadata: list[bytes] | None = None,
                    pose_bf16: bool = False) -> dict[str, object]:
    tensors: list[Tensor] = []
    names: set[str] = set()
    offset = 0
    parameter_count = 0
    for original_name, value in sorted(values):
        name = compact_name(original_name)
        if name in names:
            raise ValueError(f"compacted tensor name collision: {name}")
        names.add(name)
        if value.ndim > 4:
            raise ValueError(f"GGML supports at most four dimensions: {name} has {value.shape}")
        kind = tensor_kind(component, name, value, pose_bf16)
        shape = tuple(int(item) for item in value.shape) or (1,)
        tensor = Tensor(name, value, kind, tuple(reversed(shape)), offset)
        tensors.append(tensor)
        parameter_count += int(value.size)
        offset = aligned(offset + tensor.size)

    metadata = [
        kv_string("general.architecture", "motionbricks"),
        kv_string("general.name", general_name),
        kv_u32("general.alignment", ALIGNMENT),
        kv_u32("general.file_type",
               GGML_FTYPE_MOSTLY_BF16 if pose_bf16 and component == "pose"
               else GGML_FTYPE_ALL_F32),
        kv_u32("motionbricks.format_version", 1),
        kv_string("motionbricks.component", component),
        kv_string("motionbricks.skeleton", "g1skel34"),
        kv_string("motionbricks.upstream_revision", UPSTREAM_REVISION),
        kv_string("motionbricks.source_sha256", source_hash),
        kv_u64("motionbricks.parameter_count", parameter_count),
    ]
    if extra_metadata:
        metadata.extend(extra_metadata)
    if component == "support":
        metadata.append(kv_string("motionbricks.joint_names", ",".join(JOINT_NAMES)))

    header = bytearray(GGUF_MAGIC)
    header += struct.pack("<IQQ", GGUF_VERSION, len(tensors), len(metadata))
    for item in metadata:
        header += item
    for tensor in tensors:
        header += encoded(tensor.name)
        header += struct.pack("<I", len(tensor.dimensions))
        header += struct.pack("<" + "Q" * len(tensor.dimensions), *tensor.dimensions)
        header += struct.pack("<IQ", tensor.kind, tensor.offset)
    header += bytes(aligned(len(header)) - len(header))

    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as stream:
        stream.write(header)
        position = 0
        for tensor in tensors:
            stream.write(bytes(tensor.offset - position))
            payload = tensor.bytes()
            if len(payload) != tensor.size:
                raise AssertionError(f"incorrect payload size for {tensor.name}")
            stream.write(payload)
            position = tensor.offset + len(payload)
        # GGUF's declared data blob includes alignment after the final tensor.
        stream.write(bytes(offset - position))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    result = {
        "path": path.name,
        "sha256": sha256(path),
        "tensor_count": len(tensors),
        "parameter_count": parameter_count,
    }
    print(f"wrote {path}: {len(tensors)} tensors, {parameter_count:,} values")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--safe-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pose-bf16", action="store_true",
        help="store pose matrix weights as BF16; retain embeddings, norms, biases, and other components as F32",
    )
    args = parser.parse_args()

    safe_manifest = json.loads((args.safe_directory / "manifest.json").read_text(encoding="utf-8"))
    if safe_manifest.get("format") != SAFE_FORMAT:
        raise ValueError("not a compatible trusted-extraction manifest")
    if safe_manifest.get("upstream_revision") != UPSTREAM_REVISION:
        raise ValueError("trusted extraction came from a different upstream revision")
    for component, expected in SAFE_HASHES.items():
        path = args.safe_directory / f"{component}.safetensors"
        actual = sha256(path)
        recorded = safe_manifest["components"][component]["sha256"]
        if actual != expected or recorded != expected:
            raise ValueError(f"safe input hash mismatch for {component}: {actual}")

    selections = {
        "pose": load_selected(
            args.safe_directory / "pose.safetensors",
            lambda name: name.startswith("backbone_net.") and name != "backbone_net.initted",
            "backbone_net.",
        ),
        "root": load_selected(
            args.safe_directory / "root.safetensors",
            lambda name: name.startswith("backbone_net."),
            "backbone_net.",
        ),
        "vq-decoder": load_selected(
            args.safe_directory / "vqvae.safetensors",
            lambda name: name.startswith("pose_net.decoder.")
            or name == "pose_net.quantizer.vq._codebook.embed",
            "pose_net.",
        ),
        "support": load_selected(
            args.safe_directory / "support.safetensors", lambda _name: True, ""
        ),
    }
    source_for = {"pose": "pose", "root": "root", "vq-decoder": "vqvae", "support": "support"}
    expected = {
        "pose": (209, 136_588_272),
        "root": (150, 34_122_833),
        "vq-decoder": (51, 12_437_277),
        "support": (4, 972),
    }
    components: dict[str, object] = {}
    for component, values in selections.items():
        actual = (len(values), sum(int(value.size) for _, value in values))
        if actual != expected[component]:
            raise ValueError(f"unexpected {component} inventory: {actual}, expected {expected[component]}")
        source = source_for[component]
        components[component] = write_component(
            args.output / f"{component}.gguf", component, values, SAFE_HASHES[source],
            pose_bf16=args.pose_bf16,
        )

    bundle = {
        "format": BUNDLE_FORMAT,
        "upstream_revision": UPSTREAM_REVISION,
        "skeleton": "g1skel34",
        "joint_count": len(JOINT_NAMES),
        "inference_parameter_count": sum(
            int(components[name]["parameter_count"]) for name in ("pose", "root", "vq-decoder")
        ),
        "components": components,
        "safe_manifest_sha256": sha256(args.safe_directory / "manifest.json"),
    }
    temporary = args.output / "manifest.json.tmp"
    temporary.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output / "manifest.json")
    print(f"inference parameters: {bundle['inference_parameter_count']:,}")


if __name__ == "__main__":
    main()
