#!/usr/bin/env python3
"""Build a compact exact test of upstream's four-frame playback blend."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
from safetensors import safe_open

from build_plan_parity import JOINTS, TRUSTED_FILES, qpos_to_motion, sha256
from session_common import CAPTURE_FORMAT, UPSTREAM_REVISION


def tensors(path: Path) -> dict[str, np.ndarray]:
    with safe_open(path, framework="np") as handle:
        return {key: handle.get_tensor(key) for key in handle.keys()}


def build(capture: Path, upstream: Path, support_path: Path, output: Path) -> None:
    manifest = json.loads((capture / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != CAPTURE_FORMAT or manifest["upstream"]["revision"] != UPSTREAM_REVISION:
        raise ValueError("capture is not from the pinned upstream revision")
    scene = upstream / "motionbricks/assets/skeletons/g1/scene_29dof.xml"
    if sha256(scene) != TRUSTED_FILES["motionbricks/assets/skeletons/g1/scene_29dof.xml"]:
        raise ValueError("upstream G1 scene does not match the pinned revision")

    import mujoco

    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    support = tensors(support_path)
    parents = support["joint_parents"].astype(np.int32)
    neutral = support["neutral_joints"].astype(np.float32)
    playback = tensors(capture / manifest["capture"]["playback"]["file"])
    events = [json.loads(line) for line in (capture / "events.jsonl").read_text().splitlines() if line]

    records: list[np.ndarray] = []
    upstream_max = 0.0
    weights = np.linspace(0.3, 0.7, 4, dtype=np.float32)
    for index, event in enumerate(events):
        plan = tensors(capture / f"plan-{index:03d}.safetensors")
        raw_qpos = plan["raw_qpos"][0, :4].copy()
        filtered_qpos = plan["qpos"][0, :4]
        context_qpos = playback["context_qpos"][int(event["command_after_output_frame"])]
        formula = raw_qpos.copy()
        formula[:, :3] = context_qpos[:, :3] * (1 - weights[:, None]) + raw_qpos[:, :3] * weights[:, None]
        formula[:, 7:] = context_qpos[:, 7:] * (1 - weights[:, None]) + raw_qpos[:, 7:] * weights[:, None]
        upstream_max = max(upstream_max, float(np.max(np.abs(formula - filtered_qpos))))

        converted = []
        for qpos in (context_qpos, raw_qpos, filtered_qpos):
            roots, local = qpos_to_motion(qpos, model, data, parents, neutral)
            converted.extend((roots.reshape(-1), local.reshape(-1)))
        records.append(np.concatenate(converted).astype("<f4"))

    with output.open("wb") as stream:
        stream.write(b"MBBLEND1")
        stream.write(struct.pack("<II", len(records), JOINTS))
        for record in records:
            stream.write(record.tobytes())
    print(f"upstream formula max_abs={upstream_max:.9g} records={len(records)} output={output}")
    if upstream_max > 2e-6:
        raise RuntimeError("recorded upstream qpos does not match its documented blend")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--support", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.capture.resolve(), args.upstream_root.resolve(), args.support.resolve(),
          args.output.resolve())


if __name__ == "__main__":
    main()
