#!/usr/bin/env python3
"""Build a portable C++/Three.js replay from one traced upstream capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

from session_common import CAPTURE_FORMAT, UPSTREAM_REVISION, sha256
from session_replay import Replay, TARGET_FRAMES, write_replay


# Motion skeleton indices corresponding to MuJoCo's root plus 29 articulated
# bodies. Four virtual toe/hand end effectors do not exist as MuJoCo bodies.
MOTION_JOINT_INDICES = np.asarray(
    list(range(0, 7)) + list(range(8, 14)) + list(range(15, 25)) + list(range(26, 33)),
    dtype=np.int32,
)
MUJOCO_TO_MOTION = np.asarray(
    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float32
)
MOTION_TO_MUJOCO = MUJOCO_TO_MOTION.T
MODE_NAMES = (
    "idle", "slow_walk", "walk", "hand_crawling", "walk_boxing", "elbow_crawling",
    "stealth_walk", "injured_walk", "walk_stealth", "walk_happy_dance", "walk_zombie",
    "walk_gun", "walk_scared", "walk_left", "walk_right",
)


def load_safetensors(path: Path) -> dict[str, np.ndarray]:
    with safe_open(path, framework="np") as handle:
        return {key: handle.get_tensor(key) for key in handle.keys()}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def uncanonicalize_targets(
    joints: np.ndarray, roots: np.ndarray, origin: np.ndarray, heading: np.ndarray
) -> np.ndarray:
    canonical_motion = joints[0] + roots[0, :, None, :]
    canonical_mujoco = canonical_motion @ MOTION_TO_MUJOCO.T
    angle = float(heading[0])
    rotation = np.asarray([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    world_mujoco = canonical_mujoco @ rotation.T + origin[0]
    return world_mujoco @ MUJOCO_TO_MOTION.T


def build(capture: Path, upstream: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite replay directory: {output}")
    manifest = json.loads((capture / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != CAPTURE_FORMAT:
        raise ValueError("capture format is unsupported")
    if manifest["upstream"]["revision"] != UPSTREAM_REVISION:
        raise ValueError("capture uses a different upstream revision")
    if not manifest.get("trace", {}).get("target_boundaries"):
        raise ValueError("capture does not contain target-boundary traces")

    playback = load_safetensors(capture / manifest["capture"]["playback"]["file"])
    qpos = np.asarray(playback["emitted_qpos"], dtype=np.float32)
    if qpos.shape != (manifest["scenario"]["frame_count"], 36):
        raise ValueError(f"unexpected emitted qpos shape: {qpos.shape}")
    modes = np.asarray(playback["mode"], dtype=np.int32).reshape(-1)
    events = load_jsonl(capture / manifest["capture"]["events"]["file"])
    if len(events) != manifest["capture"]["plan_count"]:
        raise ValueError("event and plan counts disagree")

    import mujoco

    xml = upstream / "motionbricks/assets/skeletons/g1/scene_29dof.xml"
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    if model.nq != 36 or model.nbody != 31:
        raise ValueError(f"unexpected G1 MuJoCo dimensions: nq={model.nq}, nbody={model.nbody}")
    parents = np.asarray(model.body_parentid[1:], dtype=np.int32) - 1
    positions = np.empty((qpos.shape[0], model.nbody - 1, 3), dtype=np.float32)
    for frame, pose in enumerate(qpos):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        positions[frame] = np.asarray(data.xpos[1:], dtype=np.float32) @ MUJOCO_TO_MOTION.T
    if not np.allclose(positions[:, 0], qpos[:, [1, 2, 0]], atol=2e-6, rtol=0):
        raise ValueError("MuJoCo FK root does not match the captured qpos coordinate transform")

    target_positions = np.empty(
        (len(events), TARGET_FRAMES, len(MOTION_JOINT_INDICES), 3), dtype=np.float32
    )
    plan_frames, plan_modes, plan_lengths = [], [], []
    for index, event in enumerate(events):
        tensors = load_safetensors(capture / f"plan-{index:03d}.safetensors")
        joints = tensors["target_global_joint_positions"]
        roots = tensors["target_global_root_positions"]
        origin = tensors["canonical_first_frame_position"]
        heading = tensors["canonical_first_frame_heading_angle"]
        if (joints.shape != (1, TARGET_FRAMES, 34, 3) or
                roots.shape != (1, TARGET_FRAMES, 3) or origin.shape != (1, 3) or
                heading.shape != (1,)):
            raise ValueError(f"plan {index} has unexpected target trace shapes")
        world_motion = uncanonicalize_targets(joints, roots, origin, heading)
        target_positions[index] = world_motion[:, MOTION_JOINT_INDICES]
        plan_frames.append(event["first_generated_playback_frame"])
        plan_modes.append(event["mode"])
        plan_lengths.append(event["valid_length"])

    frame_plans = np.full(qpos.shape[0], -1, dtype=np.int32)
    for plan, first_frame in enumerate(plan_frames):
        last_frame = plan_frames[plan + 1] if plan + 1 < len(plan_frames) else qpos.shape[0]
        frame_plans[first_frame:last_frame] = plan

    replay = Replay(
        fps=int(manifest["scenario"]["fps"]),
        parents=parents,
        modes=modes,
        frame_plans=frame_plans,
        qpos=qpos,
        joint_positions=positions,
        plan_frames=np.asarray(plan_frames, dtype=np.uint32),
        plan_modes=np.asarray(plan_modes, dtype=np.int32),
        plan_valid_lengths=np.asarray(plan_lengths, dtype=np.uint32),
        target_positions=target_positions,
    )
    output.mkdir(parents=True)
    replay_path = output / "session.mbreplay"
    write_replay(replay_path, replay)
    replay_manifest = {
        "format": "motionbricks-portable-replay-v1",
        "source_capture": {
            "path": capture.name,
            "manifest_sha256": sha256(capture / "manifest.json"),
            "upstream_revision": UPSTREAM_REVISION,
        },
        "source_scene_xml_sha256": sha256(xml),
        "replay": {"file": replay_path.name, "sha256": sha256(replay_path)},
        "fps": replay.fps,
        "frame_count": replay.frame_count,
        "joint_count": replay.joint_count,
        "qpos_count": replay.qpos_count,
        "plan_count": replay.plan_count,
        "target_frames_per_plan": TARGET_FRAMES,
        "coordinate_system": "right_handed_y_up_z_forward",
        "joint_subset": MOTION_JOINT_INDICES.tolist(),
        "mode_names": list(MODE_NAMES),
    }
    (output / "manifest.json").write_text(
        json.dumps(replay_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.capture.resolve(), args.upstream_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
