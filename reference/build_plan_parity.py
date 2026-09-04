#!/usr/bin/env python3
"""Build compact open-loop native parity input from an accepted traced session."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

from plan_parity import Fixture, Plan, write_fixture
from session_common import CAPTURE_FORMAT, TRUSTED_FILES, UPSTREAM_REVISION, sha256


JOINTS = 34
SUPPORT_SHA256 = "229b764411652b2ab0f824481d6daf897f701a97223029759444fc5bc241ea22"
GLOBAL_ROTATION_OFFSET = 5 + 33 * 3
MOTION_JOINT_INDICES = np.asarray(
    list(range(0, 7)) + list(range(8, 14)) + list(range(15, 25)) + list(range(26, 33)),
    dtype=np.int32,
)
MUJOCO_TO_MOTION = np.asarray(
    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float32
)


def load_tensors(path: Path) -> dict[str, np.ndarray]:
    with safe_open(path, framework="np") as handle:
        return {key: handle.get_tensor(key) for key in handle.keys()}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def cont6d_matrix(value: np.ndarray) -> np.ndarray:
    x = value[..., :3]
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    z = np.cross(x, value[..., 3:6])
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    y = np.cross(z, x)
    return np.stack((x, y, z), axis=-1).astype(np.float32)


def y_rotation(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float32,
    )


def matrices_to_xyzw(matrices: np.ndarray) -> np.ndarray:
    """Stable batch matrix-to-quaternion conversion, canonicalized to w >= 0."""
    from scipy.spatial.transform import Rotation

    result = Rotation.from_matrix(matrices.reshape(-1, 3, 3)).as_quat().astype(np.float32)
    result[result[:, 3] < 0] *= -1
    return result.reshape(matrices.shape[:-2] + (4,))


def local_rotations(global_rotations: np.ndarray, parents: np.ndarray) -> np.ndarray:
    local = global_rotations.copy()
    for joint in range(1, JOINTS):
        local[:, joint] = np.swapaxes(global_rotations[:, parents[joint]], -1, -2) @ global_rotations[:, joint]
    return matrices_to_xyzw(local)


def forward_kinematics(
    roots: np.ndarray, local_xyzw: np.ndarray, parents: np.ndarray, neutral: np.ndarray
) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    frames = roots.shape[0]
    local = Rotation.from_quat(local_xyzw.reshape(-1, 4)).as_matrix().reshape(frames, JOINTS, 3, 3)
    global_rotations = np.empty_like(local, dtype=np.float32)
    positions = np.empty((frames, JOINTS, 3), dtype=np.float32)
    for joint in range(JOINTS):
        parent = int(parents[joint])
        if parent < 0:
            global_rotations[:, joint] = local[:, joint]
            positions[:, joint] = roots
        else:
            global_rotations[:, joint] = global_rotations[:, parent] @ local[:, joint]
            offset = neutral[joint] - neutral[parent]
            positions[:, joint] = positions[:, parent] + np.einsum(
                "fij,j->fi", global_rotations[:, parent], offset
            )
    return positions


def features_to_world(
    features: np.ndarray, origin_mujoco: np.ndarray, original_heading: float,
    parents: np.ndarray, neutral: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if features.ndim != 2 or features.shape[1] != 414:
        raise ValueError(f"unexpected global motion shape {features.shape}")
    roots = features[:, :3].astype(np.float32, copy=True)
    six = features[:, GLOBAL_ROTATION_OFFSET:GLOBAL_ROTATION_OFFSET + JOINTS * 6]
    global_rotations = cont6d_matrix(six.reshape(-1, JOINTS, 6))
    predicted_heading = np.arctan2(global_rotations[0, 0, 0, 2], global_rotations[0, 0, 2, 2])
    correction = y_rotation(float(original_heading) - float(predicted_heading))
    roots = roots @ correction.T
    world_origin = origin_mujoco @ MUJOCO_TO_MOTION.T
    roots[:, [0, 2]] += world_origin[[0, 2]] - roots[0, [0, 2]]
    global_rotations = correction[None, None] @ global_rotations
    local = local_rotations(global_rotations, parents)
    return roots, local, forward_kinematics(roots, local, parents, neutral)


def qpos_to_motion(
    qpos: np.ndarray, model, data, parents: np.ndarray, neutral: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    import mujoco

    frames = qpos.shape[0]
    positions = np.empty((frames, JOINTS, 3), dtype=np.float32)
    global_rotations = np.zeros((frames, JOINTS, 3, 3), dtype=np.float32)
    dead = np.ones(JOINTS, dtype=bool)
    dead[MOTION_JOINT_INDICES] = False
    for frame, pose in enumerate(qpos):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        positions[frame, MOTION_JOINT_INDICES] = np.asarray(data.xpos[1:], dtype=np.float32) @ MUJOCO_TO_MOTION.T
        matrices = np.asarray(data.xmat[1:], dtype=np.float32).reshape(-1, 3, 3)
        global_rotations[frame, MOTION_JOINT_INDICES] = (
            MUJOCO_TO_MOTION[None] @ matrices @ MUJOCO_TO_MOTION.T[None]
        )
    # Upstream canonicalizes the MuJoCo qpos before conversion, then inserts
    # each missing toe/hand joint as a global identity rotation.  The native
    # API canonicalizes the supplied local hierarchy later, so encode the
    # first root heading here: native's inverse-heading transform will turn
    # these four virtual joints back into the same global identity.
    first_root_heading = np.arctan2(
        global_rotations[0, 0, 0, 2], global_rotations[0, 0, 2, 2]
    )
    global_rotations[:, dead] = y_rotation(float(first_root_heading))
    for joint in np.flatnonzero(dead):
        parent = int(parents[joint])
        offset = neutral[joint] - neutral[parent]
        positions[:, joint] = positions[:, parent] + np.einsum(
            "fij,j->fi", global_rotations[:, parent], offset
        )
    return positions[:, 0].copy(), local_rotations(global_rotations, parents)


def target_to_world(
    tensors: dict[str, np.ndarray], parents: np.ndarray, neutral: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    joints = tensors["target_global_joint_positions"][0]
    root_offsets = tensors["target_global_root_positions"][0]
    heading = float(tensors["canonical_first_frame_heading_angle"][0])
    rotation = y_rotation(heading)
    roots = (joints[:, 0] + root_offsets) @ rotation.T
    world_origin = tensors["canonical_first_frame_position"][0] @ MUJOCO_TO_MOTION.T
    roots[:, [0, 2]] += world_origin[[0, 2]]
    global_rotations = rotation[None, None] @ tensors["target_global_joint_rotations"][0]
    local = local_rotations(global_rotations, parents)
    return roots, local, forward_kinematics(roots, local, parents, neutral)


def build(
    capture: Path, upstream: Path, support: Path, output: Path,
    expected_replay: Path | None = None,
) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite parity directory: {output}")
    manifest_path = capture / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != CAPTURE_FORMAT or manifest["upstream"]["revision"] != UPSTREAM_REVISION:
        raise ValueError("capture is not the accepted upstream format/revision")
    if not manifest.get("trace", {}).get("target_boundaries"):
        raise ValueError("capture has no target-boundary trace")
    replay_manifest = None
    if expected_replay is not None:
        replay_manifest = json.loads(
            (expected_replay / "manifest.json").read_text(encoding="utf-8")
        )
        if (replay_manifest.get("format") != "motionbricks-upstream-cpu-inference-replay-v1" or
                replay_manifest.get("upstream_revision") != UPSTREAM_REVISION or
                replay_manifest.get("source_capture_manifest_sha256") != sha256(manifest_path)):
            raise ValueError("CPU inference replay does not match the accepted capture")
    if sha256(support) != SUPPORT_SHA256:
        raise ValueError("support skeleton does not match the trusted extraction")
    support_tensors = load_tensors(support)
    parents = np.asarray(support_tensors["joint_parents"], dtype=np.int32)
    neutral = np.asarray(support_tensors["neutral_joints"], dtype=np.float32)
    if parents.shape != (JOINTS,) or neutral.shape != (JOINTS, 3):
        raise ValueError("support skeleton has unexpected dimensions")
    playback = load_tensors(capture / manifest["capture"]["playback"]["file"])
    events = load_jsonl(capture / manifest["capture"]["events"]["file"])

    import mujoco

    scene = upstream / "motionbricks/assets/skeletons/g1/scene_29dof.xml"
    if sha256(scene) != TRUSTED_FILES["motionbricks/assets/skeletons/g1/scene_29dof.xml"]:
        raise ValueError("upstream G1 scene does not match the pinned revision")
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    if model.nq != 36 or model.nbody != 31:
        raise ValueError("upstream G1 MuJoCo topology changed")
    plans: list[Plan] = []
    for index, event in enumerate(events):
        frame = int(event["command_after_output_frame"])
        tensors = load_tensors(capture / f"plan-{index:03d}.safetensors")
        context_roots, context_local = qpos_to_motion(
            np.asarray(playback["context_qpos"][frame], dtype=np.float32),
            model, data, parents, neutral,
        )
        expected_features = tensors["model_features"]
        if expected_replay is not None:
            replay_tensors = load_tensors(expected_replay / f"plan-{index:03d}.safetensors")
            expected_features = replay_tensors["model_features"]
            if (int(replay_tensors["pred_num_tokens"].reshape(-1)[0]) * 4 !=
                    int(event["valid_length"])):
                raise ValueError(f"CPU replay duration differs for plan {index}")
        expected_roots, expected_local, expected_positions = features_to_world(
            np.asarray(expected_features[0], dtype=np.float32),
            np.asarray(tensors["canonical_first_frame_position"][0], dtype=np.float32),
            float(tensors["canonical_first_frame_heading_angle"][0]), parents, neutral,
        )
        target_roots, target_local, target_positions = target_to_world(tensors, parents, neutral)
        move_mujoco = np.asarray(playback["movement_direction"][frame], dtype=np.float32)
        face_mujoco = np.asarray(playback["facing_direction"][frame], dtype=np.float32)
        plans.append(Plan(
            command_frame=frame, expected_frames=int(event["valid_length"]),
            mode=int(event["mode"]), seed=int(event["seed"]),
            movement=move_mujoco @ MUJOCO_TO_MOTION.T,
            facing=face_mujoco @ MUJOCO_TO_MOTION.T,
            context_roots=context_roots, context_local_rotations=context_local,
            expected_roots=expected_roots, expected_local_rotations=expected_local,
            expected_joint_positions=expected_positions,
            target_roots=target_roots, target_local_rotations=target_local,
            target_joint_positions=target_positions,
        ))
    output.mkdir(parents=True)
    fixture_path = output / "open-loop.mbparity"
    write_fixture(fixture_path, Fixture(parents=parents, neutral_joints=neutral, plans=plans))
    result = {
        "format": "motionbricks-open-loop-parity-v1",
        "source": {
            "capture_manifest_sha256": sha256(manifest_path),
            "support_sha256": sha256(support),
            "scene_xml_sha256": sha256(scene),
            "upstream_revision": UPSTREAM_REVISION,
            "expected_device": "cpu" if expected_replay is not None else "cuda",
            "expected_replay_manifest_sha256": (
                sha256(expected_replay / "manifest.json") if expected_replay is not None else None
            ),
        },
        "fixture": {"file": fixture_path.name, "sha256": sha256(fixture_path)},
        "fps": 30, "plans": len(plans), "joints": JOINTS,
        "comparisons": [
            "duration_exact", "target_root", "target_local_rotation", "target_fk_joint",
            "output_root", "output_local_rotation", "output_fk_joint",
        ],
        "pose_token_ids": "not exposed by the accepted observational boundary",
        "playback_blend": "excluded; expected output comes from unblended model_features",
    }
    (output / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--support", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-replay", type=Path)
    args = parser.parse_args()
    build(args.capture.resolve(), args.upstream_root.resolve(), args.support.resolve(),
          args.output.resolve(),
          args.expected_replay.resolve() if args.expected_replay is not None else None)


if __name__ == "__main__":
    main()
