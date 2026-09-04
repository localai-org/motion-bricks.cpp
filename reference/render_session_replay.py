#!/usr/bin/env python3
"""Render an observational replay through MuJoCo without running inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from session_common import sha256
from session_replay import Replay, TARGET_FRAMES, read_replay


TARGET_COLORS = (
    (1.00, 0.72, 0.20, 0.28),
    (1.00, 0.43, 0.12, 0.38),
    (1.00, 0.18, 0.18, 0.54),
    (1.00, 0.08, 0.62, 0.82),
)
PATH_COLOR = (0.10, 0.95, 0.62, 0.72)
TARGET_PATH_COLOR = (1.00, 0.34, 0.12, 0.80)


def motion_to_mujoco(values: np.ndarray) -> np.ndarray:
    """Map y-up/z-forward motion coordinates to z-up/x-forward MuJoCo."""
    return np.asarray(values, dtype=np.float64)[..., [2, 0, 1]]


def add_sphere(scene, mujoco, position, radius: float, color) -> None:
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError("MuJoCo visualization geometry capacity exceeded")
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
        np.asarray([radius, 0.0, 0.0]), np.asarray(position, dtype=np.float64),
        np.eye(3).reshape(-1), np.asarray(color, dtype=np.float32),
    )
    scene.ngeom += 1


def add_bone(scene, mujoco, start, end, radius: float, color) -> None:
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError("MuJoCo visualization geometry capacity exceeded")
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.ones(3), np.zeros(3),
        np.eye(3).reshape(-1), np.asarray(color, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom, mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
        np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64),
    )
    scene.ngeom += 1


def add_skeleton(scene, mujoco, positions, parents, color, radius: float) -> None:
    points = motion_to_mujoco(positions)
    for joint, point in enumerate(points):
        add_sphere(scene, mujoco, point, radius * (1.7 if joint == 0 else 1.15), color)
        parent = int(parents[joint])
        if parent >= 0:
            add_bone(scene, mujoco, points[parent], point, radius, color)


def add_path(scene, mujoco, positions, color, radius: float) -> None:
    if len(positions) < 2:
        return
    points = motion_to_mujoco(positions.copy())
    points[:, 2] = 0.018
    for start, end in zip(points[:-1], points[1:]):
        if np.linalg.norm(end - start) > 1e-7:
            add_bone(scene, mujoco, start, end, radius, color)


def font(size: int):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def overlay(image: np.ndarray, replay: Replay, frame: int, mode_names: list[str]) -> Image.Image:
    result = Image.fromarray(image)
    draw = ImageDraw.Draw(result, "RGBA")
    plan = int(replay.frame_plans[frame])
    mode_index = int(replay.modes[frame])
    mode = mode_names[mode_index] if 0 <= mode_index < len(mode_names) else str(mode_index)
    duration = int(replay.plan_valid_lengths[plan]) if plan >= 0 else 0
    draw.rounded_rectangle((24, 22, 560, 122), radius=12, fill=(5, 9, 15, 205))
    draw.text((42, 34), f"UPSTREAM MUJOCO  |  frame {frame:03d}/{replay.frame_count - 1}",
              font=font(20), fill=(225, 239, 248, 255))
    plan_text = "pre-plan" if plan < 0 else f"plan {plan:02d}  ·  {mode}  ·  {duration} generated frames"
    draw.text((42, 73), plan_text, font=font(18), fill=(115, 235, 193, 255))
    draw.rounded_rectangle((24, result.height - 70, 750, result.height - 22), radius=10,
                           fill=(5, 9, 15, 190))
    x = 42
    draw.rectangle((x, result.height - 54, x + 24, result.height - 34), fill=(190, 195, 202, 255))
    draw.text((x + 34, result.height - 58), "animated G1", font=font(16), fill=(232, 238, 245, 255))
    x += 160
    for target, color in enumerate(TARGET_COLORS):
        rgba = tuple(round(component * 255) for component in color)
        draw.rectangle((x, result.height - 54, x + 22, result.height - 34), fill=rgba)
        draw.text((x + 29, result.height - 58), f"T{target}", font=font(16), fill=(232, 238, 245, 255))
        x += 80
    draw.line((x, result.height - 44, x + 30, result.height - 44), fill=(26, 242, 158, 255), width=4)
    draw.text((x + 38, result.height - 58), "root path", font=font(16), fill=(232, 238, 245, 255))
    return result


def render(replay_path: Path, manifest_path: Path, upstream: Path, output: Path, max_frames: int) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite render directory: {output}")
    replay = read_replay(replay_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("format") != "motionbricks-portable-replay-v1" or
            manifest.get("replay", {}).get("sha256") != sha256(replay_path) or
            manifest.get("frame_count") != replay.frame_count or
            manifest.get("joint_count") != replay.joint_count or
            manifest.get("plan_count") != replay.plan_count):
        raise ValueError("replay manifest does not match the binary artifact")
    mode_names = manifest["mode_names"]

    import mujoco

    model = mujoco.MjModel.from_xml_path(
        str(upstream / "motionbricks/assets/skeletons/g1/scene_29dof.xml")
    )
    model.vis.global_.offwidth = 1280
    model.vis.global_.offheight = 720
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 4.3
    camera.azimuth = 132.0
    camera.elevation = -18.0
    count = replay.frame_count if max_frames <= 0 else min(max_frames, replay.frame_count)
    output.mkdir(parents=True)
    frames_dir = output / "frames"
    snapshots_dir = output / "snapshots"
    frames_dir.mkdir()
    snapshots_dir.mkdir()
    selected = {0, 60, 124, 164, 172, 225, 273, 344}

    with mujoco.Renderer(model, height=720, width=1280, max_geom=2048) as renderer:
        for frame in range(count):
            data.qpos[:] = replay.qpos[frame]
            mujoco.mj_forward(model, data)
            camera.lookat[:] = data.qpos[:3] + np.asarray([0.0, 0.0, 0.56])
            renderer.update_scene(data, camera=camera)
            plan = int(replay.frame_plans[frame])
            if plan >= 0:
                targets = replay.target_positions[plan]
                for target in range(TARGET_FRAMES):
                    add_skeleton(
                        renderer.scene, mujoco, targets[target], replay.parents,
                        TARGET_COLORS[target], 0.012,
                    )
                add_path(renderer.scene, mujoco, targets[:, 0], TARGET_PATH_COLOR, 0.008)
            trail_start = max(0, frame - 90)
            add_path(
                renderer.scene, mujoco, replay.joint_positions[trail_start:frame + 1, 0],
                PATH_COLOR, 0.008,
            )
            rendered = overlay(renderer.render(), replay, frame, mode_names)
            frame_path = frames_dir / f"frame-{frame:04d}.png"
            rendered.save(frame_path, optimize=True)
            if frame in selected or frame == count - 1:
                rendered.save(snapshots_dir / f"frame-{frame:04d}.png", optimize=True)

    summary = {
        "format": "motionbricks-mujoco-render-v1",
        "replay": replay_path.name,
        "frames_rendered": count,
        "fps": replay.fps,
        "resolution": [1280, 720],
        "camera_subject": "animated_qpos_root",
        "target_ghosts": TARGET_FRAMES,
    }
    (output / "render.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()
    render(
        args.replay.resolve(), args.manifest.resolve(), args.upstream_root.resolve(),
        args.output.resolve(), args.max_frames,
    )


if __name__ == "__main__":
    main()
