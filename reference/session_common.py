#!/usr/bin/env python3
"""Shared constants and pure helpers for upstream session capture."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


UPSTREAM_REVISION = "a0732b642c0333077e127a2f56ab0014c196bca4"
SCENARIO_FORMAT = "motionbricks-session-scenario-v1"
CAPTURE_FORMAT = "motionbricks-session-capture-v1"
COMPARISON_FORMAT = "motionbricks-session-comparison-v1"
SCENARIO_NAME = "idle-walk-turn-idle-zombie-settle"
FPS = 30
FRAME_COUNT = 345
ARTIFACT_LIMIT_BYTES = 25 * 1024 * 1024

KEY_NAMES = (
    "w", "a", "s", "d", "left", "right", "up", "down", "shift", "ctrl", "enter",
    "x", "z", "c", "v", "b", "r", "t", "f", "g", "q", "e",
)

TRUSTED_FILES = {
    "motionbricks/out/G1-clip.ckpt":
        "84afc7c229473351a24b0a7d79fc47be9dbb81bd12774285f2f60a3c0e9028df",
    "motionbricks/out/motionbricks_vqvae/version_1/checkpoints/model-step=2000000.ckpt":
        "f12a09d46ad390a8e2eecbe7219b2472fcab6b59df0a13f6a40c35cb6da4d99a",
    "motionbricks/out/motionbricks_pose/version_1/checkpoints/model-step=2000000.ckpt":
        "0223c352b308ba638a499cc5c92104da36cb1d04cea2f8ce61d54a2489f853f1",
    "motionbricks/out/motionbricks_root/version_1/checkpoints/model-step=2000000.ckpt":
        "d7299a9b1f5aca35730c36dfe7ea28075708ac8266bf384fe8e2ef9c9aee69c7",
    "motionbricks/out/motionbricks_vqvae/version_1/config.yaml":
        "027a2d7ba5f49cadeadbcc9a6b0c6784d657d5a842e41ff820c5233f1cd6c1f3",
    "motionbricks/out/motionbricks_pose/version_1/config.yaml":
        "273af770d328b458510ee6049fac8e38fa486c8792cc27b3515dddc094a7cd1f",
    "motionbricks/out/motionbricks_root/version_1/config.yaml":
        "b174f03a333f7f7857c3e2b8a32da517caddd198f228efeea2e203d046ce212a",
    "motionbricks/out/motionbricks_vqvae/version_1/hparams.yaml":
        "7af860d6c144ff71ccdb4085b8413be892194ef419031b81e231502c2181576d",
    "motionbricks/out/motionbricks_pose/version_1/hparams.yaml":
        "3c482b9f810a9c4921d2e9fe46dae4e9e70d48ac9ea7d7b19a91fad2490d75e5",
    "motionbricks/out/motionbricks_root/version_1/hparams.yaml":
        "15dfadeab65b8ab1931b4e5257a2f9ebfa57e559510af1bd5baa96cfe2baeaf5",
    "motionbricks/out/motionbricks_vqvae/version_1/stats/motion/mean.npy":
        "ca390f0081e2373ab71e860a3546cb70cc11fdfac6ce525155403e047b66fdea",
    "motionbricks/out/motionbricks_vqvae/version_1/stats/motion/std.npy":
        "fca7dd6135cbe96504b1307a7db97e9004556937a2fc72d139077b962cee6bd7",
    "motionbricks/out/motionbricks_pose/version_1/stats/motion/mean.npy":
        "ca390f0081e2373ab71e860a3546cb70cc11fdfac6ce525155403e047b66fdea",
    "motionbricks/out/motionbricks_pose/version_1/stats/motion/std.npy":
        "fca7dd6135cbe96504b1307a7db97e9004556937a2fc72d139077b962cee6bd7",
    "motionbricks/out/motionbricks_root/version_1/stats/motion/mean.npy":
        "ca390f0081e2373ab71e860a3546cb70cc11fdfac6ce525155403e047b66fdea",
    "motionbricks/out/motionbricks_root/version_1/stats/motion/std.npy":
        "fca7dd6135cbe96504b1307a7db97e9004556937a2fc72d139077b962cee6bd7",
    "motionbricks/out/motionbricks_vqvae/version_1/skeleton/joints.p":
        "8a582b7020d1609a34a9ea5ddfa597c8727b3e726e38cc80ecefe68171f43ecd",
    "motionbricks/out/motionbricks_vqvae/version_1/skeleton/parents.p":
        "4ba0237379480ef33e64b7b8564f1d38dcb3e1ad5bfe01652ea1b7c97ccacb71",
    "motionbricks/out/motionbricks_vqvae/version_1/skeleton/skeleton.yaml":
        "5820a223977a36e04e75824a7f720b57225ac30a067bdd32bc312469f3644050",
    "motionbricks/out/motionbricks_pose/version_1/skeleton/joints.p":
        "8a582b7020d1609a34a9ea5ddfa597c8727b3e726e38cc80ecefe68171f43ecd",
    "motionbricks/out/motionbricks_pose/version_1/skeleton/parents.p":
        "4ba0237379480ef33e64b7b8564f1d38dcb3e1ad5bfe01652ea1b7c97ccacb71",
    "motionbricks/out/motionbricks_pose/version_1/skeleton/skeleton.yaml":
        "5820a223977a36e04e75824a7f720b57225ac30a067bdd32bc312469f3644050",
    "motionbricks/out/motionbricks_root/version_1/skeleton/joints.p":
        "8a582b7020d1609a34a9ea5ddfa597c8727b3e726e38cc80ecefe68171f43ecd",
    "motionbricks/out/motionbricks_root/version_1/skeleton/parents.p":
        "4ba0237379480ef33e64b7b8564f1d38dcb3e1ad5bfe01652ea1b7c97ccacb71",
    "motionbricks/out/motionbricks_root/version_1/skeleton/skeleton.yaml":
        "5820a223977a36e04e75824a7f720b57225ac30a067bdd32bc312469f3644050",
    "motionbricks/assets/skeletons/g1/g1.xml":
        "5d76cf92f00dd49d6eb9fae38d7d38e46886848b602ac691051e886c3bcccfb1",
    "motionbricks/assets/skeletons/g1/g1_29dof.xml":
        "58660a6f1d0d33ffd8ee967ab3860def53e3327d956cb009dd1385ddaf430f56",
    "motionbricks/assets/skeletons/g1/scene_29dof.xml":
        "e254f11acce2ec6f6efa5bf9b15e288bbd0ca29aeeabb1e1d0fea92f65436bbf",
}

SEGMENTS = (
    {"name": "idle", "start": 0, "frames": 60, "style": "idle"},
    {"name": "walk_forward", "start": 60, "frames": 60, "style": "walk"},
    {"name": "turn_right", "start": 120, "frames": 45, "style": "walk"},
    {"name": "released_idle", "start": 165, "frames": 60, "style": "idle"},
    {"name": "zombie_walk", "start": 225, "frames": 60, "style": "walk_zombie"},
    {"name": "released_settle", "start": 285, "frames": 60, "style": "idle"},
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def scenario_control(frame: int) -> dict[str, Any]:
    """Return deterministic keys and camera values for one logical output frame."""
    if frame < 0 or frame >= FRAME_COUNT:
        raise ValueError(f"frame must be in [0, {FRAME_COUNT})")

    keys = {name: False for name in KEY_NAMES}
    camera_azimuth = 0.0
    if frame < 60:
        segment, style = "idle", "idle"
    elif frame < 120:
        segment, style = "walk_forward", "walk"
        keys["w"] = True
    elif frame < 165:
        segment, style = "turn_right", "walk"
        keys["w"] = True
        # A clockwise 90-degree camera turn makes upstream's camera-relative W
        # command turn both the movement and facing vectors to the right.
        camera_azimuth = -90.0 * float(frame - 119) / 45.0
    elif frame < 225:
        segment, style = "released_idle", "idle"
        camera_azimuth = -90.0
    elif frame < 285:
        segment, style = "zombie_walk", "walk_zombie"
        camera_azimuth = -90.0
        keys["w"] = True
        keys["f"] = True
    else:
        segment, style = "released_settle", "idle"
        camera_azimuth = -90.0

    return {
        "frame": frame,
        "segment": segment,
        "expected_style": style,
        "keys": keys,
        "camera": {
            "lookat": [0.0, 0.0, 1.0],
            "distance": 3.0,
            "azimuth_degrees": camera_azimuth,
            "elevation_degrees": -20.0,
        },
    }


def plan_seed(base_seed: int, plan_index: int) -> int:
    """Stable per-plan seed; independent of render timing and failed no-op replans."""
    if base_seed < 0 or plan_index < 0:
        raise ValueError("seeds and plan indices must be non-negative")
    return (base_seed + plan_index * 2_654_435_761) % (2**31 - 1)


def upstream_run_seed(requested_seed: int) -> int:
    """Match iteration one of upstream interactive_demo_g1.py."""
    if requested_seed < 0:
        raise ValueError("seed must be non-negative")
    return requested_seed * (1 + 2333) * 2333 % (2**32 - 1)


def scenario_manifest() -> dict[str, Any]:
    return {
        "format": SCENARIO_FORMAT,
        "name": SCENARIO_NAME,
        "fps": FPS,
        "frame_count": FRAME_COUNT,
        "command_phase": "after_emitted_frame",
        "segments": list(SEGMENTS),
        "turn": {
            "mechanism": "camera_azimuth_ramp_while_holding_w",
            "start_degrees": 0.0,
            "end_degrees": -90.0,
        },
    }
