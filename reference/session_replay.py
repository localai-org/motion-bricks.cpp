#!/usr/bin/env python3
"""Portable, bounded MotionBricks observational replay format."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


MAGIC = b"MBRPLY1\0"
VERSION = 1
HEADER = struct.Struct("<8s8I")
TARGET_FRAMES = 4


@dataclass
class Replay:
    fps: int
    parents: np.ndarray
    modes: np.ndarray
    frame_plans: np.ndarray
    qpos: np.ndarray
    joint_positions: np.ndarray
    plan_frames: np.ndarray
    plan_modes: np.ndarray
    plan_valid_lengths: np.ndarray
    target_positions: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(self.qpos.shape[0])

    @property
    def joint_count(self) -> int:
        return int(self.parents.shape[0])

    @property
    def qpos_count(self) -> int:
        return int(self.qpos.shape[1])

    @property
    def plan_count(self) -> int:
        return int(self.plan_frames.shape[0])


def _array(value, dtype, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.shape != shape:
        raise ValueError(f"{name} has shape {result.shape}, expected {shape}")
    if np.issubdtype(result.dtype, np.floating) and not np.isfinite(result).all():
        raise ValueError(f"{name} contains a non-finite value")
    return np.ascontiguousarray(result)


def validate(replay: Replay) -> Replay:
    frames, joints, qpos, plans = (
        replay.frame_count, replay.joint_count, replay.qpos_count, replay.plan_count
    )
    if replay.fps <= 0 or frames <= 0 or joints <= 0 or joints > 64 or qpos <= 0 or plans <= 0:
        raise ValueError("replay dimensions are outside supported bounds")
    replay.parents = _array(replay.parents, "<i4", (joints,), "parents")
    if replay.parents[0] != -1:
        raise ValueError("root parent must be -1")
    for joint, parent in enumerate(replay.parents):
        if joint > 0 and (parent < 0 or parent >= joint):
            raise ValueError(f"invalid parent {parent} for joint {joint}")
    replay.modes = _array(replay.modes, "<i4", (frames,), "modes")
    replay.frame_plans = _array(replay.frame_plans, "<i4", (frames,), "frame_plans")
    if np.any(replay.frame_plans < -1) or np.any(replay.frame_plans >= plans):
        raise ValueError("frame plan index is out of bounds")
    replay.qpos = _array(replay.qpos, "<f4", (frames, qpos), "qpos")
    replay.joint_positions = _array(
        replay.joint_positions, "<f4", (frames, joints, 3), "joint_positions"
    )
    replay.plan_frames = _array(replay.plan_frames, "<u4", (plans,), "plan_frames")
    replay.plan_modes = _array(replay.plan_modes, "<i4", (plans,), "plan_modes")
    replay.plan_valid_lengths = _array(
        replay.plan_valid_lengths, "<u4", (plans,), "plan_valid_lengths"
    )
    replay.target_positions = _array(
        replay.target_positions, "<f4", (plans, TARGET_FRAMES, joints, 3),
        "target_positions",
    )
    if np.any(replay.plan_frames >= frames) or np.any(np.diff(replay.plan_frames) <= 0):
        raise ValueError("plan frames must be strictly increasing and inside the session")
    return replay


def write_replay(path: Path, replay: Replay) -> None:
    replay = validate(replay)
    header = HEADER.pack(
        MAGIC, VERSION, replay.fps, replay.frame_count, replay.joint_count,
        replay.qpos_count, replay.plan_count, TARGET_FRAMES, 0,
    )
    with path.open("xb") as stream:
        stream.write(header)
        for value in (
            replay.parents, replay.modes, replay.frame_plans, replay.qpos,
            replay.joint_positions, replay.plan_frames, replay.plan_modes,
            replay.plan_valid_lengths, replay.target_positions,
        ):
            stream.write(value.tobytes(order="C"))


def read_replay(path: Path) -> Replay:
    raw = path.read_bytes()
    if len(raw) < HEADER.size:
        raise ValueError("replay is shorter than its header")
    magic, version, fps, frames, joints, qpos, plans, target_frames, flags = HEADER.unpack_from(raw)
    if magic != MAGIC or version != VERSION or target_frames != TARGET_FRAMES or flags != 0:
        raise ValueError("unsupported replay header")
    if not (0 < frames <= 1_000_000 and 0 < joints <= 64 and 0 < qpos <= 256 and 0 < plans <= frames):
        raise ValueError("replay header dimensions are outside supported bounds")
    offset = HEADER.size

    def take(dtype, shape: tuple[int, ...], name: str) -> np.ndarray:
        nonlocal offset
        count = int(np.prod(shape))
        size = count * np.dtype(dtype).itemsize
        if size > len(raw) - offset:
            raise ValueError(f"replay is truncated in {name}")
        result = np.frombuffer(raw, dtype=dtype, count=count, offset=offset).reshape(shape).copy()
        offset += size
        return result

    replay = Replay(
        fps=fps,
        parents=take("<i4", (joints,), "parents"),
        modes=take("<i4", (frames,), "modes"),
        frame_plans=take("<i4", (frames,), "frame_plans"),
        qpos=take("<f4", (frames, qpos), "qpos"),
        joint_positions=take("<f4", (frames, joints, 3), "joint_positions"),
        plan_frames=take("<u4", (plans,), "plan_frames"),
        plan_modes=take("<i4", (plans,), "plan_modes"),
        plan_valid_lengths=take("<u4", (plans,), "plan_valid_lengths"),
        target_positions=take(
            "<f4", (plans, TARGET_FRAMES, joints, 3), "target_positions"
        ),
    )
    if offset != len(raw):
        raise ValueError(f"replay has {len(raw) - offset} trailing bytes")
    return validate(replay)
