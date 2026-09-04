#!/usr/bin/env python3
"""Bounded binary format for independent open-loop agent parity plans."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


MAGIC = b"MBPARI1\0"
VERSION = 1
FPS = 30
JOINTS = 34
CONTEXT_FRAMES = 4
TARGET_FRAMES = 4
HEADER = struct.Struct("<8s7I")
PLAN_HEADER = struct.Struct("<IIiIQ6f")


@dataclass
class Plan:
    command_frame: int
    expected_frames: int
    mode: int
    seed: int
    movement: np.ndarray
    facing: np.ndarray
    context_roots: np.ndarray
    context_local_rotations: np.ndarray
    expected_roots: np.ndarray
    expected_local_rotations: np.ndarray
    expected_joint_positions: np.ndarray
    target_roots: np.ndarray
    target_local_rotations: np.ndarray
    target_joint_positions: np.ndarray


@dataclass
class Fixture:
    parents: np.ndarray
    neutral_joints: np.ndarray
    plans: list[Plan]
    fps: int = FPS


def _array(value, dtype, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.shape != shape:
        raise ValueError(f"{name} has shape {result.shape}, expected {shape}")
    if np.issubdtype(result.dtype, np.floating) and not np.isfinite(result).all():
        raise ValueError(f"{name} contains a non-finite value")
    return np.ascontiguousarray(result)


def _quaternions(value: np.ndarray, name: str) -> None:
    norms = np.sum(value * value, axis=-1)
    if np.any(norms <= 0.99) or np.any(norms >= 1.01):
        raise ValueError(f"{name} contains a non-unit quaternion")


def validate(fixture: Fixture) -> Fixture:
    if fixture.fps <= 0 or fixture.fps > 1000 or not fixture.plans or len(fixture.plans) > 10000:
        raise ValueError("fixture dimensions are outside supported bounds")
    fixture.parents = _array(fixture.parents, "<i4", (JOINTS,), "parents")
    fixture.neutral_joints = _array(
        fixture.neutral_joints, "<f4", (JOINTS, 3), "neutral_joints"
    )
    if fixture.parents[0] != -1:
        raise ValueError("root parent must be -1")
    for joint, parent in enumerate(fixture.parents):
        if joint and (parent < 0 or parent >= joint):
            raise ValueError(f"invalid parent {parent} for joint {joint}")
    for index, plan in enumerate(fixture.plans):
        prefix = f"plan[{index}]"
        if not (24 <= plan.expected_frames <= 64 and plan.expected_frames % 4 == 0):
            raise ValueError(f"{prefix} has an invalid expected frame count")
        if plan.mode < 0 or plan.mode >= 15 or plan.seed < 0 or plan.seed >= 1 << 64:
            raise ValueError(f"{prefix} has invalid command metadata")
        plan.movement = _array(plan.movement, "<f4", (3,), prefix + ".movement")
        plan.facing = _array(plan.facing, "<f4", (3,), prefix + ".facing")
        plan.context_roots = _array(
            plan.context_roots, "<f4", (CONTEXT_FRAMES, 3), prefix + ".context_roots"
        )
        plan.context_local_rotations = _array(
            plan.context_local_rotations, "<f4", (CONTEXT_FRAMES, JOINTS, 4),
            prefix + ".context_local_rotations",
        )
        _quaternions(plan.context_local_rotations, prefix + ".context_local_rotations")
        frames = plan.expected_frames
        plan.expected_roots = _array(
            plan.expected_roots, "<f4", (frames, 3), prefix + ".expected_roots"
        )
        plan.expected_local_rotations = _array(
            plan.expected_local_rotations, "<f4", (frames, JOINTS, 4),
            prefix + ".expected_local_rotations",
        )
        _quaternions(plan.expected_local_rotations, prefix + ".expected_local_rotations")
        plan.expected_joint_positions = _array(
            plan.expected_joint_positions, "<f4", (frames, JOINTS, 3),
            prefix + ".expected_joint_positions",
        )
        plan.target_roots = _array(
            plan.target_roots, "<f4", (TARGET_FRAMES, 3), prefix + ".target_roots"
        )
        plan.target_local_rotations = _array(
            plan.target_local_rotations, "<f4", (TARGET_FRAMES, JOINTS, 4),
            prefix + ".target_local_rotations",
        )
        _quaternions(plan.target_local_rotations, prefix + ".target_local_rotations")
        plan.target_joint_positions = _array(
            plan.target_joint_positions, "<f4", (TARGET_FRAMES, JOINTS, 3),
            prefix + ".target_joint_positions",
        )
    return fixture


def write_fixture(path: Path, fixture: Fixture) -> None:
    fixture = validate(fixture)
    with path.open("xb") as stream:
        stream.write(HEADER.pack(
            MAGIC, VERSION, fixture.fps, len(fixture.plans), JOINTS,
            CONTEXT_FRAMES, TARGET_FRAMES, 0,
        ))
        stream.write(fixture.parents.tobytes())
        stream.write(fixture.neutral_joints.tobytes())
        for plan in fixture.plans:
            stream.write(PLAN_HEADER.pack(
                plan.command_frame, plan.expected_frames, plan.mode, 0, plan.seed,
                *plan.movement.tolist(), *plan.facing.tolist(),
            ))
            for value in (
                plan.context_roots, plan.context_local_rotations,
                plan.expected_roots, plan.expected_local_rotations,
                plan.expected_joint_positions, plan.target_roots,
                plan.target_local_rotations, plan.target_joint_positions,
            ):
                stream.write(value.tobytes(order="C"))


def read_fixture(path: Path) -> Fixture:
    raw = path.read_bytes()
    if len(raw) < HEADER.size:
        raise ValueError("fixture is shorter than its header")
    magic, version, fps, count, joints, context, targets, flags = HEADER.unpack_from(raw)
    if (magic != MAGIC or version != VERSION or joints != JOINTS or
            context != CONTEXT_FRAMES or targets != TARGET_FRAMES or flags != 0):
        raise ValueError("unsupported parity fixture header")
    if fps <= 0 or fps > 1000 or count <= 0 or count > 10000:
        raise ValueError("fixture header dimensions are outside supported bounds")
    offset = HEADER.size

    def take(dtype, shape: tuple[int, ...], name: str) -> np.ndarray:
        nonlocal offset
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        if size > len(raw) - offset:
            raise ValueError(f"fixture is truncated in {name}")
        result = np.frombuffer(raw, dtype=dtype, count=int(np.prod(shape)), offset=offset)
        offset += size
        return result.reshape(shape).copy()

    parents = take("<i4", (JOINTS,), "parents")
    neutral = take("<f4", (JOINTS, 3), "neutral joints")
    plans: list[Plan] = []
    for index in range(count):
        if PLAN_HEADER.size > len(raw) - offset:
            raise ValueError(f"fixture is truncated in plan {index} header")
        values = PLAN_HEADER.unpack_from(raw, offset)
        offset += PLAN_HEADER.size
        command_frame, frames, mode, reserved, seed, *directions = values
        if reserved != 0 or frames < 24 or frames > 64 or frames % 4:
            raise ValueError(f"fixture has an invalid plan {index} header")
        plans.append(Plan(
            command_frame=command_frame, expected_frames=frames, mode=mode, seed=seed,
            movement=np.asarray(directions[:3], dtype=np.float32),
            facing=np.asarray(directions[3:], dtype=np.float32),
            context_roots=take("<f4", (CONTEXT_FRAMES, 3), "context roots"),
            context_local_rotations=take(
                "<f4", (CONTEXT_FRAMES, JOINTS, 4), "context rotations"
            ),
            expected_roots=take("<f4", (frames, 3), "expected roots"),
            expected_local_rotations=take(
                "<f4", (frames, JOINTS, 4), "expected rotations"
            ),
            expected_joint_positions=take(
                "<f4", (frames, JOINTS, 3), "expected positions"
            ),
            target_roots=take("<f4", (TARGET_FRAMES, 3), "target roots"),
            target_local_rotations=take(
                "<f4", (TARGET_FRAMES, JOINTS, 4), "target rotations"
            ),
            target_joint_positions=take(
                "<f4", (TARGET_FRAMES, JOINTS, 3), "target positions"
            ),
        ))
    if offset != len(raw):
        raise ValueError(f"fixture has {len(raw) - offset} trailing bytes")
    return validate(Fixture(parents=parents, neutral_joints=neutral, plans=plans, fps=fps))
