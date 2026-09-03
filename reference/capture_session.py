#!/usr/bin/env python3
"""Capture a black-box navigation session from pinned, unmodified upstream."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from session_common import (
    ARTIFACT_LIMIT_BYTES,
    CAPTURE_FORMAT,
    FRAME_COUNT,
    TRUSTED_FILES,
    UPSTREAM_REVISION,
    plan_seed,
    scenario_control,
    scenario_manifest,
    sha256,
    upstream_run_seed,
)


PACKAGE_NAMES = (
    "torch", "numpy", "mujoco", "scipy", "hydra-core", "omegaconf",
    "pytorch-lightning", "transformers", "pynput", "matplotlib",
    "vector-quantize-pytorch", "colorlog", "adam-atan2-pytorch", "safetensors",
)

BASE_PACKAGE_VERSIONS = {
    "torch": "2.7.0+cu128",
    "triton": "3.3.0",
}


def git_output(upstream: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={upstream}", "-C", str(upstream), *arguments],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def validate_upstream(upstream: Path) -> dict[str, str]:
    revision = git_output(upstream, "rev-parse", "HEAD")
    if revision != UPSTREAM_REVISION:
        raise RuntimeError(f"upstream revision is {revision}, expected {UPSTREAM_REVISION}")
    # Restrict cleanliness to executable/configuration source. A full status on
    # a read-only checkout asks Git LFS to clean binary meshes into .git/lfs/tmp.
    dirty = git_output(
        upstream,
        "status", "--porcelain", "--untracked-files=no", "--",
        "motionbricks/motionbricks", "motionbricks/scripts", "motionbricks/setup.py",
    )
    if dirty:
        raise RuntimeError(f"upstream has modified tracked files:\n{dirty}")

    identities: dict[str, str] = {}
    for relative, expected in TRUSTED_FILES.items():
        path = upstream / relative
        if not path.is_file():
            raise FileNotFoundError(f"trusted upstream input is missing: {path}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"SHA-256 mismatch for {path}: {actual}, expected {expected}")
        identities[relative] = actual
    return identities


def package_versions() -> dict[str, str]:
    versions = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def locked_package_versions(lock_path: Path) -> dict[str, str]:
    """Parse the deliberately simple, exact requirements lock."""
    from packaging.utils import canonicalize_name

    expected: dict[str, str] = {}
    for line_number, raw in enumerate(lock_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([^=\s]+)==([^\s]+)", line)
        if match is None:
            raise RuntimeError(f"non-exact requirement at {lock_path}:{line_number}: {line}")
        name = canonicalize_name(match.group(1))
        if name in expected:
            raise RuntimeError(f"duplicate locked package at {lock_path}:{line_number}: {name}")
        expected[name] = match.group(2)
    return expected


def validate_python_environment(reference_dir: Path) -> dict[str, str]:
    """Require every locked layer package and pinned base package exactly."""
    from packaging.utils import canonicalize_name

    expected = locked_package_versions(reference_dir / "session-requirements.lock")
    expected.update(BASE_PACKAGE_VERSIONS)
    installed = {
        canonicalize_name(distribution.metadata["Name"]): distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    }
    mismatches = {
        name: {"expected": version, "actual": installed.get(name, "not-installed")}
        for name, version in expected.items()
        if installed.get(name) != version
    }
    if mismatches:
        raise RuntimeError(
            "reference Python environment differs from its lock:\n"
            + json.dumps(mismatches, indent=2, sort_keys=True)
        )
    return dict(sorted(expected.items()))


def os_package_versions() -> dict[str, str]:
    """Record the complete resolved Debian package set in the image."""
    try:
        output = subprocess.check_output(
            ["dpkg-query", "-W", "-f=${binary:Package}\t${Version}\n"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {}
    return dict(sorted(line.split("\t", 1) for line in output.splitlines() if "\t" in line))


def nvidia_driver_version() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.STDOUT,
        ).splitlines()[0].strip()
    except (FileNotFoundError, subprocess.CalledProcessError, IndexError):
        return "unavailable"


def clone_cpu(value, torch):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu").contiguous().clone()
    return torch.as_tensor(value).detach().to(device="cpu").contiguous().clone()


def make_demo_args(upstream: Path) -> argparse.Namespace:
    root = upstream / "motionbricks"
    return argparse.Namespace(
        humanoid_scene_xml=str(root / "assets/skeletons/g1/scene_29dof.xml"),
        skeleton_xml=str(root / "assets/skeletons/g1/g1.xml"),
        clips_ckpt=str(root / "out/G1-clip.ckpt"),
        result_dir=str(root / "out"),
        data_root=str(root / "datasets"),
        explicit_dataset_folder=None,
        reprocess_clips=0,
        controller="wasd",
        lookat_movement_direction=0,
        pre_filter_qpos=1,
        source_root_realignment=1,
        target_root_realignment=1,
        force_canonicalization=1,
        skip_ending_target_cond=0,
        random_speed_scale=0,
        speed_scale=[0.8, 1.2],
        generate_dt=2.0,
        use_qpos=1,
        planner="default",
        allowed_mode=None,
        clips="G1",
        EXP="default",
        return_model_configs=True,
        return_dataloader=True,
        recording_dir=None,
    )


def tensor_inventory(tensors: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        name: {"shape": list(value.shape), "dtype": str(value.dtype).removeprefix("torch.")}
        for name, value in sorted(tensors.items())
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def preflight(upstream: Path) -> None:
    if os.environ.get("MOTIONBRICKS_REFERENCE_CONTAINER") != "1":
        raise RuntimeError("session preflight must run in the pinned reference session container")
    identities = validate_upstream(upstream)
    reference_dir = Path(__file__).resolve().parent
    locked_packages = validate_python_environment(reference_dir)
    system_packages = os_package_versions()
    sys.path.insert(0, str(upstream / "motionbricks"))
    os.environ.setdefault("PYNPUT_BACKEND", "dummy")

    import mujoco
    import torch
    from motionbricks.motion_backbone.demo.utils import navigation_demo  # noqa: F401

    print(json.dumps({
        "upstream_revision": UPSTREAM_REVISION,
        "trusted_file_count": len(identities),
        "locked_python_package_count": len(locked_packages),
        "resolved_os_package_count": len(system_packages),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "mujoco": mujoco.__version__,
    }, indent=2, sort_keys=True))


def capture(upstream: Path, output: Path, requested_seed: int, artifact_limit: int) -> None:
    if os.environ.get("MOTIONBRICKS_REFERENCE_CONTAINER") != "1":
        raise RuntimeError("session capture must run in the pinned reference session container")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing capture directory: {output}")

    identities = validate_upstream(upstream)
    reference_dir = Path(__file__).resolve().parent
    validate_python_environment(reference_dir)
    system_packages = os_package_versions()
    harness_identities = {
        name: sha256(reference_dir / name)
        for name in (
            "Dockerfile.session", "session-requirements.lock",
            "session_common.py", "capture_session.py",
        )
    }
    sys.path.insert(0, str(upstream / "motionbricks"))
    os.environ.setdefault("PYNPUT_BACKEND", "dummy")

    import numpy as np
    import mujoco
    import torch
    from safetensors.torch import save_file
    from motionbricks.motion_backbone.demo.utils import navigation_demo

    if not torch.cuda.is_available():
        raise RuntimeError("upstream navigation_demo requires a visible CUDA device")

    # Upstream hparams contain result paths relative to the MotionBricks project
    # root, matching the released interactive demo's documented invocation.
    os.chdir(upstream / "motionbricks")
    demo = navigation_demo(make_demo_args(upstream))
    run_seed = upstream_run_seed(requested_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    demo.full_agent.reset()

    controls: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    emitted_qpos = []
    context_qpos = []
    context_features = []
    movement = []
    facing = []
    modes = []
    allowed_tokens = []
    plan_seeds = []
    valid_lengths = []
    plans: dict[int, dict[str, Any]] = {}
    pending_features: int | None = None
    next_plan = 0

    camera = SimpleNamespace(
        cam=SimpleNamespace(
            lookat=np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
            distance=3.0,
            azimuth=0.0,
            elevation=-20.0,
        )
    )

    for frame in range(FRAME_COUNT):
        scripted = scenario_control(frame)
        camera.cam.lookat[:] = scripted["camera"]["lookat"]
        camera.cam.distance = scripted["camera"]["distance"]
        camera.cam.azimuth = scripted["camera"]["azimuth_degrees"]
        camera.cam.elevation = scripted["camera"]["elevation_degrees"]

        # Preserve upstream's observable order exactly: playback advances before
        # this logical frame's command is turned into a possible future buffer.
        qpos = demo.full_agent.get_next_frame()
        motion_context = demo.full_agent.get_context_motion_features()
        qpos_context = demo.full_agent.get_context_mujoco_qpos()
        demo.mj_data.qpos[:] = qpos

        signals = demo.controller.generate_control_signals(
            camera,
            demo.mj_model,
            demo.mj_data,
            visualize=False,
            control_info={
                "key_pressed": scripted["keys"],
                "force_idle": False,
                "allowed_mode": None,
            },
        )
        seed = plan_seed(requested_seed, next_plan)

        emitted_qpos.append(clone_cpu(qpos, torch))
        context_qpos.append(clone_cpu(qpos_context[0], torch))
        context_features.append(clone_cpu(motion_context[0], torch))
        movement.append(clone_cpu(signals["movement_direction"][0], torch))
        facing.append(clone_cpu(signals["facing_direction"][0], torch))
        modes.append(clone_cpu(signals["mode"].reshape(-1), torch))
        allowed_tokens.append(clone_cpu(signals["allowed_pred_num_tokens"][0], torch))
        plan_seeds.append(seed)

        signals["random_seed"] = torch.tensor([seed], dtype=torch.int64)
        signals["context_mujoco_qpos"] = qpos_context
        with torch.no_grad():
            generated = demo.full_agent.generate_new_frames(
                signals, demo.controller.get_controller_dt() * 2.0
            )
        mujoco.mj_forward(demo.mj_model, demo.mj_data)

        second = generated[1]
        replanned = isinstance(second, torch.Tensor) and second.numel() == 1
        valid_length = int(second.item()) if replanned else -1
        valid_lengths.append(valid_length)

        if replanned:
            if pending_features is not None:
                raise RuntimeError("a new plan arrived before the previous public feature result")
            plans[next_plan] = {"qpos": clone_cpu(generated[0], torch)}
            pending_features = next_plan
            events.append({
                "type": "replan",
                "plan": next_plan,
                "command_after_output_frame": frame,
                "first_generated_playback_frame": frame + 1,
                "segment": scripted["segment"],
                "expected_style": scripted["expected_style"],
                "mode": int(signals["mode"].item()),
                "seed": seed,
                "valid_length": valid_length,
            })
            next_plan += 1
        elif pending_features is not None:
            # The no-replan public return is (model_features, qpos), whereas a
            # replan returns (qpos, valid_length). Capture it on the next frame.
            plans[pending_features]["model_features"] = clone_cpu(generated[0], torch)
            pending_features = None

        controls.append({
            **scripted,
            "plan_seed_candidate": seed,
            "mode": int(signals["mode"].item()),
            "movement_direction": movement[-1].tolist(),
            "facing_direction": facing[-1].tolist(),
            "allowed_pred_num_tokens": allowed_tokens[-1].tolist(),
            "replanned": replanned,
            "valid_length": valid_length,
        })

    if pending_features is not None:
        raise RuntimeError("capture ended before the final plan exposed its model features")
    if not events:
        raise RuntimeError("upstream session did not produce a planning event")
    if len(plans) != len(events) or sorted(plans) != list(range(len(events))):
        raise RuntimeError("planning artifacts do not form one contiguous record per replan event")
    for event in events:
        valid_length = event["valid_length"]
        if valid_length < 24 or valid_length > 64 or valid_length % 4:
            raise RuntimeError(f"invalid generated frame count in plan {event['plan']}: {valid_length}")
        allowed = allowed_tokens[event["command_after_output_frame"]].reshape(-1)
        token_index = valid_length // 4 - 6
        if token_index >= allowed.numel() or int(allowed[token_index].item()) != 1:
            raise RuntimeError(f"plan {event['plan']} selected a disallowed duration")
        tensors = plans[event["plan"]]
        if set(tensors) != {"qpos", "model_features"}:
            raise RuntimeError(f"plan {event['plan']} is missing a public output tensor")
        if tensors["qpos"].ndim != 3 or tensors["qpos"].shape[:2] != (1, valid_length):
            raise RuntimeError(f"plan {event['plan']} has inconsistent qpos shape")
        if (tensors["model_features"].ndim != 3 or
                tensors["model_features"].shape[:2] != (1, valid_length)):
            raise RuntimeError(f"plan {event['plan']} has inconsistent model-feature shape")

    output.mkdir(parents=True)
    playback = {
        "emitted_qpos": torch.stack(emitted_qpos),
        "context_qpos": torch.stack(context_qpos),
        "context_motion_features": torch.stack(context_features),
        "movement_direction": torch.stack(movement),
        "facing_direction": torch.stack(facing),
        "mode": torch.stack(modes),
        "allowed_pred_num_tokens": torch.stack(allowed_tokens),
        "plan_seed_candidate": torch.tensor(plan_seeds, dtype=torch.int64),
        "valid_length": torch.tensor(valid_lengths, dtype=torch.int32),
    }
    save_file(playback, output / "playback.safetensors")
    write_jsonl(output / "controls.jsonl", controls)
    write_jsonl(output / "events.jsonl", events)

    plan_artifacts = []
    for index, tensors in sorted(plans.items()):
        path = output / f"plan-{index:03d}.safetensors"
        save_file(tensors, path)
        plan_artifacts.append({
            "file": path.name,
            "sha256": sha256(path),
            "tensors": tensor_inventory(tensors),
        })

    artifact_bytes = sum(path.stat().st_size for path in output.iterdir() if path.is_file())
    if artifact_bytes > artifact_limit:
        raise RuntimeError(
            f"capture artifacts use {artifact_bytes} bytes, exceeding the {artifact_limit}-byte limit"
        )

    properties = torch.cuda.get_device_properties(0)
    manifest = {
        "format": CAPTURE_FORMAT,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "harness": harness_identities,
        "upstream": {
            "revision": UPSTREAM_REVISION,
            "tracked_source_clean": True,
            "trusted_files": identities,
        },
        "scenario": scenario_manifest(),
        "seeds": {
            "requested": requested_seed,
            "upstream_run": run_seed,
            "per_plan_algorithm": "(requested + plan_index * 2654435761) mod (2^31 - 1)",
        },
        "demo_flags": {
            "controller": "wasd",
            "use_qpos": True,
            "generate_dt": 2.0,
            "pre_filter_qpos": True,
            "source_root_realignment": True,
            "target_root_realignment": True,
            "force_canonicalization": True,
            "skip_ending_target_cond": False,
            "random_speed_scale": False,
            "lookat_movement_direction": False,
        },
        "environment": {
            "container_required": True,
            "container_base": os.environ.get("MOTIONBRICKS_SESSION_BASE_IMAGE", "unknown"),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": package_versions(),
            "os_packages": system_packages,
            "cuda_runtime": torch.version.cuda,
            "nvidia_driver": nvidia_driver_version(),
            "cudnn": torch.backends.cudnn.version(),
            "gpu_name": properties.name,
            "gpu_compute_capability": [properties.major, properties.minor],
            "gpu_total_memory": properties.total_memory,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "capture": {
            "plan_count": len(plans),
            "artifact_bytes_before_manifest": artifact_bytes,
            "artifact_limit_bytes": artifact_limit,
            "playback": {
                "file": "playback.safetensors",
                "sha256": sha256(output / "playback.safetensors"),
                "tensors": tensor_inventory(playback),
            },
            "controls": {"file": "controls.jsonl", "sha256": sha256(output / "controls.jsonl")},
            "events": {"file": "events.jsonl", "sha256": sha256(output / "events.jsonl")},
            "plans": plan_artifacts,
        },
    }
    with (output / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--artifact-limit-mb", type=int, default=ARTIFACT_LIMIT_BYTES // (1024 * 1024))
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        preflight(args.upstream_root.resolve())
        return
    if args.output is None:
        parser.error("--output is required unless --preflight-only is used")
    capture(args.upstream_root.resolve(), args.output.resolve(), args.seed,
            args.artifact_limit_mb * 1024 * 1024)


if __name__ == "__main__":
    main()
