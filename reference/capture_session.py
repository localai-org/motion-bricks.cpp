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


def clone_device(value, torch):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"neural trace expected a tensor, got {type(value).__name__}")
    return value.detach().contiguous().clone()


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


def capture(
    upstream: Path,
    output: Path,
    requested_seed: int,
    artifact_limit: int,
    trace_targets: bool = False,
    trace_neural_plan: int | None = None,
    trace_neural_all: bool = False,
) -> None:
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

    target_traces: list[dict[str, Any]] = []
    neural_traces: dict[int, dict[str, Any]] = {}
    trace_frame = -1
    if trace_targets:
        original_target_transform = demo.full_agent._generate_target_joint_transforms

        def traced_target_transform(inputs):
            result = original_target_transform(inputs)
            target_traces.append({
                "frame": trace_frame,
                "spring_target_root_position": clone_cpu(inputs["target_root_position"], torch),
                "spring_target_root_positions": clone_cpu(inputs["target_root_positions"], torch),
                "spring_target_root_headings": clone_cpu(inputs["target_root_headings"], torch),
                "spring_target_heading": clone_cpu(inputs["target_root_heading"], torch),
                "spring_start_root_positions": clone_cpu(inputs["start_root_positions"], torch),
                "spring_start_root_headings": clone_cpu(inputs["start_root_headings"], torch),
                "canonical_first_frame_position": clone_cpu(inputs["first_frame_position"], torch),
                "canonical_first_frame_heading_angle": clone_cpu(inputs["first_frame_heading_angle"], torch),
                "target_global_joint_positions": clone_cpu(result[0], torch),
                "target_global_joint_rotations": clone_cpu(result[1], torch),
                "target_global_root_positions": clone_cpu(result[2], torch),
            })
            return result

        # This is an instance-local external wrapper. The upstream checkout is
        # mounted read-only and the returned/input tensors are copied only after
        # the original boundary has completed.
        demo.full_agent._generate_target_joint_transforms = traced_target_transform

    trace_neural_enabled = trace_neural_plan is not None or trace_neural_all
    if trace_neural_enabled:
        if trace_neural_plan is not None and trace_neural_plan < 0:
            raise ValueError("--trace-neural-plan must be non-negative")
        inferencer = demo.full_agent._inferencer
        trace_state: dict[str, Any] = {"call": 0, "active": False, "device": {}}

        def save_device(name, value):
            if trace_state["active"]:
                trace_state["device"][name] = clone_device(value, torch)

        def root_hook(_module, inputs, output):
            names = (
                "global_root_values", "has_global_root_values",
                "local_root_values", "has_local_root_values",
                "poses", "has_poses", "num_tokens",
            )
            for name, value in zip(names, inputs):
                save_device(f"root.{name}", value)
            for name in ("num_token_logits", "pred_num_tokens", "pred_global_root_values"):
                save_device(f"root.{name}", output[name])

        def pose_hook(_module, inputs, output):
            names = ("input_tokens", "root_condition", "pose_condition", "has_pose_condition", "num_tokens")
            for name, value in zip(names, inputs):
                save_device(f"pose.{name}", value)
            save_device("pose.logits", output["pose_logits"])

        def decoder_pre_hook(_module, inputs, kwargs):
            names = ("quantized", "external_condition", "target_condition", "has_target_condition")
            for name, value in zip(names, inputs):
                if value is not None:
                    save_device(f"decoder.{name}", value)
            if kwargs.get("token_mask") is not None:
                save_device("decoder.token_mask", kwargs["token_mask"])

        def decoder_hook(_module, _inputs, _kwargs, output):
            save_device("decoder.output", output)

        hook_handles = [
            inferencer._root_model.backbone_net.register_forward_hook(root_hook),
            inferencer._pose_model.backbone_net.register_forward_hook(pose_hook),
            inferencer._vqvae_pose_model.decoder.register_forward_pre_hook(
                decoder_pre_hook, with_kwargs=True
            ),
            inferencer._vqvae_pose_model.decoder.register_forward_hook(
                decoder_hook, with_kwargs=True
            ),
        ]
        original_predict = inferencer.predict

        def traced_predict(*args, **kwargs):
            plan_index = trace_state["call"]
            trace_state["call"] += 1
            trace_state["active"] = trace_neural_all or plan_index == trace_neural_plan
            trace_state["device"] = {}
            try:
                result = original_predict(*args, **kwargs)
                if trace_state["active"]:
                    raw_names = (
                        "input.global_root_values", "input.has_global_root_values",
                        "input.local_root_values", "input.has_local_root_values",
                        "input.local_poses", "input.has_local_poses", "input.num_tokens",
                    )
                    for name, value in zip(raw_names, args):
                        save_device(name, value)
                    if kwargs.get("allowed_pred_num_tokens") is not None:
                        save_device("input.allowed_pred_num_tokens", kwargs["allowed_pred_num_tokens"])
                    save_device("composition.pred_global_motions", result[0])
                    save_device("composition.pred_num_tokens", result[1])
                    neural_traces[plan_index] = {
                        f"neural.{name}": clone_cpu(value, torch)
                        for name, value in trace_state["device"].items()
                    }
                return result
            finally:
                trace_state["active"] = False
                trace_state["device"] = {}

        inferencer.predict = traced_predict

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
        trace_frame = frame
        traces_before = len(target_traces)
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
            if next_plan in neural_traces:
                plans[next_plan].update(neural_traces.pop(next_plan))
            if trace_targets:
                if len(target_traces) != traces_before + 1:
                    raise RuntimeError("replan did not expose exactly one target-transform boundary")
                trace = target_traces[-1]
                if trace.pop("frame") != frame:
                    raise RuntimeError("target-transform trace was associated with the wrong frame")
                plans[next_plan].update(trace)
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
        elif len(target_traces) != traces_before:
            raise RuntimeError("target-transform boundary ran without a public replan")
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
    if trace_neural_enabled:
        requested_neural_plans = set(plans) if trace_neural_all else {trace_neural_plan}
        if not requested_neural_plans.issubset(plans):
            raise RuntimeError("one or more requested neural trace plans were not generated")
        if neural_traces:
            raise RuntimeError("a neural trace was not attached to its planning artifact")
        for plan_index in requested_neural_plans:
            if not any(name.startswith("neural.") for name in plans[plan_index]):
                raise RuntimeError(f"plan {plan_index} has no neural trace tensors")
        for handle in hook_handles:
            handle.remove()
    for event in events:
        valid_length = event["valid_length"]
        if valid_length < 24 or valid_length > 64 or valid_length % 4:
            raise RuntimeError(f"invalid generated frame count in plan {event['plan']}: {valid_length}")
        allowed = allowed_tokens[event["command_after_output_frame"]].reshape(-1)
        token_index = valid_length // 4 - 6
        if token_index >= allowed.numel() or int(allowed[token_index].item()) != 1:
            raise RuntimeError(f"plan {event['plan']} selected a disallowed duration")
        tensors = plans[event["plan"]]
        if not {"qpos", "model_features"}.issubset(tensors):
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
        "trace": {
            "target_boundaries": trace_targets,
            "neural_plan": trace_neural_plan,
            "neural_all": trace_neural_all,
            "mechanism": {
                "targets": "external_instance_post_return_wrapper" if trace_targets else "disabled",
                "neural": "external_instance_wrappers_and_forward_hooks" if trace_neural_enabled else "disabled",
            },
            "upstream_source_modified": False,
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
    parser.add_argument("--trace-targets", action="store_true")
    neural_group = parser.add_mutually_exclusive_group()
    neural_group.add_argument("--trace-neural-plan", type=int)
    neural_group.add_argument("--trace-neural-all", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        preflight(args.upstream_root.resolve())
        return
    if args.output is None:
        parser.error("--output is required unless --preflight-only is used")
    capture(args.upstream_root.resolve(), args.output.resolve(), args.seed,
            args.artifact_limit_mb * 1024 * 1024, args.trace_targets,
            args.trace_neural_plan, args.trace_neural_all)


if __name__ == "__main__":
    main()
