#!/usr/bin/env python3
"""Replay captured neural inputs through upstream MotionBricks on CPU."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from capture_session import make_demo_args, validate_upstream
from session_common import CAPTURE_FORMAT, UPSTREAM_REVISION, sha256


def load_neural(path: Path) -> dict[str, torch.Tensor]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return {
            name.removeprefix("neural."): handle.get_tensor(name)
            for name in handle.keys()
            if name.startswith("neural.")
        }


def replay(upstream: Path, capture: Path, output: Path) -> None:
    if os.environ.get("MOTIONBRICKS_REFERENCE_CONTAINER") != "1":
        raise RuntimeError("inference replay must run in the pinned reference session container")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite replay directory: {output}")
    validate_upstream(upstream)
    manifest_path = capture / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("format") != CAPTURE_FORMAT or
            manifest.get("upstream", {}).get("revision") != UPSTREAM_REVISION):
        raise ValueError("capture does not match the pinned upstream session format")
    if not manifest.get("trace", {}).get("neural_all"):
        raise ValueError("capture does not contain all-plan neural traces")

    sys.path.insert(0, str(upstream / "motionbricks"))
    os.environ.setdefault("PYNPUT_BACKEND", "dummy")
    from motionbricks.motion_backbone.demo.utils import navigation_demo

    os.chdir(upstream / "motionbricks")
    demo = navigation_demo(make_demo_args(upstream))
    inferencer = demo.full_agent._inferencer
    inferencer._root_model.to("cpu")
    inferencer._pose_model.to("cpu")
    inferencer._vqvae_pose_model.to("cpu")
    inferencer._device = torch.device("cpu")

    output.mkdir(parents=True)
    records = []
    for plan_index in range(int(manifest["capture"]["plan_count"])):
        source_path = capture / f"plan-{plan_index:03d}.safetensors"
        values = load_neural(source_path)
        required = (
            "input.global_root_values", "input.has_global_root_values",
            "input.local_root_values", "input.has_local_root_values",
            "input.local_poses", "input.has_local_poses", "input.num_tokens",
            "input.allowed_pred_num_tokens", "composition.pred_global_motions",
            "composition.pred_num_tokens",
        )
        missing = [name for name in required if name not in values]
        if missing:
            raise ValueError(f"plan {plan_index} is missing neural tensors: {missing}")
        with torch.inference_mode():
            motions, tokens = inferencer.predict(
                values["input.global_root_values"], values["input.has_global_root_values"],
                values["input.local_root_values"], values["input.has_local_root_values"],
                values["input.local_poses"], values["input.has_local_poses"],
                values["input.num_tokens"],
                config={
                    "num_inference_step": 1,
                    "smooth_root_traj": False,
                    "allow_pred_out_of_reach_num_tokens": False,
                    "pose_token_sampling_use_argmax": True,
                    "skip_ending_target_cond": False,
                },
                info={},
                allowed_pred_num_tokens=values["input.allowed_pred_num_tokens"],
            )
        valid_tokens = int(tokens.item())
        valid_frames = valid_tokens * 4
        cpu = motions[:, :valid_frames].detach().cpu().contiguous()
        cuda = values["composition.pred_global_motions"][:, :valid_frames]
        difference = cpu - cuda
        path = output / f"plan-{plan_index:03d}.safetensors"
        save_file({"model_features": cpu, "pred_num_tokens": tokens.detach().cpu()}, path)
        records.append({
            "plan": plan_index,
            "frames": valid_frames,
            "duration_matches_cuda": valid_tokens == int(values["composition.pred_num_tokens"].item()),
            "cuda_max_abs": float(difference.abs().max()),
            "cuda_relative_l2": float(
                torch.linalg.vector_norm(difference)
                / torch.maximum(torch.linalg.vector_norm(cpu), torch.linalg.vector_norm(cuda))
            ),
            "file": path.name,
            "sha256": sha256(path),
        })

    replay_manifest = {
        "format": "motionbricks-upstream-cpu-inference-replay-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "upstream_revision": UPSTREAM_REVISION,
        "source_capture_manifest_sha256": sha256(manifest_path),
        "device": "cpu",
        "torch": torch.__version__,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "plans": records,
    }
    (output / "manifest.json").write_text(
        json.dumps(replay_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay(args.upstream_root.resolve(), args.capture.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
