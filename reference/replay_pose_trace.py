#!/usr/bin/env python3
"""Replay an observed pose-model boundary with pinned upstream PyTorch on CPU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

from generate_fixtures import SAFE_HASHES, make_motion_rep, state, verify


POSE_ARGS = {
    "cond_root_feature": "root_without_hip_height",
    "cond_root_feature_is_from_motion_rep": "global",
    "down_t": 2,
    "local_pose_feature": "joint_positions_and_rotations_and_hip_height",
    "max_tokens": 16,
    "min_tokens": 6,
    "n_embd": 1024,
    "n_head": 16,
    "n_layers": 16,
    "pose_feat_width": 640,
    "pose_root_mode": "pose",
    "pose_token_mlp_num_layers": 2,
    "pose_vqvae": {
        "code_dim": 256, "has_codebook": True, "nb_code": 100_000_000, "num_heads": 8,
    },
    "root_feat_width": 256,
    "root_vqvae": {"code_dim": 60, "nb_code": 1024, "num_heads": 1},
    "text_emb_dim": 4096,
    "text_embeddings": None,
    "token_length_feat_width": 128,
}


def tensor(handle, name: str) -> torch.Tensor:
    return torch.from_numpy(handle.get_tensor(f"neural.{name}").copy())


def replay(model: torch.nn.Module, path: Path) -> dict[str, object]:
    with safe_open(path, framework="np") as observed:
        tokens = tensor(observed, "pose.input_tokens").to(torch.int64)
        root = tensor(observed, "pose.root_condition")
        condition = tensor(observed, "pose.pose_condition")
        mask = tensor(observed, "pose.has_pose_condition").to(torch.bool)
        duration = tensor(observed, "pose.num_tokens").to(torch.int64)
        expected = tensor(observed, "pose.logits")
    with torch.inference_mode():
        actual = model(tokens, root, condition, mask, duration)["pose_logits"]

    valid = int(duration.item())
    difference = actual[:, :valid] - expected[:, :valid]
    expected_tokens = expected[:, :valid].argmax(dim=-1)
    actual_tokens = actual[:, :valid].argmax(dim=-1)
    return {
        "plan": int(path.stem.removeprefix("plan-")),
        "valid_tokens": valid,
        "max_abs": float(difference.abs().max()),
        "relative_l2": float(
            torch.linalg.vector_norm(difference)
            / torch.maximum(
                torch.linalg.vector_norm(actual[:, :valid]),
                torch.linalg.vector_norm(expected[:, :valid]),
            )
        ),
        "token_mismatches": int((actual_tokens != expected_tokens).sum()),
        "token_count": int(actual_tokens.numel()),
        "cuda_tokens": expected_tokens.reshape(-1).tolist(),
        "cpu_tokens": actual_tokens.reshape(-1).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--safe-directory", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    safe_pose = args.safe_directory / "pose.safetensors"
    verify(safe_pose, SAFE_HASHES["pose"])
    sys.path.insert(0, str(args.upstream_root / "motionbricks"))
    from motionbricks.motion_backbone.neural_modules.pose_backbone import pose_backbone_network

    model = pose_backbone_network(make_motion_rep(), POSE_ARGS).eval()
    missing, unexpected = model.load_state_dict(
        state(safe_pose, "backbone_net."), strict=True
    )
    if missing or unexpected:
        raise RuntimeError(f"pose state mismatch: missing={missing}, unexpected={unexpected}")

    if args.trace.is_dir():
        plans = [replay(model, path) for path in sorted(args.trace.glob("plan-*.safetensors"))]
        report = {
            "format": "motionbricks-pose-cpu-replay-set-v1",
            "plan_count": len(plans),
            "token_mismatches": sum(int(plan["token_mismatches"]) for plan in plans),
            "token_count": sum(int(plan["token_count"]) for plan in plans),
            "plans": plans,
        }
    else:
        report = {"format": "motionbricks-pose-cpu-replay-v1", **replay(model, args.trace)}
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
