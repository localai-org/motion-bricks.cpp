#!/usr/bin/env python3
"""Replay diagnostic root/decoder inputs through upstream; endpoint ablations only.

This does not change native inference or assert complete handoff parity. The
decoder ablation deliberately holds tokens and root conditioning fixed.
Run inside the existing trusted reference container, using safe intermediates.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from generate_fixtures import SAFE_HASHES, UPSTREAM_REVISION, make_motion_rep, state, verify


ROOT_ARGS = {
    "activation": "relu", "depth": 4, "dilation_growth_rate": 3, "down_t": 2,
    "global_root_feat_dim": 64, "global_root_feature": "root",
    "input_feat_mlp_num_layers": 2,
    "local_pose_feature": "joint_positions_and_rotations_and_hip_height",
    "local_root_feat_dim": 64, "local_root_feature": "root",
    "max_tokens": 16, "min_tokens": 6, "n_embd": 512, "n_head": 16,
    "n_layers_root_token": 3, "n_layers_shared": 3, "norm": "None",
    "pose_feat_dim": 256,
    "pose_vqvae": {"code_dim": 60, "nb_code": 1024, "num_heads": 1},
    "root_vqvae": {"code_dim": 60, "nb_code": 1024, "num_heads": 1},
    "text_emb_dim": 4096, "text_embeddings": None,
    "use_hard_num_token_emb_for_root_prediction": True, "width": 512,
}


def summary(value):
    return {"min": float(value.min()), "max": float(value.max())}


def comparison(actual, expected):
    difference = actual - expected
    return {**summary(actual), "max_abs_error": float(difference.abs().max()),
            "relative_l2": float(difference.norm() / expected.norm().clamp_min(1e-20))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--safe-directory", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("MOTIONBRICKS_REFERENCE_CONTAINER") != "1":
        raise RuntimeError("run in the trusted reference container")
    revision = subprocess.check_output(
        ["git", "-C", str(args.upstream_root), "rev-parse", "HEAD"], text=True).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError("unexpected upstream revision")
    subprocess.run(["git", "-C", str(args.upstream_root), "diff", "--exit-code", "HEAD",
                    "--", "motionbricks/motionbricks"], check=True)
    for component in ("root", "vqvae"):
        verify(args.safe_directory / f"{component}.safetensors", SAFE_HASHES[component])
    sys.path.insert(0, str(args.upstream_root / "motionbricks"))
    from motionbricks.motion_backbone.neural_modules.root_backbone import root_backbone_network
    from motionbricks.vqvae.neural_modules.encdec_double_cond import DoubleCondDecoder

    torch.set_num_threads(4)
    document = json.loads(args.trace.read_text())
    if document["format"] != "motionbricks-transition-debug-v1":
        raise ValueError("unexpected trace format")
    data = document["tensors"]
    count = document["selected_tokens"]
    frames = count * 4

    def tensor(key, shape, dtype=torch.float32):
        return torch.tensor(data[key], dtype=dtype).reshape(shape)

    mean = tensor("stats.mean", (-1,))
    std = tensor("stats.std", (-1,))
    denom = (std.square() + 1e-5).sqrt()

    def endpoints_identity(poses):
        result = poses.clone()
        for joint in (7, 14, 25, 33):
            feature = 100 + joint * 6
            identity = torch.tensor([1., 0., 0., 0., 1., 0.])
            result[:, :4, feature:feature + 6] = (
                identity - mean[feature + 8:feature + 14]) / denom[feature + 8:feature + 14]
        return result

    poses = tensor("root.poses", (1, 8, 304))
    corrected = endpoints_identity(poses)
    report = {"format": "motionbricks-transition-replay-v1", "upstream_revision": revision,
              "torch": torch.__version__, "selected_tokens_fixed": count,
              "source_normalized": summary(poses[:, :4]),
              "source_endpoint_identity": summary(corrected[:, :4]),
              "stage_ms": data["stage_ms.root_pose_vq_decode"]}
    root = root_backbone_network(ROOT_ARGS, make_motion_rep()).eval()
    root.load_state_dict(state(args.safe_directory / "root.safetensors", "backbone_net."), strict=True)
    with torch.inference_mode():
        for label, condition in (("root_replay", poses), ("root_endpoint_identity", corrected)):
            result = root(
                tensor("root.global_root_values", (1, 8, 5)),
                tensor("input.has_global_root_values", (1, 8), torch.bool),
                tensor("root.local_root_values", (1, 8, 4)),
                tensor("input.has_local_root_values", (1, 8), torch.bool), condition,
                tensor("input.has_local_poses", (1, 8), torch.bool),
                torch.tensor([[count]], dtype=torch.int64))["pred_global_root_values"][:, :frames]
            report[label] = comparison(result, tensor("root.pred_global_root_values", (1, frames, 5)))
    del root
    decoder = DoubleCondDecoder(
        input_emb_width=413, output_emb_width=256, down_t=2, width=512, depth=4,
        dilation_growth_rate=3, activation="relu", norm="None",
        target_cond_dim=304, external_cond_dim=2, cond_fusion_last_layer=False).eval()
    decoder.load_state_dict(state(args.safe_directory / "vqvae.safetensors", "pose_net.decoder."), strict=True)
    condition = tensor("pose.pose_condition", (1, frames, 304))
    with torch.inference_mode():
        for label, target in (("decoder_replay", condition),
                              ("decoder_endpoint_identity_fixed_other_inputs", endpoints_identity(condition))):
            result = decoder(
                tensor("decoder.quantized", (1, 256, count)),
                external_cond=tensor("decoder.external_condition", (1, frames, 2)),
                target_cond=target,
                has_target_cond=tensor("decoder.has_target_condition", (1, frames), torch.bool),
                token_mask=torch.ones((1, count), dtype=torch.bool)).transpose(1, 2)
            report[label] = comparison(result, tensor("decoder.output", (1, frames, 413)))
            heights = result[0, :, 3] * denom[8] + mean[8]
            report[label]["root_height"] = {**summary(heights), "first": float(heights[0])}
    rendered = json.dumps(report, indent=2) + "\n"
    args.output.write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
