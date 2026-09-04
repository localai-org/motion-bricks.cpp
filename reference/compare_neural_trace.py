#!/usr/bin/env python3
"""Compare one observational upstream plan trace with native semantic boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open


TOLERANCES = {
    "input": 2e-5,
    "root": 5e-4,
    "pose": 5e-4,
    "decoder": 5e-4,
}


def metric(expected: np.ndarray, actual: np.ndarray, tolerance: float) -> dict[str, object]:
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    if expected.shape != actual.shape:
        return {
            "pass": False,
            "expected_shape": list(expected.shape),
            "actual_shape": list(actual.shape),
            "reason": "shape_mismatch",
        }
    if expected.dtype == np.bool_ or actual.dtype == np.bool_ or (
        np.issubdtype(expected.dtype, np.integer) and np.issubdtype(actual.dtype, np.integer)
    ):
        mismatches = int(np.count_nonzero(expected != actual))
        return {"pass": mismatches == 0, "mismatches": mismatches, "count": int(expected.size)}
    expected64 = expected.astype(np.float64, copy=False)
    actual64 = actual.astype(np.float64, copy=False)
    finite = bool(np.isfinite(expected64).all() and np.isfinite(actual64).all())
    difference = actual64 - expected64
    maximum = float(np.max(np.abs(difference), initial=0.0))
    denominator = max(float(np.linalg.norm(expected64)), float(np.linalg.norm(actual64)), 1e-30)
    relative = float(np.linalg.norm(difference)) / denominator
    return {
        "pass": finite and maximum <= tolerance,
        "finite": finite,
        "max_abs": maximum,
        "relative_l2": relative,
        "tolerance": tolerance,
        "count": int(expected.size),
    }


def load_upstream(path: Path) -> dict[str, np.ndarray]:
    with safe_open(path, framework="np") as handle:
        return {
            name.removeprefix("neural."): handle.get_tensor(name)
            for name in handle.keys()
            if name.startswith("neural.")
        }


def load_native(path: Path) -> tuple[int, int, dict[str, np.ndarray]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("format") != "motionbricks-native-neural-trace-v1":
        raise ValueError("unsupported native neural trace")
    tensors = {name: np.asarray(values) for name, values in document["tensors"].items()}
    return int(document["plan"]), int(document["selected_tokens"]), tensors


def compare(upstream_path: Path, native_path: Path) -> dict[str, object]:
    upstream = load_upstream(upstream_path)
    plan, tokens, native = load_native(native_path)
    frames = tokens * 4
    boundaries: list[dict[str, object]] = []

    def boundary(name: str, comparisons: list[tuple[str, np.ndarray, np.ndarray]], tolerance: float):
        metrics = {key: metric(expected, actual, tolerance) for key, expected, actual in comparisons}
        boundaries.append({"name": name, "pass": all(value["pass"] for value in metrics.values()),
                           "metrics": metrics})

    boundary("raw_sparse_input", [
        ("global_root_values", upstream["input.global_root_values"].reshape(8, 5),
         native["input.global_root_values"].reshape(8, 5)),
        ("local_root_values", upstream["input.local_root_values"].reshape(8, 4),
         native["input.local_root_values"].reshape(8, 4)),
        ("local_poses", upstream["input.local_poses"].reshape(8, 303),
         native["input.local_poses"].reshape(8, 303)),
        ("has_global_root_values", upstream["input.has_global_root_values"].reshape(8),
         native["input.has_global_root_values"].reshape(8)),
        ("has_local_root_values", upstream["input.has_local_root_values"].reshape(8),
         native["input.has_local_root_values"].reshape(8)),
        ("has_local_poses", upstream["input.has_local_poses"].reshape(8),
         native["input.has_local_poses"].reshape(8)),
        ("allowed_pred_num_tokens", upstream["input.allowed_pred_num_tokens"].reshape(11),
         native["input.allowed_pred_num_tokens"].reshape(11)),
    ], TOLERANCES["input"])

    boundary("normalized_root_input", [
        ("global_root_values", upstream["root.global_root_values"].reshape(8, 5),
         native["root.global_root_values"].reshape(8, 5)),
        ("local_root_values", upstream["root.local_root_values"].reshape(8, 4),
         native["root.local_root_values"].reshape(8, 4)),
        ("poses", upstream["root.poses"].reshape(8, 304), native["root.poses"].reshape(8, 304)),
    ], TOLERANCES["input"])

    allowed = upstream["input.allowed_pred_num_tokens"].reshape(11).astype(bool)
    upstream_logits = upstream["root.num_token_logits"].reshape(12)
    eligible_logits = upstream_logits[:11][allowed]
    native_logits = native["root.num_token_logits"].reshape(12)[:11][allowed]
    root_metrics = [
        ("eligible_duration_logits", eligible_logits, native_logits),
        ("pred_global_root_values", upstream["root.pred_global_root_values"].reshape(64, 5)[:frames],
         native["root.pred_global_root_values"].reshape(frames, 5)),
    ]
    boundary("root_output", root_metrics, TOLERANCES["root"])
    expected_duration = int(np.argmax(upstream_logits)) + 6
    duration_exact = expected_duration == tokens
    boundaries[-1]["metrics"]["selected_tokens"] = {
        "pass": duration_exact, "expected": expected_duration, "actual": tokens,
    }
    boundaries[-1]["pass"] = bool(boundaries[-1]["pass"] and duration_exact)

    boundary("pose_input", [
        ("root_condition", upstream["pose.root_condition"].reshape(64, 4)[:frames],
         native["pose.root_condition"].reshape(frames, 4)),
        ("pose_condition", upstream["pose.pose_condition"].reshape(64, 304)[:frames],
         native["pose.pose_condition"].reshape(frames, 304)),
        ("has_pose_condition", upstream["pose.has_pose_condition"].reshape(64)[:frames],
         native["pose.has_pose_condition"].reshape(frames)),
    ], TOLERANCES["root"])

    expected_pose_logits = upstream["pose.logits"].reshape(16, 8, 10)[:tokens]
    actual_pose_logits = native["pose.logits"].reshape(tokens, 8, 10)
    expected_pose_tokens = np.argmax(expected_pose_logits, axis=-1)
    actual_pose_tokens = native["pose.tokens"].reshape(tokens, 8)
    boundary("pose_output", [
        ("logits", expected_pose_logits, actual_pose_logits),
        ("selected_tokens", expected_pose_tokens, actual_pose_tokens),
    ], TOLERANCES["pose"])

    boundary("decoder_input", [
        ("quantized", upstream["decoder.quantized"].reshape(256, 16)[:, :tokens],
         native["decoder.quantized"].reshape(256, tokens)),
        ("external_condition", upstream["decoder.external_condition"].reshape(64, 2)[:frames],
         native["decoder.external_condition"].reshape(frames, 2)),
        ("target_condition", upstream["decoder.target_condition"].reshape(64, 304)[:frames],
         native["pose.pose_condition"].reshape(frames, 304)),
        ("has_target_condition", upstream["decoder.has_target_condition"].reshape(64)[:frames],
         native["decoder.has_target_condition"].reshape(frames)),
    ], TOLERANCES["pose"])

    boundary("decoder_output", [
        ("normalized_local_motion", upstream["decoder.output"].reshape(413, 64)[:, :frames].T,
         native["decoder.output"].reshape(frames, 413)),
    ], TOLERANCES["decoder"])

    first_divergence = next((item["name"] for item in boundaries if not item["pass"]), None)
    return {
        "format": "motionbricks-neural-boundary-comparison-v1",
        "plan": plan,
        "tokens": tokens,
        "frames": frames,
        "pass": first_divergence is None,
        "first_divergence": first_divergence,
        "boundaries": boundaries,
    }


def compare_directories(upstream: Path, native: Path) -> dict[str, object]:
    plans = []
    for upstream_path in sorted(upstream.glob("plan-*.safetensors")):
        plan = int(upstream_path.stem.removeprefix("plan-"))
        with safe_open(upstream_path, framework="np") as handle:
            if not any(name.startswith("neural.") for name in handle.keys()):
                continue
        native_path = native / f"native-plan-{plan:03d}.json"
        if not native_path.is_file():
            raise FileNotFoundError(f"missing native trace for plan {plan}: {native_path}")
        plans.append(compare(upstream_path, native_path))
    if not plans:
        raise ValueError("upstream directory contains no neural plan traces")
    divergences: dict[str, int] = {}
    for plan in plans:
        name = str(plan["first_divergence"] or "none")
        divergences[name] = divergences.get(name, 0) + 1
    return {
        "format": "motionbricks-neural-boundary-comparison-set-v1",
        "pass": all(plan["pass"] for plan in plans),
        "plan_count": len(plans),
        "first_divergence_counts": divergences,
        "plans": plans,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=Path)
    parser.add_argument("native", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.upstream.is_dir() != args.native.is_dir():
        parser.error("upstream and native must both be files or both be directories")
    result = (compare_directories(args.upstream, args.native)
              if args.upstream.is_dir() else compare(args.upstream, args.native))
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
