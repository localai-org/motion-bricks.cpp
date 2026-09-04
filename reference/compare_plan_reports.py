#!/usr/bin/env python3
"""Compare CPU/Vulkan native outputs from two open-loop parity reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


TOLERANCES = {
    "root_m": 1.0e-3,
    "rotation_deg": 0.2,
    "joint_m": 2.0e-3,
    "target_root_m": 1.0e-4,
    "target_rotation_deg": 0.05,
    "target_joint_m": 2.0e-4,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vector_metric(left: list[float], right: list[float], width: int = 3) -> dict[str, float]:
    if len(left) != len(right) or not left or len(left) % width:
        return {"max": math.inf, "rmse": math.inf}
    norms = [
        math.sqrt(sum((left[index + axis] - right[index + axis]) ** 2 for axis in range(width)))
        for index in range(0, len(left), width)
    ]
    return {"max": max(norms), "rmse": math.sqrt(sum(value * value for value in norms) / len(norms))}


def quaternion_metric(left: list[float], right: list[float]) -> dict[str, float]:
    if len(left) != len(right) or not left or len(left) % 4:
        return {"max": math.inf, "rmse": math.inf}
    angles = []
    for index in range(0, len(left), 4):
        a, b = left[index:index + 4], right[index:index + 4]
        denominator = math.sqrt(sum(value * value for value in a) * sum(value * value for value in b))
        if denominator == 0:
            return {"max": math.inf, "rmse": math.inf}
        cosine = min(1.0, max(0.0, abs(sum(x * y for x, y in zip(a, b, strict=True))) / denominator))
        angles.append(2.0 * math.acos(cosine) * 180.0 / math.pi)
    return {"max": max(angles), "rmse": math.sqrt(sum(value * value for value in angles) / len(angles))}


def compare_trace_directories(left: Path, right: Path) -> dict:
    names = [
        "input.global_root_values", "input.local_root_values", "input.local_poses",
        "root.num_token_logits", "root.pred_global_root_values", "root.pred_local_root_values",
        "pose.logits", "decoder.quantized", "decoder.output",
    ]
    files = sorted(left.glob("native-plan-*.json"))
    if not files:
        raise ValueError("left trace directory contains no native plan traces")
    maxima = {name: {"max_abs": 0.0, "plan": 0} for name in names}
    duration_mismatches = pose_mismatches = pose_total = 0
    for left_path in files:
        right_path = right / left_path.name
        if not right_path.is_file():
            raise ValueError(f"right trace directory lacks {left_path.name}")
        first = json.loads(left_path.read_text(encoding="utf-8"))
        second = json.loads(right_path.read_text(encoding="utf-8"))
        if (first.get("format") != "motionbricks-native-neural-trace-v1" or
                second.get("format") != first.get("format") or first.get("plan") != second.get("plan")):
            raise ValueError(f"incompatible neural traces: {left_path.name}")
        duration_mismatches += first["selected_tokens"] != second["selected_tokens"]
        left_tokens = first["tensors"]["pose.tokens"]
        right_tokens = second["tensors"]["pose.tokens"]
        if len(left_tokens) != len(right_tokens):
            raise ValueError(f"pose-token dimensions differ: {left_path.name}")
        pose_total += len(left_tokens)
        pose_mismatches += sum(a != b for a, b in zip(left_tokens, right_tokens, strict=True))
        for name in names:
            a, b = first["tensors"][name], second["tensors"][name]
            if len(a) != len(b):
                raise ValueError(f"neural tensor dimensions differ for {name}: {left_path.name}")
            maximum = max((abs(x - y) for x, y in zip(a, b, strict=True)), default=0.0)
            if maximum > maxima[name]["max_abs"]:
                maxima[name] = {"max_abs": maximum, "plan": first["plan"]}
    return {
        "plan_count": len(files),
        "duration_token_mismatches": duration_mismatches,
        "pose_token_mismatches": pose_mismatches,
        "pose_token_total": pose_total,
        "continuous_maxima": maxima,
        "acceptance_role": "diagnostic; public animation metrics define backend acceptance",
    }


def compare(left_path: Path, right_path: Path, output: Path,
            hardware: str | None = None, driver: str | None = None,
            left_traces: Path | None = None, right_traces: Path | None = None) -> dict:
    left = json.loads(left_path.read_text(encoding="utf-8"))
    right = json.loads(right_path.read_text(encoding="utf-8"))
    if (left.get("format") != "motionbricks-open-loop-report-v1" or
            right.get("format") != left.get("format") or
            left.get("fps") != right.get("fps") or left.get("joints") != right.get("joints") or
            len(left.get("plans", [])) != len(right.get("plans", []))):
        raise ValueError("reports have incompatible formats, dimensions, or plan counts")
    plans = []
    fields = {
        "root_m": ("native_roots", vector_metric),
        "rotation_deg": ("native_rotations", quaternion_metric),
        "joint_m": ("native_joint_positions", vector_metric),
        "target_root_m": ("native_target_roots", vector_metric),
        "target_rotation_deg": ("native_target_rotations", quaternion_metric),
        "target_joint_m": ("native_target_joint_positions", vector_metric),
    }
    for first, second in zip(left["plans"], right["plans"], strict=True):
        identity = ("index", "command_frame", "mode", "style", "seed", "expected_frames", "movement", "facing")
        if any(first.get(key) != second.get(key) for key in identity):
            raise ValueError(f"report plan identity differs at plan {first.get('index')}")
        metrics = {}
        for name, (field, function) in fields.items():
            if field not in first or field not in second:
                raise ValueError(f"reports lack required optional field {field}; regenerate them with the current runner")
            metrics[name] = function(first[field], second[field])
        duration_exact = first["actual_frames"] == second["actual_frames"]
        passed = duration_exact and all(metrics[name]["max"] <= tolerance
                                        for name, tolerance in TOLERANCES.items())
        plans.append({
            "index": first["index"],
            "duration_exact": duration_exact,
            "passed": passed,
            "metrics": metrics,
        })
    worst = {}
    for name in TOLERANCES:
        plan = max(plans, key=lambda item: item["metrics"][name]["max"])
        worst[name] = {"max": plan["metrics"][name]["max"], "plan": plan["index"]}
    result = {
        "format": "motionbricks-native-backend-open-loop-comparison-v1",
        "left_device": left["device"], "right_device": right["device"],
        "hardware": {"name": hardware or "unspecified", "driver": driver or "unspecified"},
        "sources": {"left_sha256": sha256(left_path), "right_sha256": sha256(right_path)},
        "tolerances": TOLERANCES,
        "passed": all(plan["passed"] for plan in plans),
        "plans": plans,
        "summary": {
            "duration_matches": sum(plan["duration_exact"] for plan in plans),
            "duration_total": len(plans),
            "worst": worst,
        },
    }
    if (left_traces is None) != (right_traces is None):
        raise ValueError("left and right trace directories must be provided together")
    if left_traces is not None and right_traces is not None:
        result["neural"] = compare_trace_directories(left_traces, right_traces)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hardware")
    parser.add_argument("--driver")
    parser.add_argument("--left-traces", type=Path)
    parser.add_argument("--right-traces", type=Path)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    result = compare(args.left, args.right, args.output, args.hardware, args.driver,
                     args.left_traces, args.right_traces)
    print(json.dumps({"passed": result["passed"], **result["summary"]}, indent=2, sort_keys=True))
    if not result["passed"] and not args.report_only:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
