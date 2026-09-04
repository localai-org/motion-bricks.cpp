#!/usr/bin/env python3
"""Compare repeated black-box MotionBricks session captures."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from session_common import CAPTURE_FORMAT, COMPARISON_FORMAT, sha256


CONTRACT_PATHS = (
    ("harness",),
    ("upstream", "revision"),
    ("upstream", "trusted_files"),
    ("scenario",),
    ("seeds",),
    ("demo_flags",),
    ("environment", "python"),
    ("environment", "packages"),
    ("environment", "os_packages"),
    ("environment", "container_base"),
    ("environment", "cuda_runtime"),
    ("environment", "nvidia_driver"),
    ("environment", "cudnn"),
    ("environment", "gpu_name"),
    ("environment", "gpu_compute_capability"),
    ("environment", "deterministic_algorithms"),
    ("environment", "cudnn_deterministic"),
    ("environment", "cudnn_benchmark"),
    ("environment", "cuda_matmul_allow_tf32"),
    ("environment", "cudnn_allow_tf32"),
    ("environment", "float32_matmul_precision"),
    ("environment", "cublas_workspace_config"),
)


def nested(value: dict[str, Any], path: tuple[str, ...]) -> Any:
    for key in path:
        value = value[key]
    return value


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def tensor_files(capture: Path, manifest: dict[str, Any]) -> list[str]:
    return [manifest["capture"]["playback"]["file"]] + [
        plan["file"] for plan in manifest["capture"]["plans"]
    ]


def compare_capture_set(
    captures: list[Path],
    max_abs_tolerance: float,
    relative_l2_tolerance: float,
    core_only: bool = False,
) -> dict[str, Any]:
    if len(captures) < 2:
        raise ValueError("at least two capture directories are required")

    import numpy as np
    from safetensors import safe_open

    def load_tensors(path: Path) -> dict[str, Any]:
        with safe_open(path, framework="np") as handle:
            return {key: handle.get_tensor(key) for key in handle.keys()}

    manifests = [read_json(path / "manifest.json") for path in captures]
    for path, manifest in zip(captures, manifests):
        if manifest.get("format") != CAPTURE_FORMAT:
            raise ValueError(f"unsupported capture format in {path}")

    failures: list[str] = []
    reference_manifest = manifests[0]
    core_exclusions = {("harness",), ("environment", "os_packages")}
    contract_paths = tuple(
        path for path in CONTRACT_PATHS if not core_only or path not in core_exclusions
    )
    for index, manifest in enumerate(manifests[1:], start=1):
        for path in contract_paths:
            if nested(reference_manifest, path) != nested(manifest, path):
                failures.append(f"run {index} differs at manifest contract {'.'.join(path)}")
        for filename in ("controls.jsonl", "events.jsonl"):
            if read_text(captures[0] / filename) != read_text(captures[index] / filename):
                failures.append(f"run {index} differs in {filename}")

    expected_files = tensor_files(captures[0], reference_manifest)
    for index, manifest in enumerate(manifests[1:], start=1):
        if tensor_files(captures[index], manifest) != expected_files:
            failures.append(f"run {index} has a different tensor-file inventory")

    tensor_metrics: dict[str, dict[str, Any]] = {}
    for filename in expected_files:
        opened = [load_tensors(path / filename) for path in captures]
        keys = list(opened[0])
        names_match = True
        for index, handle in enumerate(opened[1:], start=1):
            if core_only and not set(keys).issubset(handle):
                failures.append(f"run {index} is missing reference tensors for {filename}")
                names_match = False
            elif not core_only and list(handle) != keys:
                failures.append(f"run {index} differs in tensor names for {filename}")
                names_match = False
        if not names_match:
            continue
        for key in keys:
            tensors = [handle[key] for handle in opened]
            metric_name = f"{filename}:{key}"
            reference = tensors[0]
            if any(tuple(value.shape) != tuple(reference.shape) or value.dtype != reference.dtype
                   for value in tensors[1:]):
                failures.append(f"shape or dtype differs for {metric_name}")
                continue
            if not np.issubdtype(reference.dtype, np.floating):
                exact = all(
                    np.array_equal(left, right)
                    for left, right in itertools.combinations(tensors, 2)
                )
                tensor_metrics[metric_name] = {"dtype": str(reference.dtype), "exact": exact}
                if not exact:
                    failures.append(f"discrete tensor differs for {metric_name}")
                continue

            max_abs = 0.0
            max_relative_l2 = 0.0
            finite = all(bool(np.isfinite(value).all()) for value in tensors)
            for left, right in itertools.combinations(tensors, 2):
                left64 = left.astype(np.float64).reshape(-1)
                right64 = right.astype(np.float64).reshape(-1)
                difference = right64 - left64
                if difference.size:
                    max_abs = max(max_abs, float(np.abs(difference).max()))
                    denominator = max(float(np.linalg.norm(left64)), float(np.linalg.norm(right64)))
                    relative = float(np.linalg.norm(difference)) / max(denominator, 1e-30)
                    max_relative_l2 = max(max_relative_l2, relative)
            tensor_metrics[metric_name] = {
                "dtype": str(reference.dtype),
                "finite": finite,
                "max_abs": max_abs,
                "max_relative_l2": max_relative_l2,
            }
            if not finite:
                failures.append(f"non-finite float tensor in {metric_name}")
            if not math.isfinite(max_abs) or max_abs > max_abs_tolerance:
                failures.append(
                    f"{metric_name} max_abs {max_abs:.9g} exceeds {max_abs_tolerance:.9g}"
                )
            if not math.isfinite(max_relative_l2) or max_relative_l2 > relative_l2_tolerance:
                failures.append(
                    f"{metric_name} relative_l2 {max_relative_l2:.9g} exceeds "
                    f"{relative_l2_tolerance:.9g}"
                )

    return {
        "format": COMPARISON_FORMAT,
        "scope": "core_outputs" if core_only else "full_capture",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "captures": [path.name for path in captures],
        "capture_manifests": [
            {"directory": path.name, "sha256": sha256(path / "manifest.json")}
            for path in captures
        ],
        "harness": {
            name: sha256(Path(__file__).resolve().parent / name)
            for name in ("compare_session_captures.py", "run_session_baseline.py")
        },
        "thresholds": {
            "max_abs": max_abs_tolerance,
            "max_relative_l2": relative_l2_tolerance,
        },
        "pass": not failures,
        "failures": failures,
        "tensors": tensor_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="+", type=Path)
    parser.add_argument("--max-abs", type=float, default=1e-5)
    parser.add_argument("--max-relative-l2", type=float, default=1e-6)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="compare reference tensors while allowing additional trace tensors and harness identity",
    )
    args = parser.parse_args()
    result = compare_capture_set(
        [path.resolve() for path in args.captures], args.max_abs, args.max_relative_l2,
        args.core_only,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
