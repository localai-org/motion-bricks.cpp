#!/usr/bin/env python3
"""Run repeated fresh-process captures and compare the resulting baseline."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from session_common import ARTIFACT_LIMIT_BYTES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--artifact-limit-mb", type=int, default=ARTIFACT_LIMIT_BYTES // (1024 * 1024))
    parser.add_argument("--max-abs", type=float, default=1e-5)
    parser.add_argument("--max-relative-l2", type=float, default=1e-6)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    if os.environ.get("MOTIONBRICKS_REFERENCE_CONTAINER") != "1":
        raise RuntimeError("baseline capture must run in the pinned reference session container")
    script_dir = Path(__file__).resolve().parent
    if args.preflight_only:
        subprocess.run([
            sys.executable,
            str(script_dir / "capture_session.py"),
            "--upstream-root", str(args.upstream_root.resolve()),
            "--preflight-only",
        ], check=True)
        return
    if args.output is None:
        parser.error("--output is required unless --preflight-only is used")
    if args.repeats < 3:
        raise ValueError("the accepted baseline requires at least three fresh-process captures")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing baseline directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.partial-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"staging directory already exists: {staging}")
    staging.mkdir()

    captures = []
    try:
        for index in range(args.repeats):
            destination = staging / f"run-{index:03d}"
            subprocess.run(
                [
                    sys.executable,
                    str(script_dir / "capture_session.py"),
                    "--upstream-root", str(args.upstream_root.resolve()),
                    "--output", str(destination),
                    "--seed", str(args.seed),
                    "--artifact-limit-mb", str(args.artifact_limit_mb),
                ],
                check=True,
            )
            captures.append(destination)

        subprocess.run(
            [
                sys.executable,
                str(script_dir / "compare_session_captures.py"),
                *[str(path) for path in captures],
                "--max-abs", str(args.max_abs),
                "--max-relative-l2", str(args.max_relative_l2),
                "--output", str(staging / "comparison.json"),
            ],
            check=True,
        )
        staging.rename(output)
    except Exception:
        print(f"incomplete baseline retained at {staging}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
