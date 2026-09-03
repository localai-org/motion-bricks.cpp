#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from session_common import (  # noqa: E402
    CAPTURE_FORMAT,
    FRAME_COUNT,
    SEGMENTS,
    plan_seed,
    scenario_control,
    scenario_manifest,
    upstream_run_seed,
)


class ScenarioTests(unittest.TestCase):
    def test_segments_are_contiguous_and_cover_scenario(self) -> None:
        cursor = 0
        for segment in SEGMENTS:
            self.assertEqual(segment["start"], cursor)
            cursor += segment["frames"]
        self.assertEqual(cursor, FRAME_COUNT)
        self.assertEqual(scenario_manifest()["frame_count"], FRAME_COUNT)

    def test_scripted_boundaries(self) -> None:
        self.assertFalse(any(scenario_control(59)["keys"].values()))
        self.assertTrue(scenario_control(60)["keys"]["w"])
        self.assertEqual(scenario_control(119)["camera"]["azimuth_degrees"], 0.0)
        self.assertLess(scenario_control(120)["camera"]["azimuth_degrees"], 0.0)
        self.assertEqual(scenario_control(164)["camera"]["azimuth_degrees"], -90.0)
        self.assertFalse(any(scenario_control(165)["keys"].values()))
        self.assertTrue(scenario_control(225)["keys"]["w"])
        self.assertTrue(scenario_control(225)["keys"]["f"])
        self.assertFalse(any(scenario_control(285)["keys"].values()))

    def test_seed_sequences_are_stable(self) -> None:
        seeds = [plan_seed(1234, index) for index in range(20)]
        self.assertEqual(seeds, [plan_seed(1234, index) for index in range(20)])
        self.assertEqual(len(set(seeds)), len(seeds))
        self.assertEqual(upstream_run_seed(1234), 2_424_436_653)

    def test_invalid_frames_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            scenario_control(-1)
        with self.assertRaises(ValueError):
            scenario_control(FRAME_COUNT)


class ComparatorTests(unittest.TestCase):
    def _capture(self, root: Path, name: str, float_offset: float = 0.0) -> Path:
        import numpy as np
        from safetensors.numpy import save_file

        target = root / name
        target.mkdir()
        playback = target / "playback.safetensors"
        save_file({
            "values": np.asarray([[1.0 + float_offset, 2.0]], dtype=np.float32),
            "mode": np.asarray([1, 2], dtype=np.int64),
        }, playback)
        (target / "controls.jsonl").write_text('{"frame":0}\n', encoding="utf-8")
        (target / "events.jsonl").write_text('{"plan":0}\n', encoding="utf-8")
        manifest = {
            "format": CAPTURE_FORMAT,
            "harness": {"capture_session.py": "test"},
            "upstream": {"revision": "test", "trusted_files": {}},
            "scenario": {"name": "test"},
            "seeds": {"requested": 1},
            "demo_flags": {},
            "environment": {
                "python": "test", "packages": {}, "os_packages": {},
                "container_base": "test",
                "cuda_runtime": "test", "nvidia_driver": "test", "cudnn": 1,
                "gpu_name": "test", "gpu_compute_capability": [1, 0],
                "deterministic_algorithms": False, "cudnn_deterministic": False,
                "cudnn_benchmark": False, "cuda_matmul_allow_tf32": False,
                "cudnn_allow_tf32": False, "float32_matmul_precision": "highest",
                "cublas_workspace_config": None,
            },
            "capture": {"playback": {"file": playback.name}, "plans": []},
        }
        (target / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return target

    def test_comparator_accepts_equal_captures_and_measures_drift(self) -> None:
        from compare_session_captures import compare_capture_set

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            captures = [self._capture(root, "a"), self._capture(root, "b", 5e-7)]
            result = compare_capture_set(captures, 1e-5, 1e-6)
            self.assertTrue(result["pass"])
            self.assertGreater(result["tensors"]["playback.safetensors:values"]["max_abs"], 0.0)

    def test_comparator_rejects_discrete_difference(self) -> None:
        import numpy as np
        from safetensors.numpy import save_file
        from compare_session_captures import compare_capture_set

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self._capture(root, "a")
            second = self._capture(root, "b")
            save_file({
                "values": np.asarray([[1.0, 2.0]], dtype=np.float32),
                "mode": np.asarray([1, 3], dtype=np.int64),
            }, second / "playback.safetensors")
            result = compare_capture_set([first, second], 1e-5, 1e-6)
            self.assertFalse(result["pass"])
            self.assertTrue(any("discrete tensor differs" in item for item in result["failures"]))

    def test_comparator_measures_all_pairs(self) -> None:
        from compare_session_captures import compare_capture_set

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            captures = [
                self._capture(root, "a"),
                self._capture(root, "b", 6e-6),
                self._capture(root, "c", -6e-6),
            ]
            result = compare_capture_set(captures, 1e-5, 1.0)
            self.assertFalse(result["pass"])
            metric = result["tensors"]["playback.safetensors:values"]
            self.assertGreater(metric["max_abs"], 1e-5)


if __name__ == "__main__":
    unittest.main()
