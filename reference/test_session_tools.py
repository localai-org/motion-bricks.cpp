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

    def test_core_comparison_allows_trace_tensors_and_harness_change(self) -> None:
        import numpy as np
        from safetensors.numpy import save_file
        from compare_session_captures import compare_capture_set

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = self._capture(root, "reference")
            traced = self._capture(root, "traced")
            manifest = json.loads((traced / "manifest.json").read_text(encoding="utf-8"))
            manifest["harness"] = {"capture_session.py": "changed-by-tracing-support"}
            manifest["environment"]["os_packages"] = {"renderer-library": "added"}
            (traced / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            save_file({
                "values": np.asarray([[1.0, 2.0]], dtype=np.float32),
                "mode": np.asarray([1, 2], dtype=np.int64),
                "target_trace": np.asarray([3.0], dtype=np.float32),
            }, traced / "playback.safetensors")
            full = compare_capture_set([reference, traced], 0.0, 0.0)
            core = compare_capture_set([reference, traced], 0.0, 0.0, core_only=True)
            self.assertFalse(full["pass"])
            self.assertTrue(core["pass"])


class ReplayTests(unittest.TestCase):
    def test_portable_replay_round_trip_and_truncation(self) -> None:
        import numpy as np
        from session_replay import Replay, read_replay, write_replay

        replay = Replay(
            fps=30,
            parents=np.asarray([-1, 0], dtype=np.int32),
            modes=np.asarray([0, 2], dtype=np.int32),
            frame_plans=np.asarray([-1, 0], dtype=np.int32),
            qpos=np.arange(6, dtype=np.float32).reshape(2, 3),
            joint_positions=np.arange(12, dtype=np.float32).reshape(2, 2, 3),
            plan_frames=np.asarray([1], dtype=np.uint32),
            plan_modes=np.asarray([2], dtype=np.int32),
            plan_valid_lengths=np.asarray([24], dtype=np.uint32),
            target_positions=np.arange(24, dtype=np.float32).reshape(1, 4, 2, 3),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "session.mbreplay"
            write_replay(path, replay)
            decoded = read_replay(path)
            np.testing.assert_array_equal(decoded.joint_positions, replay.joint_positions)
            np.testing.assert_array_equal(decoded.target_positions, replay.target_positions)
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "truncated"):
                read_replay(path)

    def test_target_uncanonicalization_restores_world_origin_and_heading(self) -> None:
        import numpy as np
        from build_session_replay import uncanonicalize_targets

        joints = np.zeros((1, 4, 1, 3), dtype=np.float32)
        roots = np.zeros((1, 4, 3), dtype=np.float32)
        roots[0, :, 2] = 1.0  # one metre along canonical MuJoCo +X
        world = uncanonicalize_targets(
            joints, roots,
            np.asarray([[2.0, 3.0, 0.0]], dtype=np.float32),
            np.asarray([np.pi / 2], dtype=np.float32),
        )
        np.testing.assert_allclose(world[:, 0], [[4.0, 0.0, 2.0]] * 4, atol=1e-6)


class PlanParityTests(unittest.TestCase):
    def test_fixture_round_trip_and_truncation(self) -> None:
        import numpy as np
        from plan_parity import Fixture, Plan, read_fixture, write_fixture

        parents = np.arange(-1, 33, dtype=np.int32)
        neutral = np.arange(34 * 3, dtype=np.float32).reshape(34, 3) / 100
        frames = 24
        context_rotations = np.zeros((4, 34, 4), dtype=np.float32)
        context_rotations[..., 3] = 1
        expected_rotations = np.zeros((frames, 34, 4), dtype=np.float32)
        expected_rotations[..., 3] = 1
        plan = Plan(
            command_frame=17, expected_frames=frames, mode=2, seed=1234,
            movement=np.asarray([0, 0, 1], dtype=np.float32),
            facing=np.asarray([0, 0, 1], dtype=np.float32),
            context_roots=np.zeros((4, 3), dtype=np.float32),
            context_local_rotations=context_rotations,
            expected_roots=np.zeros((frames, 3), dtype=np.float32),
            expected_local_rotations=expected_rotations,
            expected_joint_positions=np.zeros((frames, 34, 3), dtype=np.float32),
            target_roots=np.zeros((4, 3), dtype=np.float32),
            target_local_rotations=context_rotations.copy(),
            target_joint_positions=np.zeros((4, 34, 3), dtype=np.float32),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "open-loop.mbparity"
            write_fixture(path, Fixture(parents=parents, neutral_joints=neutral, plans=[plan]))
            decoded = read_fixture(path)
            self.assertEqual(decoded.plans[0].seed, 1234)
            np.testing.assert_array_equal(decoded.neutral_joints, neutral)
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "truncated"):
                read_fixture(path)

    def test_backend_report_comparator(self) -> None:
        from compare_plan_reports import compare

        report = {
            "format": "motionbricks-open-loop-report-v1", "device": "cpu", "fps": 30, "joints": 1,
            "plans": [{"index": 0, "command_frame": 4, "mode": 2, "style": "walk",
                       "seed": 1234, "expected_frames": 24, "actual_frames": 24,
                       "movement": [0, 0, 1], "facing": [0, 0, 1],
                       "native_roots": [0.0, 0.0, 0.0],
                       "native_rotations": [0.0, 0.0, 0.0, 1.0],
                       "native_joint_positions": [0.0, 0.0, 0.0],
                       "native_target_roots": [0.0, 0.0, 0.0],
                       "native_target_rotations": [0.0, 0.0, 0.0, 1.0],
                       "native_target_joint_positions": [0.0, 0.0, 0.0]}],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left, right, output = root / "cpu.json", root / "vulkan.json", root / "out.json"
            left.write_text(json.dumps(report), encoding="utf-8")
            report["device"] = "vulkan"
            report["plans"][0]["native_joint_positions"][0] = 0.002
            right.write_text(json.dumps(report), encoding="utf-8")
            left_traces, right_traces = root / "cpu-traces", root / "vulkan-traces"
            left_traces.mkdir()
            right_traces.mkdir()
            names = [
                "input.global_root_values", "input.local_root_values", "input.local_poses",
                "root.num_token_logits", "root.pred_global_root_values", "root.pred_local_root_values",
                "pose.logits", "decoder.quantized", "decoder.output",
            ]
            trace = {
                "format": "motionbricks-native-neural-trace-v1", "plan": 0, "selected_tokens": 6,
                "tensors": {name: [0.0] for name in names} | {"pose.tokens": [1, 2]},
            }
            (left_traces / "native-plan-000.json").write_text(json.dumps(trace), encoding="utf-8")
            trace["tensors"]["decoder.output"] = [0.001]
            (right_traces / "native-plan-000.json").write_text(json.dumps(trace), encoding="utf-8")
            result = compare(left, right, output, left_traces=left_traces, right_traces=right_traces)
            self.assertEqual(result["summary"]["duration_matches"], 1)
            self.assertAlmostEqual(result["summary"]["worst"]["joint_m"]["max"], 0.002)
            self.assertTrue(result["passed"])
            self.assertEqual(result["neural"]["pose_token_mismatches"], 0)
            self.assertEqual(result["neural"]["continuous_maxima"]["decoder.output"]["max_abs"], 0.001)
            report["plans"][0]["native_joint_positions"][0] = 0.0021
            right.write_text(json.dumps(report), encoding="utf-8")
            self.assertFalse(compare(left, right, output)["passed"])


if __name__ == "__main__":
    unittest.main()
