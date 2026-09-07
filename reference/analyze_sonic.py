#!/usr/bin/env python3
"""Report physical SONIC tracking diagnostics; does not claim neural parity."""
import argparse
import json
from pathlib import Path

import numpy as np


def active_physics(data, play):
    # Events occur before the next step. The previous post-step sample has
    # the same sim_time, but still belongs to the suspended/paused phase.
    return data['wall_time'] >= play['wall_time']


def analyze(run: Path, include_failed=False) -> dict:
    summary = json.loads((run / "summary.json").read_text())
    if summary["failure"] is not None and not include_failed:
        raise ValueError(f"failed run: {summary['failure']}")
    if summary["controller_exit"] != 0 and not include_failed:
        raise ValueError("controller did not exit cleanly")
    with np.load(run / "physics.npz", allow_pickle=False) as data:
        play = next(e for e in summary["events"] if e["name"] == "play_requested")
        active = active_physics(data, play)
        if not active.any() or data["band"][active].any():
            raise ValueError("no unassisted playback samples")
        for key in ("qpos", "qvel", "torque", "target_q", "target_dq", "kp", "kd"):
            if not np.isfinite(data[key]).all():
                raise ValueError(f"non-finite {key}")
        qpos = data["qpos"][active]
        physical = {
            "samples": int(active.sum()),
            "pelvis_height_min_m": float(qpos[:, 2].min()),
            "pelvis_height_max_m": float(qpos[:, 2].max()),
            "root_displacement_xy_m": float(np.linalg.norm(qpos[-1, :2] - qpos[0, :2])),
            "max_root_step_m": float(np.linalg.norm(np.diff(qpos[:, :3], axis=0), axis=1).max()),
            "contact_count_min": int(data["contacts"][active].min()),
            "contact_count_max": int(data["contacts"][active].max()),
            "unassisted": True,
        }
        if 'pd_input_q' in data:
            raw = (data['feedforward_tau'] + data['kp']*(data['target_q']-data['pd_input_q'])
                   + data['kd']*(data['target_dq']-data['pd_input_dq']))
            indices = data['body_actuator_indices']
            limits = data['torque_limits'][indices]
            expected = np.clip(raw,-limits,limits)
            # Initial frozen-state publications have no actuator step. They
            # record ctrl=0 even if default-angle F32 rounding implies a tiny
            # nonzero PD torque; validate every actual physical step instead.
            stepped=data['time']>0
            error = (expected - data['torque'][:,indices])[stepped]
            maximum = float(np.max(np.abs(error)))
            if not np.isfinite(maximum) or maximum > 1e-9:
                raise ValueError(f'PD target-to-torque boundary error: {maximum} Nm')
            physical['pd_to_torque_max_abs_error_nm'] = maximum
            physical['body_actuator_saturation_fraction'] = float(np.mean(np.abs(raw[active]) >= limits))
            physical['body_actuator_peak_limit_ratio'] = float(np.max(np.abs(raw[active])/limits))
    q = np.loadtxt(run / "state/q.csv", delimiter=",", skiprows=1, ndmin=2)
    target = np.loadtxt(run / "target-motion.csv", delimiter=",", usecols=range(36), ndmin=2)
    playing = np.loadtxt(run / "state/motion_playing.csv", delimiter=",", skiprows=1, ndmin=2)
    if q.shape != (len(target), 34) or playing.shape != (len(target), 6):
        raise ValueError("control-state and target row counts/shapes differ")
    if not np.array_equal(q[:, :5], playing[:, :5]):
        raise ValueError("control-state and playback timestamps/indices differ")
    if not np.array_equal(q[:, 0], np.arange(len(q))):
        raise ValueError("control-state indices are not consecutive")
    active = playing[:, 5] > 0.5
    if not active.any() or not all(np.isfinite(a).all() for a in (q, target, playing)):
        raise ValueError("missing or non-finite active control data")
    # Upstream GatherRobotStateToLogger records q in hardware/MuJoCo body
    # order. Its target-motion logger explicitly remaps IsaacLab targets into
    # that same order. No index inference, row truncation or alignment fitting.
    error = q[active, 5:] - target[active, 7:]
    log = (run / "controller.log").read_text()
    if "ERROR: AddressSanitizer" in log or "runtime error:" in log:
        raise ValueError("sanitizer finding in controller log")
    return {
        "schema": 1, "physical": physical, "active_control_frames": int(active.sum()),
        "failure": summary['failure'], "controller_exit": summary['controller_exit'],
        "joint_rmse_rad": float(np.sqrt(np.mean(error**2))),
        "joint_max_abs_error_rad": float(np.abs(error).max()),
        "joint_rmse_per_joint_rad": np.sqrt(np.mean(error**2, axis=0)).tolist(),
        "missed_physics_deadlines": summary["missed_physics_deadlines"],
        "interpretation": "Physical-run diagnostics, not neural parity or final tracking acceptance; no quality threshold applied.",
        "remaining_metrics": ["heading-aligned root/body error", "foot slip", "torque saturation"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.run)
    text = json.dumps(report, indent=2) + "\n"
    (args.run / "diagnostics.json").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
