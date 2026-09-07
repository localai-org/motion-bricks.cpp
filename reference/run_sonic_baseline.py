#!/usr/bin/env python3
"""Drive the official C++ SONIC controller and upstream MuJoCo environment.

Simulation-only: both DDS endpoints use loopback inside one container. The
upstream observation, inference, motor mapping and physics code is executed,
not reimplemented here. We automate keyboard/drop events and record physics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pty
import re
import select
import subprocess
import time

import mujoco
import numpy as np
import yaml

from gear_sonic.utils.mujoco_sim.base_sim import DefaultEnv
from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import UnitreeSdk2Bridge
from unitree_sdk2py.core.channel import ChannelFactoryInitialize


class RecordedEnv(DefaultEnv):
    def check_fall(self):
        # Upstream auto-resets after falls. A reference test must retain the
        # failure, not teleport and continue as though tracking succeeded.
        self.fall = bool(self.mj_data.qpos[2] < 0.2)
        if self.fall:
            raise RuntimeError(f"robot fell: pelvis height {self.mj_data.qpos[2]:.4f} m")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--motions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--play-seconds", type=float, default=8.0)
    parser.add_argument("--startup-timeout", type=float, default=300.0)
    parser.add_argument('--settle-seconds',type=float,default=2.,
                        help='Unassisted paused hold before playback; use 0 for a dynamic first pose')
    args = parser.parse_args()
    if not 0 < args.play_seconds <= 300 or not 0 < args.startup_timeout <= 1800 or not 0<=args.settle_seconds<=10:
        parser.error("invalid playback duration or startup timeout")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists() or (output / "controller.log").exists():
        parser.error("output already contains a run; use a new directory")
    bundle = args.bundle.resolve()
    manifest = json.loads((bundle / "manifest.json").read_text())
    def sha256(path):
        with Path(path).open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    for record in manifest["files"]:
        path = bundle / record["path"]
        if record.get("type") == "symlink":
            if not path.is_symlink() or str(path.readlink()) != record["target"]:
                raise ValueError(f"asset symlink changed: {record['path']}")
        elif path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"artifact changed: {record['path']}")
    source_hashes = {}
    source_files = [f"gear_sonic/utils/mujoco_sim/{name}" for name in
                    ("base_sim.py", "unitree_sdk2py_bridge.py", "robot.py", "sim_utils.py", "metric_utils.py")]
    source_files += ["gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml",
                     "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp"]
    for relative in source_files:
        expected = subprocess.check_output(["git", "-C", str(args.upstream_root), "show",
                                            f"{manifest['source_revision']}:{relative}"])
        if (args.upstream_root / relative).read_bytes() != expected:
            raise ValueError(f"upstream simulation source changed: {relative}")
        source_hashes[relative] = hashlib.sha256(expected).hexdigest()
    provenance = {"artifact_manifest_sha256": sha256(bundle / "manifest.json"),
                  "controller_sha256": sha256(args.binary), "runner_sha256": sha256(__file__),
                  "source_revision": manifest["source_revision"],
                  "model_revision": manifest["model_revision"], "simulation_sources": source_hashes}
    instrumentation = json.loads((bundle/'instrumentation.json').read_text())
    instrumented_source = bundle/'source/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp'
    if sha256(instrumented_source) != instrumentation['instrumented_sha256']:
        raise ValueError('instrumented controller source identity changed')
    provenance['instrumentation'] = instrumentation
    provenance['startup'] = {'initial_pose':'upstream default_angles','suspended_control_seconds':1.,
                             'unassisted_paused_seconds':args.settle_seconds}
    motions = (args.motions.resolve() if args.motions else bundle/'assets/gear_sonic_deploy/reference/example')
    clips = [p for p in motions.iterdir() if p.is_dir() and not p.name.startswith('.')]
    if len(clips) != 1:
        raise ValueError('reference run requires exactly one clip directory')
    provenance['motion_files'] = {str(p.relative_to(motions)):sha256(p) for p in sorted(clips[0].iterdir())
                                  if p.suffix in ('.csv','.txt','.json')}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    config_path = args.upstream_root / "gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml"
    config = yaml.safe_load(config_path.read_text())
    config.update(enable_waist=True, INTERFACE="lo", DOMAIN_ID=0, USE_JOYSTICK=0)
    config["ROBOT_SCENE"] = str(bundle / "assets" / config["ROBOT_SCENE"])
    (output / "simulation-config.json").write_text(json.dumps(config, indent=2))
    ChannelFactoryInitialize(0, "lo")
    env = RecordedEnv(config, onscreen=False, offscreen=False)
    fk_data = mujoco.MjData(env.mj_model)
    # Initialize once, before the physical experiment, using the very same
    # standing angles as upstream InitControl. Do not let an unpowered robot
    # collapse into self-contact while TensorRT builds its cached engines or
    # the controller runs its pre-policy InitControl motor interpolation.
    parameters = (args.upstream_root / "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp").read_text()
    match = re.search(r"default_angles\s*=\s*\{([^}]+)\}", parameters)
    if match is None:
        raise ValueError("upstream standing angles not found")
    angles = np.fromstring(re.sub(r"//[^\n]*", "", match[1]), sep=",")
    if angles.shape != (29,) or not np.isfinite(angles).all():
        raise ValueError("invalid upstream standing angles")
    env.mj_data.qpos[env.body_joint_index + env.qpos_offset - 1] = angles
    mujoco.mj_forward(env.mj_model, env.mj_data)
    bridge = UnitreeSdk2Bridge(config)
    env.set_unitree_bridge(bridge)
    command = [str(args.binary.resolve()), "lo", str(bundle / "policy/model_decoder.onnx"),
               str(motions), "--encoder-file", str(bundle / "policy/model_encoder.onnx"),
               "--obs-config", str(bundle / "policy/observation_config.yaml"),
               "--disable-crc-check", "--input-type", "keyboard", "--output-type", "zmq",
               "--policy-precision", "32", "--enable-csv-logs", "--logs-dir", str(output / "state"),
               "--policy-input-logfile", str(output / "policy-input.csv"),
               "--target-motion-logfile", str(output / "target-motion.csv")]
    (output / "command.json").write_text(json.dumps(command, indent=2))
    master, slave = pty.openpty()
    child_env = dict(os.environ)
    # CUDA's virtual-address reservations conflict with ASan's inaccessible
    # shadow gap on some drivers. Keep instrumentation enabled while allowing
    # those reservations; see sonic_cuda_probe.cpp for the isolated repro.
    child_env.setdefault("ASAN_OPTIONS", "detect_leaks=0:halt_on_error=1:protect_shadow_gap=0")
    child_env.setdefault("UBSAN_OPTIONS", "halt_on_error=1:print_stacktrace=1")
    child_env['SONIC_TRACE_FILE'] = str(output/'boundaries.jsonl')
    child = subprocess.Popen(command, stdin=slave, stdout=slave, stderr=slave,
                             cwd=output, env=child_env, start_new_session=True)
    os.close(slave)
    os.set_blocking(master, False)
    rows = {name: [] for name in ("time", "wall_time", "qpos", "qvel", "body_pos", "body_quat",
                                 "torque", "target_q", "target_dq", "kp", "kd", "feedforward_tau",
                                 "pd_input_q", "pd_input_dq", "contacts", "band")}
    events = []
    started = time.monotonic()
    next_step = started
    init_ready = False
    physics_started = False
    control_start = drop_time = play_time = None
    last_status = started
    log_tail = ""
    failure = None
    missed_steps = 0

    def event(name):
        item = {"name": name, "wall_time": time.monotonic() - started,
                "sim_time": float(env.mj_data.time)}
        events.append(item)
        print(json.dumps(item), flush=True)

    try:
        with (output / "controller.log").open("wb") as log:
            while True:
                now = time.monotonic()
                if select.select([master], [], [], 0)[0]:
                    try:
                        chunk = os.read(master, 262144)
                    except OSError:
                        chunk = b""
                    log.write(chunk)
                    log.flush()
                    log_tail = (log_tail + chunk.decode(errors="replace"))[-32768:]
                if child.poll() is not None:
                    raise RuntimeError(f"controller exited with status {child.returncode}; see controller.log")
                if not init_ready and "Init Done" in log_tail:
                    init_ready = True
                    os.write(master, b"]")
                    event("start_requested")
                if control_start is None and "transitioning to CONTROL state" in log_tail:
                    control_start = now
                    event("control_started")
                if control_start is not None and drop_time is None and now - control_start >= 1:
                    env.elastic_band.enable = False
                    drop_time = now
                    event("suspension_removed")
                if drop_time is not None and play_time is None and now - drop_time >= args.settle_seconds:
                    os.write(master, b"t")
                    play_time = now
                    event("play_requested")
                if play_time is not None and now - play_time >= args.play_seconds:
                    event("playback_window_finished")
                    break
                if control_start is None and now - started > args.startup_timeout:
                    raise TimeoutError("controller startup timed out; see controller.log")
                # Freeze the received body command while upstream reads it for
                # this step, so the saved target really produced these torques.
                with bridge.low_cmd_lock:
                    applied = {name: [getattr(m, field) for m in bridge.low_cmd.motor_cmd[:29]]
                               for name, field in (("target_q", "q"), ("target_dq", "dq"),
                                                   ("kp", "kp"), ("kd", "kd"), ("feedforward_tau", "tau"))}
                    if bridge.use_sensor:
                        applied['pd_input_q'] = env.mj_data.sensordata[:29].copy()
                        applied['pd_input_dq'] = env.mj_data.sensordata[29:58].copy()
                    else:
                        applied['pd_input_q'] = env.mj_data.qpos[env.body_joint_index + env.qpos_offset - 1].copy()
                        applied['pd_input_dq'] = env.mj_data.qvel[env.body_joint_index + env.qvel_offset - 1].copy()
                    if bridge.low_cmd_received and control_start is not None:
                        if not physics_started:
                            physics_started = True
                            event("physics_started")
                        env.sim_step()
                    else:
                        bridge.PublishLowState(env.prepare_obs())
                if not np.isfinite(env.mj_data.qpos).all() or not np.isfinite(env.mj_data.qvel).all():
                    raise RuntimeError("non-finite physical state")
                if init_ready:
                    data = env.mj_data
                    # mj_step's cached body transforms need not correspond to
                    # the newly integrated qpos. Compute recording FK in a
                    # separate data object without changing physical state.
                    fk_data.qpos[:] = data.qpos
                    mujoco.mj_kinematics(env.mj_model, fk_data)
                    rows["time"].append(data.time)
                    rows["wall_time"].append(now - started)
                    for key, value in (("qpos", data.qpos), ("qvel", data.qvel),
                                       ("body_pos", fk_data.xpos), ("body_quat", fk_data.xquat),
                                       ("torque", data.ctrl)):
                        rows[key].append(value.copy())
                    for name, value in applied.items():
                        rows[name].append(value)
                    rows["contacts"].append(data.ncon)
                    rows["band"].append(env.elastic_band.enable)
                if now - last_status >= 5:
                    print(f"sim={env.mj_data.time:.2f}s height={env.mj_data.qpos[2]:.3f} "
                          f"control={control_start is not None} playing={play_time is not None}", flush=True)
                    last_status = now
                next_step += env.sim_dt
                delay = next_step - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -env.sim_dt:
                    missed_steps += 1
                    # Do not skip physics steps. Record lag and resume wall-clock
                    # pacing without a large burst of stale sensor publications.
                    next_step = time.monotonic()
    except Exception as exc:
        failure = str(exc)
        event("failure")
    finally:
        if child.poll() is None:
            os.write(master, b"o")
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        with (output / "controller.log").open("ab") as log:
            while select.select([master], [], [], 0)[0]:
                try:
                    chunk = os.read(master, 262144)
                except OSError:
                    break
                if not chunk:
                    break
                log.write(chunk)
        os.close(master)
        if failure is None and (not (output/'boundaries.jsonl').exists() or (output/'boundaries.jsonl').stat().st_size == 0):
            failure = 'controller did not produce required boundary trace'
        np.savez_compressed(output / "physics.npz", **{k: np.asarray(v) for k, v in rows.items()},
                            body_names=np.asarray([env.mj_model.body(i).name for i in range(env.mj_model.nbody)]),
                            body_parents=env.mj_model.body_parentid,
                            terminal_qpos=env.mj_data.qpos.copy(),terminal_qvel=env.mj_data.qvel.copy(),
                            terminal_time=env.mj_data.time,
                            torque_limits=env.torque_limit, body_actuator_indices=env.body_joint_index-1)
        summary = {"schema": 1, "failure": failure, "events": events,
                   "monotonic_start_seconds": started,
                   "physics_samples": len(rows["time"]), "physics_dt": env.sim_dt,
                   "wall_seconds": time.monotonic() - started, "missed_physics_deadlines": missed_steps,
                   "controller_exit": child.returncode, "mujoco_version": mujoco.__version__,
                   "upstream_auto_reset_disabled": True,
                   "initial_pose": "upstream default_angles; physics waits for CONTROL activation, not initialization motor commands",
                   "provenance": provenance,
                   "tracking_acceptance": "not yet evaluated"}
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
    if failure:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
