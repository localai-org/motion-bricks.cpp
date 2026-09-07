#!/usr/bin/env python3
"""Create timestamp-aligned reference/physical playback and drift diagnostics.

Align heading exactly as upstream's initial heading state (first logged base
quaternion minus first reference heading). Align horizontal translation ONCE
at first active control row. Do not align height, fit trajectories, or recenter
frames independently. Target root translation is diagnostic: G1 mode 0 does
not directly consume world root translation or a desired root velocity.
"""
import argparse
import json
from pathlib import Path
import re

import numpy as np
from scipy.spatial.transform import Rotation

from export_sonic_motion import MJ_TO_MOTION, physical_fk, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--title', required=True)
    p.add_argument('--include-failed',action='store_true',help='Export a clearly labelled failed run for diagnosis, never acceptance')
    args = p.parse_args()
    from analyze_sonic import analyze
    diagnostics = analyze(args.run,include_failed=args.include_failed)
    summary = json.loads((args.run/'summary.json').read_text())
    state = np.loadtxt(args.run/'state/q.csv',delimiter=',',skiprows=1)
    base = np.loadtxt(args.run/'state/base_quat.csv',delimiter=',',skiprows=1)
    flags = np.loadtxt(args.run/'state/motion_playing.csv',delimiter=',',skiprows=1)
    target = np.loadtxt(args.run/'target-motion.csv',delimiter=',',usecols=range(36))
    if not np.array_equal(state[:,:5],base[:,:5]):
        raise ValueError('base and joint state timelines differ')
    log = (args.run/'controller.log').read_text()
    if len(re.findall('Reset heading state to ',log)) != 1:
        raise ValueError('multiple heading resets need explicit per-event handling')
    # The same hardware-order names as upstream's low-command body motors.
    names = ([f'{s}_{j}_joint' for s in ('left','right') for j in
              ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')]
             + ['waist_yaw_joint','waist_roll_joint','waist_pitch_joint']
             + [f'{s}_{j}_joint' for s in ('left','right') for j in
                ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw')])
    def yaw(quat):
        matrix = Rotation.from_quat(np.asarray(quat)[[1,2,3,0]]).as_matrix()
        return np.arctan2(matrix[1,0],matrix[0,0])
    heading = yaw(base[0,5:9])-yaw(target[0,3:7])
    rotation = Rotation.from_euler('z',heading)
    active = flags[:,5] > .5
    timestamps = state[active,3]/1000-summary['monotonic_start_seconds']
    target = target[active].copy()
    target[:,:3] = rotation.apply(target[:,:3])
    target[:,3:7] = (rotation*Rotation.from_quat(target[:,[4,5,6,3]])).as_quat()[:,[3,0,1,2]]
    with np.load(args.run/'physics.npz',allow_pickle=False) as recorded:
        # Shutdown can record one final policy row just after the last physics
        # sample. Report and omit that unbracketed endpoint, never extrapolate.
        covered = (timestamps >= recorded['wall_time'][0]) & (timestamps <= recorded['wall_time'][-1])
        excluded = int((~covered).sum())
        if (excluded > 1 or not covered[0] or
                timestamps[-1] > recorded['wall_time'][-1]+summary['physics_dt']+.001):
            raise ValueError('control timeline extends beyond permitted final physics endpoint')
        timestamps,target = timestamps[covered],target[covered]
        diagnostics['playback_unbracketed_shutdown_rows'] = excluded
        physics = recorded['body_pos'].reshape(len(recorded['wall_time']),-1)
        actual_all = np.stack([np.interp(timestamps,recorded['wall_time'],physics[:,i]) for i in range(physics.shape[1])],axis=-1).reshape(len(timestamps),-1,3)
        offset = actual_all[0,1,:2]-target[0,:2]
        target[:,:2] += offset
        expected,_,body_names = physical_fk(args.scene,names,target)
        ids = [recorded['body_names'].tolist().index(name) for name in body_names]
        actual = actual_all[:,ids]
        parents = []
        for index in ids:
            parent = int(recorded['body_parents'][index])
            parents.append(ids.index(parent) if parent in ids else -1)
        if parents.count(-1) != 1 or parents[0] != -1:
            raise ValueError('physical body subset is not a rooted tree')
        contact = np.interp(timestamps,recorded['wall_time'],recorded['contacts']).round().astype(int)
    delta = actual-expected
    root_error = np.linalg.norm(delta[:,0,:2],axis=-1)
    body_error = np.linalg.norm(delta,axis=-1)
    relative_body_error = np.linalg.norm(delta-delta[:,0:1],axis=-1)
    diagnostics.update(root_xy_rmse_m=float(np.sqrt(np.mean(root_error**2))),
                       root_xy_final_error_m=float(root_error[-1]),
                       body_position_rmse_m=float(np.sqrt(np.mean(body_error**2))),
                       root_relative_body_rmse_m=float(np.sqrt(np.mean(relative_body_error**2))),
                       reference_displacement_xy_m=float(np.linalg.norm(expected[-1,0,:2]-expected[0,0,:2])))
    diagnostics['remaining_metrics'] = []
    for filename,key in (('contacts.json','contact_slip'),('boundary-checks.json','boundaries')):
        path=args.run/filename
        if path.exists():
            diagnostics[key]=json.loads(path.read_text())
        else:
            diagnostics['remaining_metrics'].append(key)
    # MuJoCo -> existing Y-up/Z-forward browser coordinates.
    actual,expected = actual@MJ_TO_MOTION.T,expected@MJ_TO_MOTION.T
    result = dict(format='motionbricks-sonic-playback-v1',title=args.title,
                  recorded=True,frames=len(timestamps),fps=50,
                  times=(timestamps-timestamps[0]).tolist(),
                  joints=[dict(name=name,parent=parent,position=expected[0,i].tolist()) for i,(name,parent) in enumerate(zip(body_names,parents))],
                  actual_positions=actual.reshape(-1).tolist(),reference_positions=expected.reshape(-1).tolist(),
                  root_error_m=root_error.tolist(),body_error_m=body_error.mean(axis=-1).tolist(),
                  contacts=contact.tolist(),diagnostics=diagnostics,
                  alignment=dict(heading_radians=float(heading),xy_offset=offset.tolist(),
                                 rule='initial controller heading; one XY translation at playback start; no height correction or subsequent realignment'),
                  provenance=summary['provenance'],physics_sha256=sha256(args.run/'physics.npz'))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,separators=(',',':'))+'\n')
    (args.run/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    print(json.dumps(diagnostics,indent=2))


if __name__ == '__main__':
    main()
