#!/usr/bin/env python3
"""Validate actual upstream encoder/decoder/motor boundaries from a run trace.

Reconstructs mode-0 future windows from the source CSV, heading alignment,
oldest-first state history, previous-action timing, and action-to-motor mapping.
No neural output is substituted for the policy. Tolerances are fixed here:
3e-6 for quaternion-derived observations, 1e-8 for rounded CSV state, and exact
F32 equality for source joint windows, tokens and motor targets.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from export_sonic_motion import HARDWARE_TO_ISAAC, sha256
from analyze_sonic import active_physics


def rotation(quat):
    return Rotation.from_quat(np.asarray(quat)[..., [1,2,3,0]])


def match_command(candidates, timestamp):
    """Repeated equal targets prove content, but cannot prove message freshness."""
    eligible = [row for row in candidates if row['command_time'] <= timestamp + .006]
    if not eligible:
        raise AssertionError('command applied before production')
    if len(candidates) > 1:
        return None
    row = eligible[0]
    if timestamp - row['command_time'] > .05:
        raise AssertionError('stale policy target applied for >50ms')
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--clip',type=Path,required=True)
    args = p.parse_args()
    rows = [json.loads(line) for line in (args.run/'boundaries.jsonl').read_text().splitlines()]
    source_q = np.loadtxt(args.clip/'joint_pos.csv',delimiter=',',skiprows=1)
    source_dq = np.loadtxt(args.clip/'joint_vel.csv',delimiter=',',skiprows=1)
    source_root = np.loadtxt(args.clip/'body_quat.csv',delimiter=',',skiprows=1)[:,:4]
    read = lambda name: np.loadtxt(args.run/f'state/{name}.csv',delimiter=',',skiprows=1)
    raw_q,raw_dq,raw_base,raw_angular,raw_action = map(read,('q','dq','base_quat','base_ang_vel','action'))
    if not rows or len(rows)!=len(raw_q):
        raise ValueError('trace and state row count mismatch')
    for array in (raw_dq,raw_base,raw_angular,raw_action):
        if not np.array_equal(array[:,:5],raw_q[:,:5]):
            raise ValueError('state timestamps/indices disagree')
    checks = {}
    def check(name,actual,expected,tol=3e-6):
        a,b = np.asarray(actual),np.asarray(expected)
        if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise AssertionError(f'{name}: shape/non-finite failure at trace {index}')
        maximum=float(np.max(np.abs(a-b)))
        previous=checks.setdefault(name,dict(max_abs_error=0.,tolerance=tol))
        previous['max_abs_error']=max(previous['max_abs_error'],maximum)
        if maximum>tol:
            raise AssertionError(f'{name}: {maximum} > {tol} at trace {index}')
    def yaw(quat):
        r=rotation(quat).as_matrix()
        return np.arctan2(r[1,0],r[0,0])
    heading=Rotation.from_euler('z',yaw(rows[0]['history'][-1]['base_quat'])-yaw(source_root[0]))
    inverse_order=np.argsort(HARDWARE_TO_ISAAC)
    clamped=padding=0
    for index,row in enumerate(rows):
        if row['index']!=index or row['mode']!=0 or len(row['history'])!=10:
            raise ValueError(f'unexpected trace index/mode/history at {index}')
        if not 0<=row['frame']<len(source_q):
            raise ValueError('invalid source frame')
        future=row['frame']+(np.arange(10)*5 if row['playing'] else np.zeros(10,dtype=int))
        clamped+=int(np.sum(future>=len(source_q)))
        future=np.minimum(future,len(source_q)-1)
        encoder=np.zeros(1762,dtype=np.float32)
        reference=rotation(row['history'][-1]['base_quat']).inv()*heading*rotation(source_root[future])
        fields={'encoder_mode_4':np.zeros(4),
                'motion_joint_positions_10frame_step5':source_q[future].reshape(-1),
                'motion_joint_velocities_10frame_step5':(source_dq[future] if row['playing'] else np.zeros((10,29))).reshape(-1),
                'motion_anchor_orientation_10frame_step5':reference.as_matrix()[...,:2].reshape(-1)}
        layout={x['name']:x for x in row['encoder_layout']}
        if not set(fields)<=layout.keys(): raise ValueError('required encoder observations missing')
        for name,value in fields.items():
            item=layout[name]
            if item['dimension']!=len(value): raise ValueError('encoder field dimension changed')
            start=item['offset']
            encoder[start:start+len(value)]=value
            actual=np.asarray(row['encoder_input'])[start:start+len(value)]
            check(name,actual,encoder[start:start+len(value)],3e-6 if 'orientation' in name else 0)
        check('encoder_all_fields_and_unused_zeros',row['encoder_input'],encoder)
        check('initial_heading_alignment',rotation(row['apply_heading']).as_matrix(),heading.as_matrix())
        h=row['history']
        # Zero quaternion padding is an upstream startup convention. Its
        # quat_rotate formula produces +Z gravity for those unavailable rows.
        quats=np.array([e['base_quat'] for e in h])
        gravity=np.tile([0.,0.,1.],(10,1))
        valid=np.linalg.norm(quats,axis=1)>0
        gravity[valid]=rotation(quats[valid]).inv().apply(np.tile([0.,0.,-1.],(valid.sum(),1)))
        padding+=int((~valid).sum())
        decoder=np.concatenate([row['tokens'],np.array([e['base_ang_vel'] for e in h]).reshape(-1),
                                np.array([e['body_q'] for e in h]).reshape(-1),
                                np.array([e['body_dq'] for e in h]).reshape(-1),
                                np.array([e['last_action'] for e in h]).reshape(-1),gravity.reshape(-1)]).astype(np.float32)
        check('decoder_history_packing',row['decoder_input'],decoder)
        check('encoder_token_to_decoder',row['decoder_input'][:64],row['tokens'],0)
        for j,entry in enumerate(h):
            source=index-9+j
            if source<0:
                for key in ('body_q','body_dq','base_ang_vel','last_action','base_quat'):
                    check('startup_zero_padding_'+key,entry[key],np.zeros(len(entry[key])),0)
                continue
            if entry['index']!=source: raise AssertionError('history order/cursor differs')
            check('history_joint_order_and_default_offset',entry['body_q'],
                  (raw_q[source,5:]-row['default_angles'])[HARDWARE_TO_ISAAC],1e-8)
            check('history_velocity_order',entry['body_dq'],raw_dq[source,5:][HARDWARE_TO_ISAAC],1e-8)
            check('history_base_quaternion',entry['base_quat'],raw_base[source,5:],1e-8)
            check('history_base_angular_velocity',entry['base_ang_vel'],raw_angular[source,5:],1e-8)
            check('history_previous_action',entry['last_action'],rows[source-1]['actions'] if source else np.zeros(29),0)
            check('history_action_csv',entry['last_action'],raw_action[source,5:],1e-8)
        wanted=(np.array(row['default_angles'])+np.array(row['action_scale'])*np.array(row['actions'])[inverse_order]).astype(np.float32)
        check('action_scale_order_and_motor_target',row['target_q'],wanted,0)
    # Match each applied active physical target to its exact published F32
    # command, then measure delivery latency. No nearest-pose matching.
    summary=json.loads((args.run/'summary.json').read_text())
    start=summary['monotonic_start_seconds']
    lookup={}
    for row in rows:
        lookup.setdefault(np.asarray(row['target_q'],dtype=np.float32).tobytes(),[]).append(row)
    first_delivery={}
    ambiguous_samples=0
    with np.load(args.run/'physics.npz',allow_pickle=False) as data:
        play=next((e for e in summary['events'] if e['name']=='play_requested'),None)
        eligible=active_physics(data,play) if play else (data['wall_time']+start>=rows[0]['command_time']+.02)
        for sample in np.flatnonzero(eligible):
            key=np.asarray(data['target_q'][sample],dtype=np.float32).tobytes()
            if key not in lookup: raise AssertionError('applied motor target did not come from a traced policy command')
            timestamp=float(data['wall_time'][sample])+start
            row=match_command(lookup[key],timestamp)
            if row is None:
                ambiguous_samples+=1
                continue
            first_delivery.setdefault(row['index'],timestamp-row['command_time'])
    periods=np.diff([r['state_time'] for r in rows])
    report=dict(schema=1,result='pass',trace_sha256=sha256(args.run/'boundaries.jsonl'),
                source_clip_manifest_sha256=sha256(args.clip/'manifest.json') if (args.clip/'manifest.json').exists() else None,
                control_rows=len(rows),active_rows=sum(r['playing'] for r in rows),checks=checks,
                clamped_future_frames=clamped,zero_quaternion_history_padding=padding,
                matched_physical_commands=len(first_delivery),
                identical_target_samples_excluded_from_latency=ambiguous_samples,
                first_delivery_max_seconds=max(first_delivery.values(),default=None),
                physical_run_failure=summary['failure'],
                control_period_max_seconds=float(periods.max()),
                control_period_p99_seconds=float(np.quantile(periods,.99)),
                observed_periods_over_30ms=int(np.sum(periods>.03)),
                interpretation='Actual buffers and physical targets validated; neural inference numerics are a separate check.')
    (args.run/'boundary-checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
