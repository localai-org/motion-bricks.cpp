#!/usr/bin/env python3
"""Export measured SOMA chains for the existing demo overlay viewer."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from compare_soma_playback import transform_errors
from compare_soma import summary


def build_plan(index, name, expected, actual, steps):
    for step in steps:
        comparison=step['comparisons'][name]
        if comparison['duration_matches'] != (step['reference_frames']==comparison['frames']):
            raise ValueError('Transition duration flag contradicts its frame counts')
    aligned = 0
    for step_index, step in enumerate(steps):
        if not step['comparisons'][name]['duration_matches']: break
        aligned += step['reference_frames'] - (1 if step_index else 0)
    duration_exact = all(step['comparisons'][name]['duration_matches'] for step in steps)
    expected_frames = sum(step['reference_frames'] for step in steps) - len(steps) + 1
    actual_frames = sum(step['comparisons'][name]['frames'] for step in steps) - len(steps) + 1
    if len(expected) != expected_frames or len(actual) != actual_frames or aligned > min(len(expected),len(actual)):
        raise ValueError('Chain arrays do not match transition frame counts')
    if aligned:
        distance,angle=transform_errors(expected[:aligned],actual[:aligned])
        metrics=dict(root_m=summary(distance[:,0]),joint_m=summary(distance),rotation_deg=summary(angle))
        per_frame=dict(root_m=distance[:,0].tolist(),rotation_deg=angle.max(axis=1).tolist())
    else:
        metrics=dict(root_m=None,joint_m=None,rotation_deg=None)
        per_frame=dict(root_m=[],rotation_deg=[])
    return dict(index=index,style=name,expected_frames=len(expected),actual_frames=len(actual),
        aligned_frames=aligned,error_scope='matching-duration prefix',
        pose_token_mismatches=sum(step['comparisons'][name]['pose_token_mismatches'] for step in steps),
        duration_exact=duration_exact,movement=[0,0,1],facing=[0,0,1],
        expected_joint_positions=expected[...,:3].reshape(-1).tolist(),
        native_joint_positions=actual[...,:3].reshape(-1).tolist(),
        expected_target_joint_positions=expected[-4:,...,:3].reshape(-1).tolist(),
        native_target_joint_positions=actual[-4:,...,:3].reshape(-1).tolist(),
        metrics=metrics,per_frame=per_frame)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chain',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--acceptance',type=Path,help='Verified check_soma_chain.py result for this exact chain')
    args=parser.parse_args()
    report=json.loads((args.chain/'report.json').read_text())
    data=np.load(args.chain/'chains.npz')
    expected=data['reference'];plans=[]
    for index,name in enumerate(('closed_loop','replay')):
        plans.append(build_plan(index,name,expected,data[name],report['steps']))
    output=dict(format='motionbricks-soma-chain-view-v1',device=report['backend'],fps=30,
        joints=expected.shape[1],parents=data['parents'].tolist(),joint_names=report['joint_names'],
        neutral_joints=expected[0,:,:3].reshape(-1).tolist(),plans=plans,
        source_revision=report['source_revision'],precision=report['precision'],
        context_policy=report['context_policy'],steps=report['steps'],passed=False,
        validation_status='Measured chain errors; full parity acceptance remains in progress')
    output['native_provenance']={key:report[key] for key in
        ('runner_sha256','bundle_manifest_sha256','runtime_artifacts') if key in report}
    if args.acceptance:
        acceptance=json.loads(args.acceptance.read_text())
        def digest(path):
            with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
        if (acceptance['format']!='motionbricks-soma-chain-acceptance-v1' or
                acceptance['chain_sha256']!=digest(args.chain/'chains.npz') or
                acceptance['chain_report_sha256']!=digest(args.chain/'report.json') or
                any(acceptance[key]!=report[key] for key in ('precision','backend','source_revision'))):
            raise ValueError('Acceptance report does not describe this exact chain')
        if (type(acceptance['passed']) is not bool or
                len(acceptance['comparisons'])!=2*len(report['steps']) or
                acceptance['passed']!=all(row['passed'] for row in acceptance['comparisons'])):
            raise ValueError('Inconsistent acceptance report')
        if acceptance['passed'] and any(not plan['duration_exact'] for plan in plans):
            raise ValueError('Cannot accept a duration-mismatched chain')
        output['chain_validation']=dict(passed=acceptance['passed'],scope=acceptance['scope'],
                                        tolerances=acceptance['tolerances'],sha256=digest(args.acceptance))
    args.output.write_text(json.dumps(output,separators=(',',':'))+'\n')
    print(args.output)


if __name__=='__main__':main()
