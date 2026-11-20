#!/usr/bin/env python3
"""Recompute SOMA chain acceptance from reference captures and native outputs.

Checks both same-context replay and independent feedback. A pass covers this
captured chain and the recorded tolerances, not all SOMA inputs or precisions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from compare_native_taps import read_gguf
from compare_soma_playback import transform_errors
from run_soma_chain import pose_token_errors
from compare_soma import compare_duration_decisions


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def evaluate(expected, actual, expected_contacts, actual_contacts, decisions, limits):
    failures=[];metrics={}
    for name in ('world','corrected'):
        if expected[name].shape!=actual[name].shape:
            failures.append(name+': duration or skeleton shape mismatch')
            continue
        distance,angle=transform_errors(expected[name],actual[name])
        metrics[name]=dict(position_m=float(distance.max()),root_position_m=float(distance[:,0].max()),
                           rotation_deg=float(angle.max()))
        if distance.max()>limits['position_m']:failures.append(name+': position ceiling exceeded')
        if angle.max()>limits['rotation_deg']:failures.append(name+': rotation ceiling exceeded')
    if expected_contacts.shape!=actual_contacts.shape:
        failures.append('contact shape mismatch')
    elif not np.isfinite(expected_contacts).all() or not np.isfinite(actual_contacts).all():
        raise ValueError('Nonfinite contacts')
    else:
        metrics['contacts_absolute']=float(np.abs(expected_contacts-actual_contacts).max())
        if metrics['contacts_absolute']>limits['contacts_absolute']:failures.append('contact ceiling exceeded')
    if not decisions['pose_token_count_matches'] or decisions['pose_token_mismatches']:
        failures.append('pose-token decisions differ')
    return dict(passed=not failures,failures=failures,metrics=metrics,decisions=decisions)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chain',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    policy=Path(__file__).with_name('soma-chain-tolerances.json')
    limits=json.loads(policy.read_text())
    if (limits['format']!='motionbricks-soma-chain-tolerances-v1' or
            not limits['duration_exact'] or not limits['pose_tokens_exact'] or
            any(not np.isfinite(limits[k]) or limits[k]<=0 for k in ('position_m','rotation_deg','contacts_absolute'))):
        raise ValueError('Invalid acceptance policy')
    pin=json.loads(Path(__file__).with_name('soma-reference.json').read_text())
    report=json.loads((args.chain/'report.json').read_text())
    if report['format']!='motionbricks-soma-chain-v1' or report['source_revision']!=pin['source_revision']:
        raise ValueError('Unrecognized or unpinned chain')
    if report['precision'] not in ('fp32','mixed-fp16') or not report['steps']:
        raise ValueError('Invalid chain precision or empty chain')
    records=[];joined={name:[] for name in ('reference','replay','closed_loop')}
    previous_reference=previous_native=None
    for index,step in enumerate(report['steps']):
        if step['index']!=index:raise ValueError('Nonsequential chain steps')
        directory=args.chain/f'step-{index:03d}'
        source=Path(step.get('reference_step',directory))
        capture=source/'capture';playback=source/'playback'
        capture_manifest=json.loads((capture/'manifest.json').read_text())
        if capture_manifest['reference']!=pin or capture_manifest['precision']!=report['precision']:
            raise ValueError('Reference capture provenance differs')
        capture_hash=digest(capture/'tensors.npz')
        if capture_hash!=capture_manifest['capture_sha256']:
            raise ValueError('Reference capture checksum mismatch')
        if report['precision']=='mixed-fp16' and capture_manifest.get('provider')!='CUDAExecutionProvider':
            raise ValueError('Mixed-FP16 reference must execute on CUDA')
        playback_manifest=json.loads((playback/'manifest.json').read_text())
        if playback_manifest['source_revision']!=pin['source_revision'] or playback_manifest['playback_implementation']!='upstream-warp-cpu':
            raise ValueError('Playback must use pinned upstream kernels')
        with np.load(capture/'tensors.npz') as data:
            reference_tokens=int(data['output/pred_num_tokens'].item())
        if playback_manifest['tokens']!=reference_tokens or playback_manifest['frames']!=4*reference_tokens-6:
            raise ValueError('Reference playback duration differs from the neural capture')
        _,fixture=read_gguf(playback/'playback.gguf')
        with np.load(playback/'tensors.npz') as data:
            expected={'world':data['corrected_world'],'corrected':data['corrected_local']}
            contacts=data['contacts']
        joints=expected['world'].shape[1]
        if len(expected['world'])!=playback_manifest['frames']:
            raise ValueError('Reference playback frame metadata is inconsistent')
        if step['reference_frames']!=len(expected['world']):
            raise ValueError('Chain report reference duration differs from captured playback')
        if joints!=len(report['joint_names']):raise ValueError('Joint count changed')
        if previous_reference is not None and not np.array_equal(fixture['context'].reshape(4,joints,7),previous_reference):
            raise ValueError('Reference chain did not feed back its last four corrected poses')
        previous_reference=expected['corrected'][-4:].copy()
        joined['reference'].append(expected['world'][1:] if index else expected['world'])
        for mode in ('replay','closed_loop'):
            path=directory/('closed.gguf' if mode=='closed_loop' and index else 'replay.gguf')
            metadata,native=read_gguf(path)
            if metadata.get('input.feature_source','native-encoder')!='native-encoder':
                raise ValueError('Captured-feature diagnostics cannot validate a native pose-to-pose chain')
            precision='fp32' if report['precision']=='fp32' else 'mixed-fp16-experimental'
            if metadata['precision']!=precision or metadata['backend']!=report['backend']:
                raise ValueError('Native precision/backend differs from the chain report')
            if metadata['soma.joints']!=joints:raise ValueError('Native skeleton metadata differs')
            if mode=='closed_loop' and index:
                _,closed_input=read_gguf(directory/'closed-input.gguf')
                if not np.array_equal(closed_input['context'].reshape(4,joints,7),previous_native):
                    raise ValueError('Native chain did not feed back its own corrected poses')
                for name in ('target','parents','pose_to_model','reference','model_to_pose','target_count','heading_joints'):
                    if not np.array_equal(closed_input[name],fixture[name]):
                        raise ValueError('Closed-loop request changed '+name)
            actual={name:np.asarray(native[name]).reshape(-1,joints,7) for name in ('world','corrected')}
            if metadata['soma.frames']!=len(actual['world']) or metadata['soma.tokens']*4-6!=len(actual['world']):
                raise ValueError('Native output duration metadata is inconsistent')
            declared=step['comparisons'][mode]
            if (declared['frames']!=len(actual['world']) or
                    declared['duration_matches']!=(len(expected['world'])==len(actual['world']))):
                raise ValueError('Chain report native duration differs from evaluated outputs')
            if not np.isfinite(native['pose_logits']).all() or not np.isfinite(native['output_features']).all():
                raise ValueError('Nonfinite native neural output')
            result=evaluate(expected,actual,contacts,native['contacts'].reshape(-1,4),pose_token_errors(capture,native),limits)
            with np.load(capture/'tensors.npz') as reference:
                duration=compare_duration_decisions(reference,native,metadata)
            result['duration_decisions']=duration
            if not duration['allowed_masks_match'] or not all(duration[side]['selection_consistent'] for side in ('reference','native')):
                result['failures'].append('duration mask or argmax selection is inconsistent')
                result['passed']=False
            result.update(index=index,mode=mode,native_sha256=digest(path),
                          reference_capture_sha256=capture_hash,reference_playback_sha256=digest(playback/'tensors.npz'),
                          matmul_operands=metadata.get('arithmetic.matmul_operands','unrecorded'),
                          root_arithmetic_policy=metadata.get('arithmetic.root_policy','unrecorded'),
                          matmul_arithmetic_policy=metadata.get('arithmetic.matmul_policy','unrecorded'),
                          norm_arithmetic_policy=metadata.get('arithmetic.norm_policy','unrecorded'),
                          rsqrt_table_sha256=metadata.get('arithmetic.rsqrt_table_sha256'))
            records.append(result)
            joined[mode].append(actual['world'][1:] if index else actual['world'])
            if mode=='closed_loop':previous_native=actual['corrected'][-4:].copy()
    # The demo must show these actual step outputs, including shared-endpoint removal.
    with np.load(args.chain/'chains.npz') as demo:
        for name,parts in joined.items():
            if not np.array_equal(demo[name],np.concatenate(parts)):
                raise ValueError('Demo chain differs from evaluated outputs: '+name)
    output=dict(format='motionbricks-soma-chain-acceptance-v1',passed=all(r['passed'] for r in records),
                source_revision=pin['source_revision'],precision=report['precision'],backend=report['backend'],
                chain_report_sha256=digest(args.chain/'report.json'),chain_sha256=digest(args.chain/'chains.npz'),
                tolerances=limits,tolerances_sha256=digest(policy),comparisons=records,
                scope='This captured chain only; does not establish complete SOMA parity')
    args.output.write_text(json.dumps(output,indent=2)+'\n')
    print(('PASS' if output['passed'] else 'FAIL'),args.chain)
    for row in records:
        if not row['passed']:print(row['index'],row['mode'],', '.join(row['failures']))
    if not output['passed']:raise SystemExit(1)


if __name__=='__main__':main()
