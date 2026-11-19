#!/usr/bin/env python3
"""Capture upstream FP32 chains, same-context native replay, and native closed-loop drift.

At 30 Hz each complete corrected transition feeds its last four poses into the
next request. Shared endpoint frames are emitted only once. This deterministic
chain exercises generation, not the interactive player's asynchronous scheduler.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from compare_native_taps import read_gguf
from compare_soma_playback import transform_errors
from compare_soma import summary
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from convert_to_gguf import write_component


def pose_token_errors(capture, native):
    with np.load(capture/'tensors.npz') as data:
        key='layer/model._mmm_pose_model.mmm_net/0/output/pose_logits'
        if key not in data:key='graph//mmm_net/_proj_pose_output_logit/Add_output_0'
        logits=data[key]
        tokens=int(data['output/pred_num_tokens'].item())
        expected=logits.reshape(-1,16).argmax(axis=-1)[:tokens*5]
    actual=native['pose_logits'].reshape(-1,16).argmax(axis=-1)
    count=min(len(actual),len(expected))
    return dict(compared_pose_tokens=count,pose_token_count_matches=len(actual)==len(expected),
                pose_token_mismatches=int(np.count_nonzero(actual[:count]!=expected[:count])))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream',type=Path)
    parser.add_argument('--checkpoints',type=Path)
    parser.add_argument('--reference-chain',type=Path,help='Reuse a previously captured upstream chain')
    parser.add_argument('--precision',choices=('fp32','mixed-fp16'))
    parser.add_argument('--onnx-model',type=Path,help='Official mixed-FP16 graph for CUDA reference capture')
    parser.add_argument('--onnx-python',type=Path,default=Path(sys.executable),help='Python interpreter with ONNX Runtime CUDA')
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--runner',type=Path,required=True)
    parser.add_argument('--motion')
    parser.add_argument('--target-times',type=float,nargs='+')
    parser.add_argument('--backend',choices=('cpu','vulkan'),default='cpu')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--runtime-artifact',type=Path,action='append',default=[],
                        help='Record and check an exact backend library, shader, or diagnostic patch used during capture')
    args=parser.parse_args()
    if args.reference_chain:
        reference_report=json.loads((args.reference_chain/'report.json').read_text())
        if reference_report['format']!='motionbricks-soma-chain-v1' or reference_report['precision'] not in ('fp32','mixed-fp16'):
            parser.error('Reference chain must be a SOMA chain')
        if args.precision is not None and args.precision!=reference_report['precision']:
            parser.error('Precision must match the reused reference chain')
        args.precision=reference_report['precision']
        times=[step['target_time'] for step in reference_report['steps']]
        if args.motion not in (None,reference_report['motion']) or (args.target_times is not None and args.target_times!=times):
            parser.error('Motion and target times must match the reused reference chain')
        args.motion=reference_report['motion'];args.target_times=times
    else:
        args.precision=args.precision or 'fp32'
        if any(value is None for value in (args.upstream,args.motion,args.target_times)):
            parser.error('Fresh captures require --upstream, --motion and --target-times')
        if args.precision=='fp32' and args.checkpoints is None:
            parser.error('FP32 captures require --checkpoints')
        if args.precision=='mixed-fp16' and args.onnx_model is None:
            parser.error('Mixed-FP16 captures require --onnx-model and a CUDA-capable --onnx-python')
    def runtime_identity():
        identities=[]
        for path in args.runtime_artifact:
            with path.open('rb') as stream:
                identities.append(dict(path=str(path.resolve()),sha256=hashlib.file_digest(stream,'sha256').hexdigest()))
        return identities
    runtime_artifacts=runtime_identity()
    args.output.mkdir(parents=True,exist_ok=False)
    scripts=Path(__file__).resolve().parent
    def run(script,*arguments):
        subprocess.run([sys.executable,str(scripts/script),*map(str,arguments)],check=True)
    def native(fixture,output):
        with output.with_suffix('.log').open('w') as log:
            subprocess.run([str(args.runner),str(args.bundle),str(fixture),str(output),args.backend,args.precision],
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        return read_gguf(output)
    reference_context=None
    native_context=None
    chains={name:[] for name in ('reference','replay','closed_loop')}
    steps=[]
    for index,time in enumerate(args.target_times):
        directory=args.output/f'step-{index:03d}'
        directory.mkdir()
        poses=directory/'poses';capture=directory/'capture';playback=directory/'playback'
        extra=[]
        if args.reference_chain:
            source=args.reference_chain/f'step-{index:03d}'
            # Follow the original capture when reusing a native replay report.
            source=Path(reference_report['steps'][index].get('reference_step',source))
            poses=source/'poses';capture=source/'capture';playback=source/'playback'
        elif reference_context is not None:
            context_path=directory/'reference-context.npy';np.save(context_path,reference_context)
            extra=['--context-poses',context_path]
        if not args.reference_chain:
            run('prepare_soma_inputs.py','--upstream',args.upstream,'--motion',args.motion,
                '--target-time',time,'--output',poses,*extra)
            if args.precision=='fp32':
                run('capture_soma.py','--upstream',args.upstream,'--checkpoints',args.checkpoints,
                    '--inputs',poses/'inputs.npz','--output',capture)
            else:
                subprocess.run([str(args.onnx_python),str(scripts/'capture_soma_onnx.py'),
                    '--model',str(args.onnx_model),'--inputs',str(poses/'inputs.npz'),
                    '--output',str(capture),'--provider','cuda','--tap-layers'],check=True)
            run('capture_soma_playback.py','--upstream',args.upstream,'--poses',poses,
                '--capture',capture,'--output',playback)
        pose_manifest=json.loads((poses/'manifest.json').read_text())
        capture_manifest=json.loads((capture/'manifest.json').read_text())
        if capture_manifest['precision']!=args.precision:
            raise ValueError('Reference capture precision differs from the chain')
        if args.precision=='mixed-fp16' and capture_manifest.get('provider')!='CUDAExecutionProvider':
            raise ValueError('Mixed-FP16 chain requires CUDA reference arithmetic')
        with (capture/'tensors.npz').open('rb') as stream:
            capture_digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if capture_digest!=capture_manifest['capture_sha256']:
            raise ValueError('Reference neural capture checksum mismatch')
        if args.reference_chain:
            for key in ('source_revision','motion_sha256','skeleton_sha256'):
                if pose_manifest[key]!=reference_report[key]:
                    raise ValueError(f'Reused reference {key} mismatch')
        ref=np.load(playback/'tensors.npz')
        ref_local=ref['corrected_local'];ref_world=ref['corrected_world']
        replay_metadata,replay=native(playback/'playback.gguf',directory/'replay.gguf')
        _,fixture=read_gguf(playback/'playback.gguf')
        if native_context is not None:
            values=[(name,np.array(native_context if name=='context' else value)) for name,value in fixture.items()]
            write_component(directory/'closed-input.gguf','chain-input',values,
                            hashlib.sha256(native_context.tobytes()).hexdigest(),skeleton='soma30',
                            upstream_revision=pose_manifest['source_revision'])
            closed_metadata,closed=native(directory/'closed-input.gguf',directory/'closed.gguf')
        else:
            closed_metadata,closed=replay_metadata,replay
        step=dict(index=index,target_time=time,reference_frames=len(ref_local),comparisons={})
        step['reference_capture_sha256']=capture_digest
        step['reference_runtime']=capture_manifest.get('runtime','pytorch')
        step['reference_runtime_identity']=capture_manifest.get('runtime_identity')
        step['reference_provider']=capture_manifest.get('provider','pytorch-cpu')
        step['reference_cuda_gemm_options']=capture_manifest.get('cuda_gemm_options')
        step['reference_step']=str(poses.parent.resolve())
        step['reference_fixture_sha256']=hashlib.sha256((playback/'playback.gguf').read_bytes()).hexdigest()
        if args.reference_chain:
            expected=reference_report['steps'][index].get('reference_fixture_sha256')
            if expected and expected!=step['reference_fixture_sha256']:
                raise ValueError('Reused reference fixture checksum mismatch')
        joints=len(pose_manifest['joint_names'])
        for name,metadata,data in [('replay',replay_metadata,replay),('closed_loop',closed_metadata,closed)]:
            actual=np.array(data['world']).reshape(-1,joints,7)
            count=min(len(actual),len(ref_world))
            distance,angle=transform_errors(ref_world[:count],actual[:count])
            step['comparisons'][name]=dict(duration_matches=len(actual)==len(ref_world),
                matmul_operands=metadata.get('arithmetic.matmul_operands','unrecorded'),
                matmul_accumulation=metadata.get('arithmetic.matmul_accumulation','unrecorded'),
                root_arithmetic_policy=metadata.get('arithmetic.root_policy','unrecorded'),
                root_projection_policy=metadata.get('arithmetic.root_projection_policy','unrecorded'),
                matmul_arithmetic_policy=metadata.get('arithmetic.matmul_policy','unrecorded'),
                nvjet3_arithmetic_policy=metadata.get('arithmetic.nvjet3_policy','unrecorded'),
                nvjet8_arithmetic_policy=metadata.get('arithmetic.nvjet8_policy','unrecorded'),
                pose_scores_arithmetic_policy=metadata.get('arithmetic.pose_scores_policy','unrecorded'),
                root_conv_arithmetic_policy=metadata.get('arithmetic.root_conv_policy','unrecorded'),
                conditioning_arithmetic_policy=metadata.get('arithmetic.conditioning_policy','unrecorded'),
                decoder_conv_arithmetic_policy=metadata.get('arithmetic.decoder_conv_policy','unrecorded'),
                nvjet_split4_arithmetic_policy=metadata.get('arithmetic.nvjet_split4_policy','unrecorded'),
                norm_arithmetic_policy=metadata.get('arithmetic.norm_policy','unrecorded'),
                rsqrt_table_sha256=metadata.get('arithmetic.rsqrt_table_sha256'),
                  root_softmax_policy=metadata.get('arithmetic.root_softmax_policy','unrecorded'),
                  pose_softmax_policy=metadata.get('arithmetic.pose_softmax_policy','unrecorded'),
                  exp2_table_sha256=metadata.get('arithmetic.exp2_table_sha256'),
                frames=len(actual),joint_position_m=summary(distance),joint_rotation_deg=summary(angle),
                root_position_m=summary(distance[:,0]),
                per_frame=dict(root_position_m=distance[:,0].tolist(),
                               max_joint_position_m=distance.max(axis=1).tolist(),
                               max_joint_rotation_deg=angle.max(axis=1).tolist()))
            step['comparisons'][name].update(pose_token_errors(capture,data))
            chains[name].append(actual[1:] if index else actual)
        chains['reference'].append(ref_world[1:] if index else ref_world)
        reference_context=ref_local[-4:].copy()
        native_context=np.array(closed['corrected']).reshape(-1,joints,7)[-4:].copy()
        steps.append(step)
        print(f'Completed SOMA chain step {index+1}/{len(args.target_times)}',flush=True)
    np.savez(args.output/'chains.npz',**{name:np.concatenate(parts) for name,parts in chains.items()},
             parents=fixture['parents'])
    report=dict(format='motionbricks-soma-chain-v1',source_revision=pose_manifest['source_revision'],
        precision=args.precision,backend=args.backend,motion=args.motion,motion_sha256=pose_manifest['motion_sha256'],
        runner_sha256=hashlib.sha256(args.runner.read_bytes()).hexdigest(),
        bundle_manifest_sha256=hashlib.sha256((args.bundle/'manifest.json').read_bytes()).hexdigest(),
        skeleton_sha256=pose_manifest['skeleton_sha256'],sample_rate=30,
        joint_names=pose_manifest['joint_names'],steps=steps,
        context_policy='last four endpoint-corrected poses; shared endpoints emitted once',
        duration_mismatches=sum(not c['duration_matches'] for step in steps for c in step['comparisons'].values()))
    if runtime_identity()!=runtime_artifacts:
        raise ValueError('Runtime artifacts changed while the chain was executing')
    if runtime_artifacts:report['runtime_artifacts']=runtime_artifacts
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f"Chain report: {args.output/'report.json'}")
    if report['duration_mismatches']:
        raise SystemExit('Duration mismatch: per-step common-prefix errors do not establish parity')


if __name__=='__main__':main()
