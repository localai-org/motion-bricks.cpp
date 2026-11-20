#!/usr/bin/env python3
"""Measure experimental native mixed precision against the pinned CUDA capture.

Reports errors and discrete decisions without declaring a parity pass. Native
uses explicit half rounding with FP32 accumulation; CUDA reduction order can
change half rounding and must be measured rather than assumed equivalent.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from compare_native_taps import read_gguf
from compare_soma import compare,summary,compare_duration_decisions


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--native',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    manifest=json.loads((args.reference/'manifest.json').read_text())
    pin=json.loads(Path(__file__).with_name('soma-reference.json').read_text())
    if manifest['reference']!=pin:raise ValueError('Reference pin differs from the configured SOMA target')
    if manifest['precision']!='mixed-fp16' or manifest['provider']!='CUDAExecutionProvider':
        raise ValueError('Expected verified CUDA mixed-FP16 capture')
    with (args.reference/'tensors.npz').open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=manifest['capture_sha256']:
            raise ValueError('Reference capture checksum mismatch')
    metadata,native=read_gguf(args.native)
    if metadata.get('precision')!='mixed-fp16-experimental':raise ValueError('Expected experimental mixed-precision native output')
    with np.load(args.reference/'tensors.npz') as reference:
        tokens=metadata['soma.tokens']
        features=np.zeros((1,64,366),np.float32)
        features[0,:tokens*4]=native['output_features'].reshape(tokens*4,366)
        report=compare(reference,{'output/pred_num_tokens':np.array([tokens]),'output/pred_global_motions':features})
        report['duration_decisions']=compare_duration_decisions(reference,native,metadata)
        expected=reference['graph//mmm_net/_proj_pose_output_logit/Add_output_0'].reshape(-1,16)
        valid=int(reference['output/pred_num_tokens'].item())*5
        actual=native['pose_logits'].reshape(-1,16)
        count=min(valid,len(actual))
        if not np.isfinite(actual[:count]).all() or not np.isfinite(expected[:count]).all():
            raise ValueError('Non-finite pose logits')
        report['pose_token_mismatches']=int(np.count_nonzero(actual[:count].argmax(-1)!=expected[:count].argmax(-1)))
        report['pose_token_count_matches']=len(actual)==valid
        report['pose_token_differences']=[]
        for index in np.flatnonzero(actual[:count].argmax(-1)!=expected[:count].argmax(-1)):
            a=actual[index].astype(float);e=expected[index].astype(float)
            native_choice=int(a.argmax());reference_choice=int(e.argmax())
            report['pose_token_differences'].append(dict(token=int(index//5),head=int(index%5),
                reference_choice=reference_choice,native_choice=native_choice,
                reference_winner_margin=float(np.sort(e)[-1]-np.sort(e)[-2]),
                native_winner_margin=float(np.sort(a)[-1]-np.sort(a)[-2]),
                reference_choice_gap=float(e[reference_choice]-e[native_choice]),
                native_choice_gap=float(a[native_choice]-a[reference_choice])))
        report['compared_pose_tokens']=count
        report['pose_logit_absolute']=summary(np.abs(actual[:count].astype(float)-expected[:count].astype(float)))
        report['input_absolute']={name:summary(np.abs(native['input.'+name].ravel()-reference['input/'+name].ravel()))
                                  for name in ('global_root_values','local_root_values','local_poses')}
        report['root_input_half_boundaries']={}
        for name,node in [('normalized_global','Div_3'),('normalized_local','Div_4'),('normalized_poses','Div_5')]:
            a=native[name].ravel().astype(np.float16)
            e=reference['graph//'+node+'_output_0__fp16'].ravel()
            if a.shape!=e.shape or not np.isfinite(a).all() or not np.isfinite(e).all():
                raise ValueError('Invalid normalized root input boundary')
            report['root_input_half_boundaries'][name]=dict(values=int(a.size),mismatches=int(np.count_nonzero(a!=e)),
                absolute=summary(np.abs(a.astype(float)-e.astype(float))))
    report.update(format='motionbricks-soma-mixed-diagnostic-v1',reference=str(args.reference),native=str(args.native),
                  reference_capture_sha256=manifest['capture_sha256'],
                  reference_runtime_identity=manifest.get('runtime_identity'),
                  cuda_gemm_options=manifest.get('cuda_gemm_options','unrecorded'),
                  provider_options=manifest.get('provider_options'),
                  native_matmul_operands=metadata.get('arithmetic.matmul_operands','unrecorded'),
                  native_matmul_accumulation=metadata.get('arithmetic.matmul_accumulation','unrecorded'),
                  native_matmul_policy=metadata.get('arithmetic.matmul_policy','unrecorded'),
                  native_root_policy=metadata.get('arithmetic.root_policy','unrecorded'),
                  native_root_projection_policy=metadata.get('arithmetic.root_projection_policy','unrecorded'),
                  native_aligned_block_order=metadata.get('arithmetic.aligned_block_order','unrecorded'),
                  native_nvjet3_policy=metadata.get('arithmetic.nvjet3_policy','unrecorded'),
                  native_nvjet8_policy=metadata.get('arithmetic.nvjet8_policy','unrecorded'),
                  native_pose_scores_policy=metadata.get('arithmetic.pose_scores_policy','unrecorded'),
                  native_root_conv_policy=metadata.get('arithmetic.root_conv_policy','unrecorded'),
                  native_conditioning_policy=metadata.get('arithmetic.conditioning_policy','unrecorded'),
                  native_decoder_conv_policy=metadata.get('arithmetic.decoder_conv_policy','unrecorded'),
                  native_nvjet_split4_policy=metadata.get('arithmetic.nvjet_split4_policy','unrecorded'),
                  native_norm_policy=metadata.get('arithmetic.norm_policy','unrecorded'),
                  rsqrt_table_sha256=metadata.get('arithmetic.rsqrt_table_sha256'),
                  root_softmax_policy=metadata.get('arithmetic.root_softmax_policy','unrecorded'),
                  pose_softmax_policy=metadata.get('arithmetic.pose_softmax_policy','unrecorded'),
                  exp2_table_sha256=metadata.get('arithmetic.exp2_table_sha256'),
                  feature_source=metadata.get('input.feature_source','native-encoder'),
                  backend=metadata['backend'],status='experimental; parity not established')
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('duration mismatches:',report['token_count_mismatches'],'pose token mismatches:',report['pose_token_mismatches'],
          'max raw rotation deg:',report['requests'][0]['joint_rotation_deg']['max'])


if __name__=='__main__':main()
