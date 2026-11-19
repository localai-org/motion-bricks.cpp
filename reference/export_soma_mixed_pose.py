#!/usr/bin/env python3
"""Export CUDA pose boundaries and raw inputs for mixed-FP16 native probes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from convert_to_gguf import write_component,kv_string


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    manifest=json.loads((args.capture/'manifest.json').read_text())
    pin=json.loads(Path(__file__).with_name('soma-reference.json').read_text())
    if (manifest['reference']!=pin or manifest['precision']!='mixed-fp16' or
            manifest['provider']!='CUDAExecutionProvider' or manifest['tapped_graph']!='source'):
        raise ValueError('Requires pinned mixed-FP16 CUDA source-graph capture')
    with (args.capture/'tensors.npz').open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest()
    if digest!=manifest['capture_sha256']:raise ValueError('Capture checksum mismatch')
    # These names belong to the pinned graph. Inputs are the actual operands at
    # the neural boundary; root-predictor differences cannot propagate into this probe.
    with np.load(args.capture/'tensors.npz') as data:
        tokens=data['graph//mmm_net_1/Add_output_0']-np.arange(5)[None,None,:]*17
        roots=data['graph//mmm_net_1/Reshape_output_0__fp16'].reshape(1,64,4)
        poses=data['graph//Add_15_output_0__fp16']
        mask=data['graph//mmm_net_1/Cast_2_output_0'].reshape(1,64)
        duration=data['output/pred_num_tokens'].reshape(-1)
        logits=data['graph//mmm_net/_proj_pose_output_logit/Add_output_0']
        if data['graph//mmm_net_1/Add_2_output_0'].dtype!=np.float32:
            raise ValueError('Pinned pose positional addition is not FP32')
        if tokens.shape!=(1,16,5) or poses.shape!=(1,64,268) or logits.shape!=(1,16,80):
            raise ValueError('Unexpected pinned pose boundary shapes')
        if not np.isin(mask,[0,1]).all() or not np.isin(tokens,np.arange(17)).all():
            raise ValueError('Invalid pose inputs')
        if duration.shape!=(1,) or not 6<=duration[0]<=16:raise ValueError('Invalid duration')
        values=[('input.pose_tokens',tokens.astype(np.int32)),('input.root_values',roots.astype(np.float32)),
                ('input.pose_cond',poses.astype(np.float32)),('input.has_pose_cond',mask.astype(np.int32)),
                ('input.num_tokens',duration.astype(np.int32)),('output.pose_logits',logits.astype(np.float32))]
        for name in ('global_root_values','local_root_values','local_poses',
                     'has_global_root_values','has_local_root_values','has_local_poses','allowed_pred_num_tokens'):
            value=data['input/'+name]
            if not np.isfinite(value).all():raise ValueError('Nonfinite raw feature input')
            values.append(('full.'+name,value.astype(np.int32 if name.startswith(('has_','allowed_')) else np.float32)))
        values.append(('full.output',data['output/pred_global_motions'].astype(np.float32)))
    write_component(args.output,'soma-mixed-pose-fixture',values,digest,skeleton='soma30',
                    upstream_revision=pin['source_revision'],extra_metadata=[kv_string('fixture.precision','mixed-fp16')])


if __name__=='__main__':main()
