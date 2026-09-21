#!/usr/bin/env python3
"""Convert pinned SONIC ONNX MLPs, optionally including SMPL mode 2; never read pickle.

Mode 1 and other model releases remain unsupported. Tensor extraction is
checked against ONNX initializer shapes.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import onnx
from onnx import numpy_helper

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from convert_to_gguf import write_component, kv_string, sha256

HASHES={'encoder':'013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3',
        'decoder':'c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed'}
DIMS={'encoder':[640,2048,1024,512,512,64],'decoder':[994,2048,2048,1024,1024,512,512,29]}

SMPL_DIMS=[840,2048,1024,512,512,64]

def load_models(directory):
    models={}
    for key,wanted in HASHES.items():
        path=directory/f'model_{key}.onnx'
        if sha256(path)!=wanted: raise ValueError(f'unsupported/corrupt official SONIC {key}')
        model=onnx.load(path,load_external_data=False)
        onnx.checker.check_model(model)
        if any(t.data_location for t in model.graph.initializer): raise ValueError('external tensors not supported')
        models[key]=model
    return models

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--include-smpl',action='store_true',help='include human-pose mode 2 alongside G1 mode 0')
    args=p.parse_args()
    if args.output.exists(): p.error('output exists; use a new filename')
    models=load_models(args.policy)
    values=[]
    mapping={}
    components=[('encoder',models['encoder'],DIMS['encoder'],'g1'),
                ('decoder',models['decoder'],DIMS['decoder'],'g1_dyn')]
    if args.include_smpl: components.append(('smpl_encoder',models['encoder'],SMPL_DIMS,'smpl'))
    for key,model,dims,branch in components:
        initializers={t.name:numpy_helper.to_array(t) for t in model.graph.initializer}
        for layer,(nin,nout) in enumerate(zip(dims,dims[1:])):
            original=(f'module.encoders.{branch}.module.{layer*2}.weight' if key!='decoder' else f'onnx::MatMul_{136+layer}')
            bias=f'module.{"decoders" if key=="decoder" else "encoders"}.{branch}.module.{layer*2}.bias'
            w=initializers[original] if key!='decoder' else initializers[original].T
            b=initializers[bias]
            if w.shape!=(nout,nin) or b.shape!=(nout,): raise ValueError('unexpected initializer shape')
            for suffix,array,name in [('weight',w,original),('bias',b,bias)]:
                if array.dtype!=np.float32 or not np.isfinite(array).all(): raise ValueError('invalid initializer')
                target=f'{key}.{layer}.{suffix}'
                values.append((target,np.ascontiguousarray(array)))
                mapping[target]={'onnx_name':name,'transpose':key=='decoder' and suffix=='weight'}
    meta=[kv_string('sonic.architecture','g1-smpl-mode02-mlp-fsq32-v1' if args.include_smpl else 'g1-mode0-mlp-fsq32-v1')]
    meta += [kv_string(f'sonic.{key}_sha256',value) for key,value in HASHES.items()]
    report=write_component(args.output,'sonic',values,HASHES['encoder'],general_name='NVIDIA GEAR-SONIC G1 + SMPL' if args.include_smpl else 'NVIDIA GEAR-SONIC G1 mode 0',extra_metadata=meta)
    report.update(schema=1,source_onnx_sha256=HASHES,initializer_mapping=mapping,converter_sha256=sha256(Path(__file__)),
                  scope=('G1 mode 0 and SMPL mode 2' if args.include_smpl else 'G1 mode 0 only')+'; encoder FSQ32 and decoder; original released model, not low-latency/v1.1')
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__': main()
