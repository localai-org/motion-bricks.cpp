#!/usr/bin/env python3
"""Convert only the pinned official ONNX G1 mode-0 MLPs; never read pickle.

Other SONIC encoder modes are intentionally not exported. The native loader
rejects them. Tensor extraction is checked against ONNX initializer shapes.
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
    args=p.parse_args()
    if args.output.exists(): p.error('output exists; use a new filename')
    models=load_models(args.policy)
    values=[]
    mapping={}
    for key,model in models.items():
        initializers={t.name:numpy_helper.to_array(t) for t in model.graph.initializer}
        for layer,(nin,nout) in enumerate(zip(DIMS[key],DIMS[key][1:])):
            original=(f'module.encoders.g1.module.{layer*2}.weight' if key=='encoder' else f'onnx::MatMul_{136+layer}')
            bias=f'module.{"encoders.g1" if key=="encoder" else "decoders.g1_dyn"}.module.{layer*2}.bias'
            w=initializers[original] if key=='encoder' else initializers[original].T
            b=initializers[bias]
            if w.shape!=(nout,nin) or b.shape!=(nout,): raise ValueError('unexpected initializer shape')
            for suffix,array,name in [('weight',w,original),('bias',b,bias)]:
                if array.dtype!=np.float32 or not np.isfinite(array).all(): raise ValueError('invalid initializer')
                target=f'{key}.{layer}.{suffix}'
                values.append((target,np.ascontiguousarray(array)))
                mapping[target]={'onnx_name':name,'transpose':key=='decoder' and suffix=='weight'}
    meta=[kv_string('sonic.architecture','g1-mode0-mlp-fsq32-v1')]
    meta += [kv_string(f'sonic.{key}_sha256',value) for key,value in HASHES.items()]
    report=write_component(args.output,'sonic',values,HASHES['encoder'],general_name='NVIDIA GEAR-SONIC G1 mode 0',extra_metadata=meta)
    report.update(schema=1,source_onnx_sha256=HASHES,initializer_mapping=mapping,converter_sha256=sha256(Path(__file__)),
                  scope='G1 mode 0 only; encoder FSQ32 and decoder; original released model, not low-latency/v1.1')
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__': main()
