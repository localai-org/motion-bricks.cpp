#!/usr/bin/env python3
"""Independent same-state ONNX layer fixtures, including exact FSQ tokens.

Reference sessions explicitly disable graph optimization for inspectable
layer boundaries. Each captured decoder input is preserved, not replaced by
new encoder tokens. Full encoder->decoder composition is also recorded.
"""
import argparse
import copy
import json
from pathlib import Path
import struct
import numpy as np
import onnx
import onnxruntime as ort
from convert_sonic_gguf import load_models, HASHES, DIMS, sha256

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--trace',type=Path,action='append',default=[])
    p.add_argument('--mode',type=int,choices=(0,2),default=0)
    p.add_argument('--stride',type=int,default=10)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.stride<1 or args.output.exists(): p.error('invalid stride/output exists')
    if args.mode==0 and not args.trace: p.error('mode 0 requires a captured robot trace')
    sessions={}
    names={}
    options=ort.SessionOptions(); options.intra_op_num_threads=4; options.inter_op_num_threads=1
    options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    for key,model in load_models(args.policy).items():
        names[key]=[f'/{"smpl" if args.mode==2 else "g1"}/module/module.{2*i}/Gemm_output_0' if key=='encoder' else f'/g1_dyn/module/module.{2*i}/Add_output_0' for i in range(len(DIMS[key])-1)]
        names[key]+=['encoded_tokens' if key=='encoder' else 'action']
        traced=copy.deepcopy(model)
        del traced.graph.output[:]
        for name in names[key]: traced.graph.output.append(onnx.helper.make_tensor_value_info(name,onnx.TensorProto.FLOAT,None))
        sessions[key]=ort.InferenceSession(traced.SerializeToString(),sess_options=options,providers=['CPUExecutionProvider'])
    rows=[]
    upstream_errors=[]
    for path in args.trace:
        lines=path.read_text().splitlines()
        for index in sorted(set(range(0,len(lines),args.stride))|{len(lines)-1}):
            row=json.loads(lines[index])
            if row['encoder_input'][0]!=args.mode: raise ValueError('trace encoder mode mismatch')
            rows.append((row['encoder_input'],row['decoder_input'],row['tokens'],row['actions']))
    rng=np.random.default_rng(731)
    for i in range(64 if args.mode==2 else 16):
        enc=rng.normal(0,.25,1762).astype(np.float32); enc[:4]=0
        dec=rng.normal(0,.25,994).astype(np.float32)
        if i==0: enc[:]=0; dec[:]=0
        if args.mode==2:
            enc[0]=2
            # Half the synthetic cases are smooth, plausible human reference
            # windows; remaining cases exercise arbitrary observation values.
            # These test inference, not the SOMA adapter or robot tracking.
            if 0<i<32:
                joints=np.array([[0,0,0],[-.1,0,-.05],[.1,0,-.05],[0,0,.1],
                    [-.1,0,-.45],[.1,0,-.45],[0,0,.25],[-.1,0,-.85],
                    [.1,0,-.85],[0,0,.4],[-.1,.15,-.9],[.1,.15,-.9],
                    [0,0,.55],[-.08,0,.45],[.08,0,.45],[0,0,.7],
                    [-.2,0,.45],[.2,0,.45],[-.45,0,.3],[.45,0,.3],
                    [-.65,0,.15],[.65,0,.15],[-.7,0,.1],[.7,0,.1]],dtype=np.float32)
                window=np.repeat(joints[None],10,axis=0)
                phase=np.arange(10,dtype=np.float32)*.02+i*.1
                window[:,18:,2]+=np.sin(phase)[:,None]*.1
                enc[922:1642]=window.reshape(-1)
                theta=phase*.2
                rotation=np.stack([np.cos(theta),-np.sin(theta),np.sin(theta),np.cos(theta),theta*0,theta*0],axis=-1)
                enc[1642:1702]=rotation.reshape(-1)
                enc[1702:]=rng.normal(0,.1,60)
        rows.append((enc,dec,None,None))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('wb') as stream:
        stream.write(b'MBSONIC1'); stream.write(struct.pack('<I',len(rows)))
        for enc,dec,tokens,actions in rows:
            enc=np.array(enc,dtype=np.float32).reshape(1,-1); dec=np.array(dec,dtype=np.float32).reshape(1,-1)
            e=sessions['encoder'].run(names['encoder'],{'obs_dict':enc})
            d=sessions['decoder'].run(names['decoder'],{'obs_dict':dec})
            chained=dec.copy(); chained[:,:64]=e[-1]
            action=sessions['decoder'].run(['action'],{'obs_dict':chained})[0]
            for a in [enc,dec,*e,*d,action]: stream.write(np.asarray(a,dtype='<f4').tobytes())
            if tokens is not None:
                upstream_errors.append({'token_mismatches':int(np.sum(e[-1].reshape(-1)!=tokens)),
                                        'action_max_abs_error':float(np.max(np.abs(d[-1].reshape(-1)-actions)))})
    report=dict(schema=1,source_onnx_sha256=HASHES,fixture_sha256=sha256(args.output),
                encoder_mode=args.mode, synthetic_samples=64 if args.mode==2 else 16,
                capture_sha256=sha256(Path(__file__)),onnxruntime=ort.__version__,samples=len(rows),
                traces={str(path):sha256(path) for path in args.trace},layer_names=names,
                vs_captured_tensorrt={'samples_with_different_tokens':sum(r['token_mismatches']>0 for r in upstream_errors),
                    'token_mismatches':sum(r['token_mismatches'] for r in upstream_errors),
                    'decoder_max_abs_error':max((r['action_max_abs_error'] for r in upstream_errors),default=0)})
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
