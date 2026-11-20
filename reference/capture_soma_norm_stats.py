#!/usr/bin/env python3
"""Capture CUDA LayerNorm statistics, requiring bitwise output equality to SOMA."""
import argparse
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np,onnx,onnxruntime as ort
from onnx import helper,numpy_helper,TensorProto
from capture_soma_onnx import runtime_identity
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model',type=Path,required=True)
parser.add_argument('--capture',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();model_path=args.model;capture=args.capture
pin=json.loads(Path(__file__).with_name('soma-reference.json').read_text());m=json.loads((capture/'manifest.json').read_text())
identity=next(a for a in pin['source_assets'] if a['path'].endswith('_fp16.onnx'))
if (m['reference']!=pin or m['provider']!='CUDAExecutionProvider' or m['precision']!='mixed-fp16' or
    m['tapped_graph']!='source' or m['model']!=identity or
    hashlib.sha256(model_path.read_bytes()).hexdigest()!=identity['sha256'] or
    hashlib.sha256((capture/'tensors.npz').read_bytes()).hexdigest()!=m['capture_sha256']):
 raise ValueError('Requires verified official CUDA mixed-FP16 source capture')
model=onnx.load(model_path);weights={t.name:numpy_helper.to_array(t) for t in model.graph.initializer};c=np.load(capture/'tensors.npz')
def value(name):return weights[name] if name in weights else c['graph/'+name]
ort.preload_dlls();opts=ort.SessionOptions();opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL;opts.intra_op_num_threads=1
arrays={};names=[]
for n in model.graph.node:
 if n.op_type!='LayerNormalization':continue
 vals=[value(name) for name in n.input];expected=value(n.output[0]);x,s,b=vals
 if any(v.dtype!=np.float32 or not np.isfinite(v).all() for v in [*vals,expected]):
  raise ValueError('Expected finite FP32 normalization operands')
 attrs={a.name:helper.get_attribute_value(a) for a in n.attribute}
 inputs=[helper.make_tensor_value_info(key,TensorProto.FLOAT,v.shape) for key,v in zip(['x','s','b'],vals)]
 outputs=[helper.make_tensor_value_info('y',TensorProto.FLOAT,x.shape)]+[helper.make_tensor_value_info(key,TensorProto.FLOAT,x.shape[:-1]+(1,)) for key in ['mean','invstd']]
 graph=helper.make_graph([helper.make_node('LayerNormalization',['x','s','b'],['y','mean','invstd'],**attrs)],'isolatednorm',inputs,outputs)
 isolated=helper.make_model(graph,opset_imports=model.opset_import);isolated.ir_version=model.ir_version
 sess=ort.InferenceSession(isolated.SerializeToString(),opts,providers=['CUDAExecutionProvider'])
 if sess.get_providers()[0]!='CUDAExecutionProvider':raise ValueError('CUDA provider unavailable')
 y,mean,invstd=sess.run(None,dict(zip(['x','s','b'],vals)))
 if not np.array_equal(y,expected):raise ValueError('Isolated output differs: '+n.name)
 stem=f'norm.{len(names)}.';names.append(n.name)
 for key,v in zip(['input','scale','bias','expected','mean','invstd'],[x,s,b,y,mean,invstd]):arrays[stem+key]=v
 arrays[stem+'epsilon']=np.array(attrs.get('epsilon',1e-5),np.float32)
if not names:raise ValueError('No normalization nodes')
out=args.output;out.mkdir(parents=True,exist_ok=False)
np.savez(out/'tensors.npz',**arrays)
manifest=dict(format='motionbricks-soma-norm-statistics-v1',status='isolated CUDA LayerNorm stats; all outputs bitwise equal to pinned full-graph capture',reference=pin,model=identity,capture_sha256=m['capture_sha256'],names=names,runtime_identity=runtime_identity('CUDAExecutionProvider',sess.get_provider_options()),stats_sha256=hashlib.sha256((out/'tensors.npz').read_bytes()).hexdigest())
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Captured',len(names),'norms; all outputs bitwise equal')
