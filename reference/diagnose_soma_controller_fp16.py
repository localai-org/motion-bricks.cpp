#!/usr/bin/env python3
"""Summarize the two traced FP16 controller failures, with public API equivalence."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import onnx
from compare_native_taps import read_gguf
from check_soma_api_chain import SomaAPI,digest
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True)
p.add_argument('--library',type=Path,required=True)
p.add_argument('--bundle',type=Path,required=True)
p.add_argument('--onnx',type=Path,required=True)
a=p.parse_args()
graph=onnx.load(a.onnx)
api=SomaAPI(a.library,a.bundle,'vulkan')
results=[]
try:
 for event,projection in [(7,'local'),(79,'global')]:
  root=a.root/f'event-{event:03d}'
  fixture=a.root/'fixture-007/input.gguf' if event==7 else root/'input.gguf'
  fm,f=read_gguf(fixture)
  if event==7:api.set_skeleton(f)
  meta,native=read_gguf(root/'native.gguf');tm,taps=read_gguf(root/'rounding.gguf.layers.gguf')
  upstream=np.load(root/'upstream/tensors.npz')
  public=api.generate(f['context'],f['target'],3,fm['fixture.allowed_token_bits'],fm['fixture.heading_angle'])
  public_equal=all(np.array_equal(public[k].reshape(-1),native[v].reshape(-1)) for k,v in [('corrected','corrected'),('world','world'),('features','output_features'),('contacts','contacts')])
  stem='_proj_'+projection+'_root_value'
  node=next(n for n in graph.graph.node if n.name=='/mmm_net/'+stem+'/MatMul')
  weight=onnx.numpy_helper.to_array(next(t for t in graph.graph.initializer if t.name==node.input[1]))
  input_values=upstream['graph/'+node.input[0]]
  expected=upstream['graph/'+node.output[0]]
  def tap(suffix):
   key=next(k[:-5] for k,v in tm.items() if k.endswith('.name') and str(v).endswith('/'+stem+suffix))
   return taps[key].reshape(expected.shape)
  full,half=tap('.matmul_f32'),tap('.matmul_half')
  differences=[]
  for index in np.argwhere(half!=expected):
   ix=tuple(index);products=input_values[ix[:2]].astype(np.float64)*weight[:,ix[-1]].astype(np.float64)
   differences.append(dict(index=index.tolist(),exact_dot=float(products.sum()),native_fp32_sum=float(full[ix]),native_fp16=float(half[ix]),upstream_fp16=float(expected[ix]),halfway=float((np.float64(half[ix])+np.float64(expected[ix]))/2)))
  count=meta['soma.tokens'];codes=native['pose_logits'].reshape(count,5,16).argmax(-1);refcodes=upstream['graph//ArgMax_output_0'][0,:count]
  code_diffs=[dict(index=i.tolist(),native=int(codes[tuple(i)]),upstream=int(refcodes[tuple(i)])) for i in np.argwhere(codes!=refcodes)]
  encoded={k:bool(np.array_equal(native['input.'+k].reshape(upstream['input/'+k].shape),upstream['input/'+k])) for k in ('global_root_values','local_root_values','local_poses')}
  normalized=bool(np.array_equal(native['normalized_'+projection].reshape(input_values.shape).astype(np.float16),input_values))
  results.append(dict(event=event,public_api_matches_runner=public_equal,encoded_inputs_exact=encoded,normalized_projection_inputs_exact=normalized,first_divergent_projection=stem,projection_differences=differences,pose_code_changes=code_diffs,duration_matches=int(upstream['output/pred_num_tokens'].item())==count,upstream_taps_sha256=digest(root/'upstream/tensors.npz'),native_taps_sha256=digest(root/'rounding.gguf.layers.gguf')))
finally:api.close()
_,feedback=read_gguf(a.root/'feedback-001/native.gguf');inputs=np.load(a.root/'feedback-001/inputs.npz')
feedback_differences={}
for name in ('global_root_values','local_root_values','local_poses'):
 x=feedback['input.'+name].reshape(inputs[name].shape);y=inputs[name]
 feedback_differences[name]=dict(max_absolute=float(np.max(abs(x-y))),raw_fp16_unequal=int(np.sum(x.astype(np.float16)!=y.astype(np.float16))))
report=dict(format='soma-controller-fp16-divergence-v1',library_sha256=digest(a.library),onnx_sha256=digest(a.onnx),events=results,feedback_second_generation_encoded_differences=feedback_differences,feedback_note='Raw input FP16 differences are a sensitivity diagnostic; the model also normalizes and retargets these inputs before its FP16 projections. This is not proof of the first differing normalized model value in feedback.')
(a.root/'diagnosis.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
if not all(x['public_api_matches_runner'] and all(x['encoded_inputs_exact'].values()) and x['normalized_projection_inputs_exact'] for x in results):raise SystemExit('Trace equivalence failed')
