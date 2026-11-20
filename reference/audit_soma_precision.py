#!/usr/bin/env python3
"""Inventory declared and executed precision boundaries for the pinned SOMA export.

This audit is the implementation contract for mixed-precision work. It does not
label FP16 weight storage, or a promoted CPU arithmetic operation, as FP16 compute.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import onnx
from onnx import TensorProto
from fetch_soma import verify


def audit(path):
    graph=onnx.shape_inference.infer_shapes(onnx.load(path),strict_mode=True).graph
    types={x.name:x.type.tensor_type.elem_type for x in [*graph.input,*graph.output,*graph.value_info]}
    types.update({x.name:x.data_type for x in graph.initializer})
    nodes=[];casts=[];counts=Counter()
    for node in graph.node:
        outputs=[dict(name=name,dtype=TensorProto.DataType.Name(types[name])) for name in node.output if name]
        inputs=[dict(name=name,dtype=TensorProto.DataType.Name(types[name]) if name in types else 'unknown') for name in node.input if name]
        nodes.append(dict(name=node.name,op=node.op_type,inputs=inputs,outputs=outputs))
        for value in outputs:counts[(node.op_type,value['dtype'])]+=1
        if node.op_type=='Cast' and inputs and outputs and inputs[0]['dtype']!=outputs[0]['dtype']:
            casts.append(nodes[-1])
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(sha256=digest,nodes=nodes,casts=casts,
                operator_output_types=[dict(op=op,dtype=dtype,count=count)for (op,dtype),count in sorted(counts.items())])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--capture',type=Path,required=True,help='--tap-executed-graph capture directory')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    pin=json.loads(Path(__file__).with_name('soma-reference.json').read_text())
    identity=next(x for x in pin['source_assets'] if x['path'].endswith('_fp16.onnx'))
    if not verify(args.model,identity):raise ValueError('Not the pinned mixed-FP16 model')
    manifest=json.loads((args.capture/'manifest.json').read_text())
    if manifest.get('tapped_graph')!='runtime-lowered' or manifest['model']!=identity:
        raise ValueError('Capture does not use this model with executed-graph taps')
    if any(manifest['instrumentation_max_absolute_error'].values()):raise ValueError('Instrumentation changed execution')
    source=audit(args.model);executed=audit(args.capture/'runtime-lowered.onnx')
    if executed['sha256']!=manifest['lowered_graph_sha256']:raise ValueError('Lowered graph hash mismatch')
    report=dict(format='motionbricks-soma-precision-audit-v1',source_revision=pin['source_revision'],
                provider=manifest['provider'],runtime=manifest['runtime'],source=source,executed=executed)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    for label,data in [('source',source),('executed',executed)]:
        compute=[row for row in data['operator_output_types'] if row['op'] in ('MatMul','Gemm','Conv','LayerNormalization','Softmax')]
        print(label,compute)


if __name__=='__main__':main()
