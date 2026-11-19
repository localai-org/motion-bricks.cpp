#!/usr/bin/env python3
"""Compare convolution bias-rounding boundaries using exact upstream operands.

FP32 products may still differ from CUDA reductions. This isolates whether an
additional half storage boundary before bias better explains captured Conv outputs.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper
from audit_soma_elementwise import digest
from audit_soma_matmul import product_errors


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    manifest=json.loads((args.capture/'manifest.json').read_text())
    pin=json.loads(Path(__file__).with_name('soma-reference.json').read_text())
    identity=next(a for a in pin['source_assets'] if a['path'].endswith('_fp16.onnx'))
    if (manifest['reference']!=pin or manifest['precision']!='mixed-fp16' or
            manifest['provider']!='CUDAExecutionProvider' or manifest['tapped_graph']!='source' or
            manifest['model']!=identity or digest(args.model)!=identity['sha256'] or
            digest(args.capture/'tensors.npz')!=manifest['capture_sha256']):
        raise ValueError('Expected verified official mixed-FP16 CUDA source capture')
    model=onnx.load(args.model);weights={t.name:t for t in model.graph.initializer};rows=[]
    with np.load(args.capture/'tensors.npz',allow_pickle=False) as capture:
        def operand(name):
            if name in weights:return numpy_helper.to_array(weights[name])
            return capture['graph/'+name]
        for node in model.graph.node:
            if node.op_type!='Conv':continue
            if len(node.input)!=3:raise ValueError('Expected convolution bias')
            a,k,b=map(operand,node.input);expected=operand(node.output[0])
            attrs={v.name:onnx.helper.get_attribute_value(v) for v in node.attribute}
            if (a.ndim!=3 or k.ndim!=3 or b.shape!=(k.shape[0],) or attrs.get('group',1)!=1 or
                    any(v.dtype!=np.float16 for v in (a,k,b,expected))):
                raise ValueError('Unsupported convolution layout/precision')
            dilation=attrs.get('dilations',[1])[0];stride=attrs.get('strides',[1])[0];pads=attrs.get('pads',[0,0])
            padded=np.pad(a,((0,0),(0,0),(pads[0],pads[1])))
            columns=np.stack([padded[:,:,j*dilation:j*dilation+expected.shape[-1]*stride:stride]
                              for j in range(k.shape[-1])],axis=-1)
            columns=columns.transpose(0,2,1,3).reshape(a.shape[0],expected.shape[-1],-1)
            product=(columns.astype(np.float32)@k.reshape(k.shape[0],-1).astype(np.float32).T).transpose(0,2,1)
            bias=b.astype(np.float32)[None,:,None]
            fused=(product+bias).astype(np.float16)
            separate=(product.astype(np.float16).astype(np.float32)+bias).astype(np.float16)
            rows.append(dict(node=node.name,values=int(expected.size),
                             fused_bias=product_errors(fused,expected),
                             separate_half_bias=product_errors(separate,expected)))
    if not rows:raise ValueError('No convolution nodes')
    summary={label:dict(mismatches=sum(row[label]['mismatches'] for row in rows),
                       max_absolute=max(row[label]['max_absolute'] for row in rows))
             for label in ('fused_bias','separate_half_bias')}
    report=dict(format='motionbricks-soma-isolated-conv-audit-v1',source_revision=pin['source_revision'],
                model_sha256=identity['sha256'],capture_sha256=manifest['capture_sha256'],
                runtime=manifest['runtime'],provider_options=manifest.get('provider_options'),
                nodes=rows,summary=summary,status='isolated diagnostic; does not establish native parity')
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
