#!/usr/bin/env python3
"""Export exact CUDA operands for an isolated native MatMul/Gemm/Conv probe."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import onnx
from onnx import numpy_helper
from audit_soma_matmul import digest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from convert_to_gguf import write_component, kv_string


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--node', default='/mmm_net/_proj_local_pose/MatMul')
    parser.add_argument('--repeat-rows',type=int,default=1,
                        help='Repeat captured rows to probe kernel shape effects; reference output is repeated, not recaptured')
    parser.add_argument('--flatten-rows',action='store_true',help='Flatten captured leading input dimensions for a rank-two weight matrix; kernel-shape diagnostic only')
    parser.add_argument('--take-rows',type=int,default=0,help='Take a prefix of captured matrix rows for kernel-shape diagnostics; not a new upstream capture')
    args = parser.parse_args()
    if args.take_rows<0:parser.error('take-rows must be nonnegative')
    if not 1<=args.repeat_rows<=64:parser.error('repeat-rows must be between 1 and 64')
    manifest = json.loads((args.capture / 'manifest.json').read_text())
    pin = json.loads(Path(__file__).with_name('soma-reference.json').read_text())
    identity = next(a for a in pin['source_assets'] if a['path'].endswith('_fp16.onnx'))
    if (manifest['reference'] != pin or manifest['precision'] != 'mixed-fp16' or
            manifest['provider'] != 'CUDAExecutionProvider' or manifest['tapped_graph'] != 'source' or
            manifest['model'] != identity or digest(args.model) != identity['sha256'] or
            digest(args.capture / 'tensors.npz') != manifest['capture_sha256']):
        raise ValueError('Expected verified official mixed-FP16 CUDA source capture')
    model = onnx.load(args.model)
    weights = {t.name: t for t in model.graph.initializer}
    nodes = [node for node in model.graph.node if node.name == args.node and node.op_type in ('MatMul','Gemm','Conv')]
    if len(nodes) != 1: raise ValueError('Expected one named MatMul, Gemm or Conv')
    node = nodes[0]
    with np.load(args.capture / 'tensors.npz', allow_pickle=False) as capture:
        def operand(name):
            if name in weights: return numpy_helper.to_array(weights[name])
            return capture['graph/' + name]
        a, b = [operand(name) for name in node.input[:2]]; expected = operand(node.output[0])
        bias=None
        conv_metadata=[]
        if node.op_type=='Conv':
            attrs={x.name:onnx.helper.get_attribute_value(x) for x in node.attribute}
            if a.ndim!=3 or b.ndim!=3 or a.shape[0]!=1 or attrs.get('group',1)!=1 or len(node.input)!=3:
                raise ValueError('Conv fixture requires one batch and group, rank-three operands and bias')
            channels,spatial=b.shape[1:]
            if a.shape[1]!=channels or expected.ndim!=3 or expected.shape[:2]!=(1,b.shape[0]):
                raise ValueError('Invalid Conv input/output shape')
            stride=attrs.get('strides',[1])[0];dilation=attrs.get('dilations',[1])[0];pads=attrs.get('pads',[0,0])
            if stride<1 or dilation<1 or len(pads)!=2 or min(pads)<0:raise ValueError('Invalid Conv attributes')
            bias=operand(node.input[2])
            padded=np.pad(a[0],((0,0),(pads[0],pads[1])))
            columns=np.stack([padded[:,i*dilation:i*dilation+expected.shape[-1]*stride:stride]
                              for i in range(spatial)],axis=-1)
            a=columns.transpose(1,0,2).reshape(expected.shape[-1],channels*spatial)
            b=b.reshape(b.shape[0],channels*spatial).T
            expected=expected[0].T
            conv_metadata=[kv_string('fixture.conv_'+k,str(v)) for k,v in
                           [('spatial',spatial),('channels',channels),('dilation',dilation)]]
            if bias.dtype!=np.float16 or bias.shape!=(b.shape[-1],) or not np.isfinite(bias).all():
                raise ValueError('Expected finite half Conv bias')
        if node.op_type=='Gemm':
            attrs={x.name:onnx.helper.get_attribute_value(x) for x in node.attribute}
            if attrs.get('alpha',1)!=1 or attrs.get('beta',1)!=1 or len(node.input)!=3:
                raise ValueError('Probe supports Gemm with unit alpha/beta and explicit bias')
            if attrs.get('transA',0):a=a.T
            if attrs.get('transB',0):b=b.T
            bias=operand(node.input[2])
            if bias.dtype!=np.float16 or bias.shape!=(b.shape[-1],) or not np.isfinite(bias).all():
                raise ValueError('Expected finite half vector bias')
        if any(x.dtype != np.float16 or not np.isfinite(x).all() for x in (a,b,expected)):
            raise ValueError('Expected finite half operands and output')
        if not (2<=a.ndim<=4 and 2<=b.ndim<=4) or a.shape[-1]!=b.shape[-2]:
            raise ValueError('Probe requires rank 2–4 matrix operands with matching inner dimensions')
        shape=np.broadcast_shapes(a.shape[:-2],b.shape[:-2])+(a.shape[-2],b.shape[-1])
        if expected.shape!=shape:raise ValueError('Unexpected output shape')
        if a.ndim==3 and a.shape[0]==1 and b.ndim==2:
            a=a[0];expected=expected[0]  # Preserve existing projection fixtures.
        if np.broadcast_shapes(a.shape[:-2],b.shape[:-2])!=a.shape[:-2]:
            raise ValueError('GGML probe requires left operand to carry the broadcast batch dimensions')
        if args.flatten_rows:
            if b.ndim!=2:raise ValueError('Flattening rows requires rank-two weights')
            a=a.reshape(-1,a.shape[-1]);expected=expected.reshape(-1,expected.shape[-1])
        # Diagnostic only: higher precision is not the CUDA acceptance target.
        high_precision = a.astype(np.float64) @ b.astype(np.float64)
        if node.op_type=='Conv':high_precision=high_precision.astype(np.float16).astype(np.float64)
        if bias is not None:high_precision+=bias.astype(np.float64)
        high_precision=high_precision.astype(np.float16)
        tensors = [('input',a.astype(np.float32)),('weight',b.swapaxes(-1,-2).copy().astype(np.float32)),
                   ('expected',expected.astype(np.float32)),
                   ('fp64_rounded_diagnostic',high_precision.astype(np.float32))]
        if bias is not None:tensors.append(('bias',bias.astype(np.float32)))
        if args.take_rows:
            if args.take_rows>a.shape[-2]:raise ValueError('Requested row count exceeds captured rows')
            tensors=[(name,value if name in ('weight','bias') else value[..., :args.take_rows, :]) for name,value in tensors]
        if args.repeat_rows!=1:
            tensors=[(name,value if name in ('weight','bias') else np.concatenate([value]*args.repeat_rows,axis=-2)) for name,value in tensors]
    write_component(args.output, 'soma-matmul-fixture', tensors, manifest['capture_sha256'],
                    skeleton='soma30', upstream_revision=pin['source_revision'],
                    extra_metadata=[kv_string('fixture.node',node.name),
                                    kv_string('fixture.repeat_rows',str(args.repeat_rows)),
                                    kv_string('fixture.take_rows',str(args.take_rows)),
                                    kv_string('fixture.flatten_rows',str(args.flatten_rows).lower()),
                                    kv_string('fixture.precision','mixed-fp16')]+conv_metadata)


if __name__ == '__main__': main()
