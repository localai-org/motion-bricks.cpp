#!/usr/bin/env python3
"""Audit isolated FP16 MatMul arithmetic using captured upstream operands.

No errors are propagated between operations: every product uses the independently
captured CUDA inputs and official initializer values. FP64 is a high-precision
diagnostic, not an assertion about CUDA's required reduction order. This audit
cannot establish native model parity or justify discrete-decision mismatches.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def product_errors(actual, expected):
    if actual.shape!=expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('Invalid isolated matrix-product output')
    delta=np.abs(actual.astype(np.float64)-expected.astype(np.float64))
    # Measure output differences in local half steps, including subnormals.
    spacing=np.abs(np.spacing(expected.astype(np.float16)).astype(np.float64))
    return dict(mismatches=int(np.count_nonzero(actual!=expected)),
                max_absolute=float(delta.max()),rms=float(np.sqrt(np.mean(delta**2))),
                max_local_half_steps=float(np.max(delta/spacing)))


def block_round_toward_zero(a, b, block):
    """Hypothesis probe, not an implementation of NVIDIA tensor-core arithmetic."""
    result = np.zeros(np.broadcast_shapes(a.shape[:-2], b.shape[:-2]) +
                      (a.shape[-2], b.shape[-1]), dtype=np.float32)
    for offset in range(0, a.shape[-1], block):
        partial = (a[..., offset:offset+block].astype(np.float64) @
                   b[..., offset:offset+block, :].astype(np.float64))
        exact = result.astype(np.float64) + partial
        rounded = exact.astype(np.float32)
        overshoot = np.abs(rounded.astype(np.float64)) > np.abs(exact)
        result = np.where(overshoot, np.nextafter(rounded, np.float32(0)), rounded)
    return result.astype(np.float16)


def aligned_block_product(a, b, block, fractional_bits, *, return_fp32=False, residue_first=False):
    """Finite-half diagnostic inspired by https://arxiv.org/abs/2512.07004.

    Products retain the sum of operand exponents (no product normalization).
    Align products and the prior accumulator to the largest exponent, truncate
    each aligned significand, sum exactly, then truncate to FP32. Contiguous
    reduction blocks are a hypothesis, not a cuBLAS kernel-order guarantee.
    """
    if block <= 0 or not 23 <= fractional_bits <= 27:
        raise ValueError('Invalid alignment hypothesis')
    if any(x.dtype != np.float16 or not np.isfinite(x).all() for x in (a,b)):
        raise ValueError('Alignment probe requires finite half operands')
    a,b=a.astype(np.float64),b.astype(np.float64)
    result=np.zeros(np.broadcast_shapes(a.shape[:-2],b.shape[:-2])+
                    (a.shape[-2],b.shape[-1]),dtype=np.float64)
    remainder=a.shape[-1]%block if residue_first else 0
    spans=([(0,remainder)] if remainder else [])+[(offset,min(offset+block,a.shape[-1])) for offset in range(remainder,a.shape[-1],block)]
    for offset,end in spans:
        left=a[...,offset:end,None]
        right=b[...,None,offset:end,:]
        products=left*right
        # Half subnormal operands are normalized by frexp; zero products do
        # not set the alignment exponent. No FP32 underflow is possible here.
        exponents=(np.frexp(left)[1]-1)+(np.frexp(right)[1]-1)
        exponents=np.where(products!=0,exponents,-1024)
        largest=np.maximum(exponents.max(axis=-2),
                           np.where(result!=0,np.frexp(result)[1]-1,-1024))
        largest=np.maximum(largest,-126)
        quantum=np.ldexp(np.ones_like(result),largest-fractional_bits)
        exact=(np.trunc(products/quantum[...,None,:]).sum(axis=-2)+
               np.trunc(result/quantum))*quantum
        rounded=exact.astype(np.float32)
        overshoot=np.abs(rounded.astype(np.float64))>np.abs(exact)
        result=np.where(overshoot,np.nextafter(rounded,np.float32(0)),rounded).astype(np.float64)
    return result.astype(np.float32 if return_fp32 else np.float16)


def split_half_product(a, b, split):
    """Hypothesis: independent 16/25 products, half partials, FP32 reduction."""
    if split <= 0 or split % 16:
        raise ValueError('Split width must be a positive multiple of 16')
    result=None
    for offset in range(0,a.shape[-1],split):
        partial=aligned_block_product(a[...,offset:offset+split],
                                      b[...,offset:offset+split,:],16,25).astype(np.float32)
        result=partial if result is None else result+partial
    if result is None:raise ValueError('Empty reduction')
    return result.astype(np.float16)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--node-prefix',default='',help='Limit the diagnostic to named source nodes')
    parser.add_argument('--rounding-blocks',type=int,nargs='+',default=[],
                        help='Probe partial-sum rounding toward zero; not a CUDA contract')
    parser.add_argument('--alignment-blocks',type=int,nargs='+',default=[],
                        help='Probe per-product alignment truncation with 24/25 fractional bits')
    parser.add_argument('--split-k-blocks',type=int,nargs='+',default=[],
                        help='Probe half partial products and FP32 reduction using aligned 16/25 arithmetic')
    parser.add_argument('--residue-first',action='store_true',
                        help='For alignment probes, process the short K remainder before full blocks')
    args=parser.parse_args()
    if any(value<=0 for value in args.rounding_blocks+args.alignment_blocks):parser.error('Block sizes must be positive')
    if any(value<=0 or value%16 for value in args.split_k_blocks):parser.error('Split widths must be positive multiples of 16')
    manifest=json.loads((args.capture/'manifest.json').read_text())
    pinned=json.loads(Path(__file__).with_name('soma-reference.json').read_text())
    if (manifest['reference']!=pinned or manifest['precision']!='mixed-fp16' or
            manifest['provider']!='CUDAExecutionProvider' or manifest['tapped_graph']!='source'):
        raise ValueError('Requires pinned CUDA source-graph mixed-FP16 capture')
    identity=next(x for x in pinned['source_assets'] if x['path'].endswith('_fp16.onnx'))
    if manifest['model']!=identity or digest(args.model)!=identity['sha256']:
        raise ValueError('Official mixed-FP16 graph checksum mismatch')
    if digest(args.capture/'tensors.npz')!=manifest['capture_sha256']:
        raise ValueError('Capture checksum mismatch')
    model=onnx.load(args.model)
    weights={tensor.name:tensor for tensor in model.graph.initializer}
    rows=[]
    with np.load(args.capture/'tensors.npz',allow_pickle=False) as capture:
        def operand(name):
            if name in weights:return numpy_helper.to_array(weights[name])
            if 'graph/'+name in capture:return capture['graph/'+name]
            return capture['input/'+name]
        for node in model.graph.node:
            if node.op_type!='MatMul' or not node.name.startswith(args.node_prefix):continue
            a,b=map(operand,node.input);expected=operand(node.output[0])
            if expected.dtype!=np.float16:continue
            if a.dtype!=np.float16 or b.dtype!=np.float16:
                raise ValueError(f'Unexpected operand precision: {node.name}')
            row=dict(node=node.name,inputs=list(node.input),output=node.output[0],
                     left_shape=list(a.shape),right_shape=list(b.shape),values=int(expected.size))
            for label,dtype in [('fp32_accumulation',np.float32),('fp64_diagnostic',np.float64)]:
                actual=(a.astype(dtype)@b.astype(dtype)).astype(np.float16)
                row[label]=product_errors(actual,expected)
            for block in args.rounding_blocks:
                row[f'block_{block}_toward_zero_hypothesis']=product_errors(block_round_toward_zero(a,b,block),expected)
            for block in args.alignment_blocks:
                for bits in (24,25):
                    row[f'aligned_{block}_{bits}_hypothesis']=product_errors(aligned_block_product(a,b,block,bits,residue_first=args.residue_first),expected)
            for split in args.split_k_blocks:
                row[f'split_half_{split}_hypothesis']=product_errors(split_half_product(a,b,split),expected)
            rows.append(row)
    if not rows:raise ValueError('No captured FP16 matrix products')
    report=dict(format='motionbricks-soma-isolated-matmul-audit-v1',
                source_revision=pinned['source_revision'],model_sha256=identity['sha256'],
                capture_sha256=manifest['capture_sha256'],runtime=manifest['runtime'],
                runtime_identity=manifest.get('runtime_identity'),
                provider_options=manifest.get('provider_options'),
                cuda_gemm_options=manifest.get('cuda_gemm_options','unrecorded'),
                node_prefix=args.node_prefix,rounding_blocks=args.rounding_blocks,
                alignment_blocks=args.alignment_blocks,
                residue_first=args.residue_first,
                split_k_blocks=args.split_k_blocks,
                alignment_model_source='https://arxiv.org/html/2512.07004v3',
                alignment_model_scope='finite half operands; contiguous blocks; product alignment and FP32 truncation hypothesis; not verified RTX hardware emulation',
                numpy_version=np.__version__,nodes=rows,
                summary={label:dict(nodes_with_mismatches=sum(row[label]['mismatches']>0 for row in rows),
                                   mismatches=sum(row[label]['mismatches'] for row in rows),
                                   values=sum(row['values'] for row in rows),
                                   max_absolute=max(row[label]['max_absolute'] for row in rows))
                         for label in ('fp32_accumulation','fp64_diagnostic',
                                       *(f'block_{block}_toward_zero_hypothesis' for block in args.rounding_blocks),
                                       *(f'aligned_{block}_{bits}_hypothesis' for block in args.alignment_blocks for bits in (24,25)),
                                       *(f'split_half_{split}_hypothesis' for split in args.split_k_blocks))},
                status='isolated arithmetic diagnostic; does not establish native parity')
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['summary'],indent=2))


if __name__=='__main__':main()
