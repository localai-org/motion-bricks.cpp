#!/usr/bin/env python3
"""Check isolated mixed-FP16 elementwise rules against independent CUDA taps.

Each operation receives its own captured inputs: this rules out propagated error
for the tested operation, but cannot establish native or full-model parity.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


OPERATIONS = {'Add', 'Sub', 'Mul', 'Div', 'Sqrt', 'Relu', 'LeakyRelu', 'Where'}


def evaluate(node, operands):
    values = [value.astype(np.float32) for value in operands]
    kind = node.op_type
    if kind == 'Add': result = values[0] + values[1]
    elif kind == 'Sub': result = values[0] - values[1]
    elif kind == 'Mul': result = values[0] * values[1]
    elif kind == 'Div': result = values[0] / values[1]
    elif kind == 'Sqrt': result = np.sqrt(values[0])
    elif kind == 'Relu': result = np.maximum(values[0], 0)
    elif kind == 'Where': result = np.where(operands[0], values[1], values[2])
    elif kind == 'LeakyRelu':
        alpha = next((attr.f for attr in node.attribute if attr.name == 'alpha'), .01)
        # CUDA's half LeakyRelu kernel converts the coefficient to its operand type.
        result = np.where(values[0] >= 0, values[0], values[0] * float(np.float16(alpha)))
    else: raise ValueError('Unsupported operation: ' + kind)
    return result.astype(np.float16)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.capture / 'manifest.json').read_text())
    pin = json.loads(Path(__file__).with_name('soma-reference.json').read_text())
    if (manifest['reference'] != pin or manifest['precision'] != 'mixed-fp16' or
            manifest['provider'] != 'CUDAExecutionProvider' or manifest['tapped_graph'] != 'source'):
        raise ValueError('Requires pinned CUDA source-graph mixed-FP16 capture')
    identity = next(asset for asset in pin['source_assets'] if asset['path'].endswith('_fp16.onnx'))
    if manifest['model'] != identity or digest(args.model) != identity['sha256']:
        raise ValueError('Official mixed-FP16 model checksum mismatch')
    if digest(args.capture / 'tensors.npz') != manifest['capture_sha256']:
        raise ValueError('Capture checksum mismatch')
    model = onnx.load(args.model)
    weights = {tensor.name: tensor for tensor in model.graph.initializer}
    rows = []
    with np.load(args.capture / 'tensors.npz', allow_pickle=False) as capture:
        def operand(name):
            if name in weights: return numpy_helper.to_array(weights[name])
            if 'graph/' + name in capture: return capture['graph/' + name]
            return capture['input/' + name]
        for node in model.graph.node:
            if node.op_type not in OPERATIONS: continue
            expected = operand(node.output[0])
            if expected.dtype != np.float16: continue
            inputs = [operand(name) for name in node.input]
            actual = evaluate(node, inputs)
            if (actual.shape != expected.shape or not np.isfinite(actual).all() or
                    not np.isfinite(expected).all()):
                raise ValueError('Nonfinite or mismatched output: ' + node.name)
            delta = actual.astype(np.float64) - expected.astype(np.float64)
            rows.append(dict(node=node.name, operation=node.op_type, inputs=list(node.input),
                             input_dtypes=[str(value.dtype) for value in inputs],
                             output=node.output[0], values=int(expected.size),
                             mismatches=int(np.count_nonzero(actual != expected)),
                             max_absolute=float(np.abs(delta).max())))
    if not rows: raise ValueError('No captured FP16 elementwise operations')
    summary = dict(nodes=len(rows), values=sum(row['values'] for row in rows),
                   nodes_with_mismatches=sum(row['mismatches'] > 0 for row in rows),
                   mismatches=sum(row['mismatches'] for row in rows),
                   max_absolute=max(row['max_absolute'] for row in rows))
    report = dict(format='motionbricks-soma-isolated-elementwise-audit-v1',
                  source_revision=pin['source_revision'], model_sha256=identity['sha256'],
                  capture_sha256=manifest['capture_sha256'], runtime=manifest['runtime'],
                  provider_options=manifest.get('provider_options'),
                  cuda_gemm_options=manifest.get('cuda_gemm_options', 'unrecorded'),
                  numpy_version=np.__version__, operations=sorted(OPERATIONS),
                  summary=summary, nodes=rows,
                  status='isolated arithmetic diagnostic; does not establish native parity')
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': main()
