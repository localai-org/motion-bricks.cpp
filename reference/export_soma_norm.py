#!/usr/bin/env python3
"""Export independent CUDA layer-normalization operands for native isolation."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import onnx
from onnx import numpy_helper
from audit_soma_elementwise import digest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from convert_to_gguf import write_component, kv_string


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
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
    tensors = []; names = []
    with np.load(args.capture / 'tensors.npz', allow_pickle=False) as capture:
        def operand(name):
            if name in weights: return numpy_helper.to_array(weights[name])
            return capture['graph/' + name]
        for node in model.graph.node:
            if node.op_type != 'LayerNormalization': continue
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            if attrs.get('axis', -1) != -1 or attrs.get('stash_type', 1) != 1:
                raise ValueError('Unsupported normalization contract')
            values = [operand(name) for name in node.input] + [operand(node.output[0])]
            if len(values) != 4 or any(v.dtype != np.float32 or not np.isfinite(v).all() for v in values):
                raise ValueError('Expected finite FP32 normalization operands')
            x, scale, bias, expected = values
            if x.shape != expected.shape or scale.shape != (x.shape[-1],) or bias.shape != scale.shape:
                raise ValueError('Unexpected normalization dimensions')
            stem = f'norm.{len(names)}.'; names.append(node.name)
            tensors.extend((stem + key, value) for key, value in zip(('input','scale','bias','expected'),values))
            tensors.append((stem + 'epsilon', np.array([attrs.get('epsilon',1e-5)],np.float32)))
    if not names: raise ValueError('No normalization nodes')
    write_component(args.output, 'soma-norm-fixture', tensors, manifest['capture_sha256'],
                    skeleton='soma30', upstream_revision=pin['source_revision'],
                    extra_metadata=[kv_string('fixture.norm_names',json.dumps(names)),
                                    kv_string('fixture.precision','mixed-fp16')])


if __name__ == '__main__': main()
