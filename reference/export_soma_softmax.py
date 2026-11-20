#!/usr/bin/env python3
"""Export all pinned CUDA softmax inputs/outputs for isolated backend testing."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import numpy as np
import onnx


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    pin = json.loads(Path(__file__).with_name('soma-reference.json').read_text())
    manifest = json.loads((args.capture / 'manifest.json').read_text())
    identity = next(x for x in pin['source_assets'] if x['path'].endswith('_fp16.onnx'))
    if (manifest['reference'] != pin or manifest['precision'] != 'mixed-fp16' or
            manifest['provider'] != 'CUDAExecutionProvider' or manifest['tapped_graph'] != 'source' or
            manifest['model'] != identity or digest(args.model) != identity['sha256'] or
            digest(args.capture / 'tensors.npz') != manifest['capture_sha256']):
        raise ValueError('Requires a verified pinned mixed-FP16 CUDA source capture')
    model = onnx.load(args.model)
    nodes = [n for n in model.graph.node if n.op_type == 'Softmax']
    if len(nodes) != 24:
        raise ValueError('Pinned softmax node inventory changed')
    args.output.mkdir(parents=True, exist_ok=False)
    rows, values = [], 0
    path = args.output / 'softmax-reference.bin'
    with np.load(args.capture / 'tensors.npz') as capture, path.open('wb') as stream:
        stream.write(struct.pack('<I', len(nodes)))
        for node in nodes:
            axis = next((a.i for a in node.attribute if a.name == 'axis'), -1)
            x = capture['graph/' + node.input[0]]
            y = capture['graph/' + node.output[0]]
            if (x.dtype != np.float32 or y.dtype != np.float32 or x.shape != y.shape or
                    x.ndim < 2 or axis not in (-1, x.ndim - 1) or not np.isfinite(y).all() or
                    np.isnan(x).any() or np.isposinf(x).any()):
                raise ValueError('Unexpected pinned softmax input/output contract')
            width = x.shape[-1]
            count = x.size // width
            if not 1 <= width <= 1024 or not 1 <= count <= 4096:
                raise ValueError('Softmax shape exceeds probe limits')
            stream.write(struct.pack('<II', width, count))
            stream.write(np.ascontiguousarray(x, dtype='<f4').tobytes())
            stream.write(np.ascontiguousarray(y, dtype='<f4').tobytes())
            rows.append(dict(name=node.name, width=width, rows=count, values=x.size))
            values += x.size
    if values != 93312:
        raise ValueError('Pinned softmax value inventory changed')
    result = dict(format='motionbricks-softmax-fixture-v1', fixture_sha256=digest(path),
                  source_capture=str(args.capture), source_capture_sha256=manifest['capture_sha256'],
                  source_revision=pin['source_revision'], model_sha256=identity['sha256'],
                  values=values, nodes=len(nodes), layers=rows,
                  layout='little-endian uint32 node count; per node uint32 width, rows, contiguous float32 input then expected output')
    (args.output / 'softmax-reference-manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'Exported {len(nodes)} softmax nodes, {values} values')


if __name__ == '__main__':
    main()
