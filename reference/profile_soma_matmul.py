#!/usr/bin/env python3
"""Identify isolated CUDA MatMul, Gemm, or Conv kernels, requiring exact pinned capture outputs.

Requires the GPU reference environment and an explicit CUPTI library path. This
records kernel names, not performance counters, and never changes the parity
reference or claims that an isolated kernel proves full-model parity.
"""
import argparse
import ctypes as C
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper
from audit_soma_matmul import digest
from capture_soma_onnx import runtime_identity


class CallbackData(C.Structure):
    # CUpti_CallbackData, cupti_callbacks.h (CUPTI 13 API).
    _fields_ = [('site', C.c_int), ('function', C.c_char_p),
                ('params', C.c_void_p), ('retval', C.c_void_p),
                ('symbol', C.c_char_p), ('context', C.c_void_p),
                ('uid', C.c_uint32), ('correlation', C.c_void_p), ('id', C.c_uint32)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--node', required=True)
    parser.add_argument('--cupti', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    pin = json.loads(Path(__file__).with_name('soma-reference.json').read_text())
    manifest = json.loads((args.capture/'manifest.json').read_text())
    identity = next(x for x in pin['source_assets'] if x['path'].endswith('_fp16.onnx'))
    if (manifest['reference'] != pin or manifest['model'] != identity or
            manifest['precision'] != 'mixed-fp16' or manifest['tapped_graph'] != 'source' or
            manifest['provider'] != 'CUDAExecutionProvider' or digest(args.model) != identity['sha256'] or
            digest(args.capture/'tensors.npz') != manifest['capture_sha256']):
        raise ValueError('Requires verified pinned CUDA mixed-FP16 source capture')
    model = onnx.load(args.model)
    weights = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    nodes = [n for n in model.graph.node if n.name == args.node and n.op_type in ('MatMul', 'Gemm', 'Conv')]
    if len(nodes) != 1:
        raise ValueError('Expected exactly one source MatMul, Gemm, or Conv node')
    node = nodes[0]
    with np.load(args.capture/'tensors.npz', allow_pickle=False) as capture:
        def value(name):
            return weights[name] if name in weights else capture['graph/'+name]
        operands = [value(k).copy() for k in node.input]
        a, b = operands[:2]
        expected = value(node.output[0]).copy()
    if any(x.dtype != np.float16 or not np.isfinite(x).all() for x in (*operands, expected)):
        raise ValueError('Requires finite FP16 operands and output')

    cupti = C.CDLL(str(args.cupti.resolve()))
    def checked(status):
        if status:
            message = C.c_char_p()
            cupti.cuptiGetResultString(status, C.byref(message))
            raise RuntimeError(f'CUPTI {status}: {message.value!r}')
    kernels = []
    @C.CFUNCTYPE(None, C.c_void_p, C.c_int, C.c_uint32, C.c_void_p)
    def callback(user, domain, callback_id, raw):
        if domain != 1:
            return
        data = C.cast(raw, C.POINTER(CallbackData)).contents
        # symbolName is defined only on launch callbacks. Ignore unrelated APIs.
        if data.site == 0 and data.function and data.function.startswith(b'cuLaunch') and data.symbol:
            kernels.append(dict(api=data.function.decode(), kernel=data.symbol.decode()))
    subscriber = C.c_void_p()
    checked(cupti.cuptiSubscribe(C.byref(subscriber), callback, None))
    try:
        checked(cupti.cuptiEnableDomain(1, subscriber, 1))
        ort.preload_dlls()
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.intra_op_num_threads = 1
        names = ['a', 'b', 'c'][:len(operands)]
        attrs = {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}
        graph = helper.make_graph([helper.make_node(node.op_type, names, ['y'], **attrs)], 'isolated_matmul',
            [helper.make_tensor_value_info(k, TensorProto.FLOAT16, x.shape) for k, x in zip(names, operands)],
            [helper.make_tensor_value_info('y', TensorProto.FLOAT16, expected.shape)])
        isolated = helper.make_model(graph, opset_imports=model.opset_import)
        isolated.ir_version = model.ir_version
        session = ort.InferenceSession(isolated.SerializeToString(), options, providers=['CUDAExecutionProvider'])
        if session.get_providers()[0] != 'CUDAExecutionProvider':
            raise ValueError('CUDA provider unavailable')
        # Exclude first-run cuDNN algorithm search from launch attribution.
        session.run(None, dict(zip(names, operands)))
        kernels.clear()
        actual = session.run(None, dict(zip(names, operands)))[0]
        if not np.array_equal(actual, expected):
            raise ValueError('Isolated operation differs from full source capture')
        if not kernels or any(x['kernel'] == 'Unknown' for x in kernels):
            raise ValueError('No usable CUDA kernel attribution')
        runtime = runtime_identity('CUDAExecutionProvider', session.get_provider_options())
    finally:
        checked(cupti.cuptiUnsubscribe(subscriber))
    report = dict(format='motionbricks-soma-cuda-matmul-profile-v1', source_revision=pin['source_revision'],
                  node=node.name, operation=node.op_type, model_sha256=identity['sha256'], capture_sha256=manifest['capture_sha256'],
                  cupti_sha256=digest(args.cupti), runtime=runtime, kernels=kernels,
                  left_shape=list(a.shape), right_shape=list(b.shape), values=int(actual.size),
                  isolated_output_bitwise_equal=True, scope='isolated kernel attribution, not full-model parity')
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(values=report['values'], kernels=kernels, exact=True)))


if __name__ == '__main__':
    main()
