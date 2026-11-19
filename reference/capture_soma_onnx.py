#!/usr/bin/env python3
"""Capture the official SOMA FP32 or mixed-FP16 ONNX graph on CPU or CUDA.

--tap-layers exposes every inferred tensor output in the graph, including
functional attention projections and FP16/FP32 casts absent from module hooks.
Graph optimizations are disabled to retain the upstream precision boundaries.
"""
import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

from fetch_soma import verify
def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def runtime_identity(provider, provider_options):
    """Record arithmetic-relevant environment details, without dumping the environment."""
    packages={dist.metadata['Name']:dist.version for dist in importlib.metadata.distributions()
              if dist.metadata.get('Name','').lower().startswith(('nvidia-','onnxruntime'))}
    result=dict(platform=platform.system(),machine=platform.machine(),python=platform.python_version(),
                numpy=np.__version__,onnx=onnx.__version__,packages=packages,
                environment={key:os.environ.get(key) for key in
                             ('CUDA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG',
                              'NVIDIA_TF32_OVERRIDE','ORT_CUDA_GEMM_OPTIONS')})
    if provider!='CUDAExecutionProvider':return result
    # Query the same CUDA ordinal selected by ORT, including visibility remapping.
    driver=(ctypes.WinDLL('nvcuda.dll') if os.name=='nt' else ctypes.CDLL('libcuda.so.1'))
    def checked(name,*args):
        status=getattr(driver,name)(*args)
        if status:raise RuntimeError(f'{name} failed while recording CUDA provenance: {status}')
    checked('cuInit',0)
    ordinal=int(provider_options[provider]['device_id'])
    device=ctypes.c_int();version=ctypes.c_int();major=ctypes.c_int();minor=ctypes.c_int()
    checked('cuDeviceGet',ctypes.byref(device),ordinal)
    name=ctypes.create_string_buffer(256)
    checked('cuDeviceGetName',name,len(name),device)
    checked('cuDriverGetVersion',ctypes.byref(version))
    checked('cuDeviceComputeCapability',ctypes.byref(major),ctypes.byref(minor),device)
    result['cuda_device']=dict(ordinal=ordinal,name=name.value.decode(),
                               compute_capability=[major.value,minor.value],driver_api_version=version.value)
    maps=Path('/proc/self/maps')
    if maps.exists():
        result['loaded_cuda_libraries']=sorted({Path(line.split()[-1]).name for line in maps.read_text().splitlines()
            if '/' in line and Path(line.split()[-1]).name.startswith(('libcuda.','libcudart.','libcublas','libcudnn'))})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tap-layers', action='store_true')
    parser.add_argument('--tap-executed-graph', action='store_true',
                        help='Expose runtime-lowered nodes, including CPU FP16 promotions, rather than source nodes')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--provider', choices=('cpu','cuda'), default='cpu')
    parser.add_argument('--cuda-gemm-options',type=int,choices=(0,2),default=0,
                        help='ORT CUDA GEMM policy: 0=default; 2=disallow reduced-precision reductions')
    args = parser.parse_args()
    if args.tap_executed_graph and not args.tap_layers:
        parser.error('--tap-executed-graph requires --tap-layers')
    if args.provider!='cuda' and args.cuda_gemm_options:
        parser.error('--cuda-gemm-options requires --provider cuda')
    manifest = json.loads(Path(__file__).with_name('soma-reference.json').read_text())
    candidates = [x for x in manifest['source_assets'] if x['path'].endswith('.onnx')]
    identity = next((x for x in candidates if verify(args.model, x)), None)
    if identity is None:
        raise ValueError('Model must match an official pinned SOMA ONNX artifact')
    precision = 'mixed-fp16' if identity['path'].endswith('_fp16.onnx') else 'fp32'
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    provider='CPUExecutionProvider' if args.provider=='cpu' else 'CUDAExecutionProvider'
    if args.provider=='cuda':
        os.environ['ORT_CUDA_GEMM_OPTIONS']=str(args.cuda_gemm_options)
        ort.preload_dlls()
    if provider not in ort.get_available_providers():
        raise ValueError(f'Requested provider unavailable: {provider}')
    providers=[provider]
    with np.load(args.inputs, allow_pickle=False) as archive:
        inputs = {name: archive[name].copy() for name in archive.files}
    with tempfile.TemporaryDirectory(prefix='soma-ort-') as temporary:
        lowered = Path(temporary) / 'executed.onnx'
        options.enable_profiling=args.provider=='cuda'
        options.profile_file_prefix=str(Path(temporary)/'placement')
        if args.tap_executed_graph:
            options.optimized_model_filepath = str(lowered)
        session = ort.InferenceSession(str(args.model), options, providers=providers)
        if session.get_providers()[0]!=provider:
            raise ValueError('Runtime fell back from the requested execution provider')
        provider_options=session.get_provider_options()
        original_names = [x.name for x in session.get_outputs()]
        baseline = dict(zip(original_names, session.run(None, inputs)))
        placement=[]
        if options.enable_profiling:
            profile=json.loads(Path(session.end_profiling()).read_text())
            counts=Counter((event['args'].get('op_name'),event['args'].get('provider'))
                           for event in profile if event.get('cat')=='Node' and event.get('args',{}).get('provider'))
            placement=[dict(op=op,provider=ep,count=count) for (op,ep),count in sorted(counts.items())]
        del session
        options.enable_profiling=False
        options.optimized_model_filepath = ''
        lowered_graph = onnx.load(lowered) if args.tap_executed_graph else None
        lowered_sha256 = sha256(lowered) if args.tap_executed_graph else None
    tensor_outputs = {}
    nodes = []
    trace_error = {}
    if args.tap_layers:
        graph = onnx.shape_inference.infer_shapes(
            lowered_graph if lowered_graph is not None else onnx.load(args.model), strict_mode=True)
        info = {x.name: x for x in [*graph.graph.value_info, *graph.graph.output]}
        exposed = set(original_names)
        for node in graph.graph.node:
            nodes.append(dict(name=node.name, op=node.op_type, inputs=list(node.input), outputs=list(node.output)))
            for name in node.output:
                if not name or name in exposed:
                    continue
                if name not in info or not info[name].type.HasField('tensor_type'):
                    raise ValueError(f'Cannot expose graph output {name!r}; missing tensor type')
                graph.graph.output.append(info[name])
                exposed.add(name)
        session = ort.InferenceSession(graph.SerializeToString(), options, providers=providers)
        if session.get_providers()[0]!=provider:
            raise ValueError('Instrumented runtime fell back from requested provider')
        del graph
        results = dict(zip([x.name for x in session.get_outputs()], session.run(None, inputs)))
        for name in original_names:
            delta = np.abs(results[name].astype(np.float64) - baseline[name].astype(np.float64))
            trace_error[name] = float(delta.max())
            np.testing.assert_array_equal(results[name], baseline[name])
        tensor_outputs.update({'graph/' + name: value for name, value in results.items()})
    tensor_outputs.update({'input/' + name: value for name, value in inputs.items()})
    tensor_outputs.update({'output/' + name: value for name, value in baseline.items()})
    for name, value in baseline.items():
        if not np.isfinite(value).all():
            raise ValueError(f'Non-finite output: {name}')
    args.output.mkdir(parents=True, exist_ok=False)
    if lowered_graph is not None:
        onnx.save(lowered_graph, args.output / 'runtime-lowered.onnx')
        if sha256(args.output / 'runtime-lowered.onnx') != lowered_sha256:
            raise ValueError('Lowered graph serialization changed its recorded identity')
    np.savez(args.output / 'tensors.npz', **tensor_outputs)
    report = dict(schema_version=1, reference=manifest, model=identity,
                  precision=precision, runtime=ort.__version__, provider=provider,
                  provider_options=provider_options,
                  runtime_identity=runtime_identity(provider,provider_options),
                  cuda_gemm_options=args.cuda_gemm_options if args.provider=='cuda' else None,
                  graph_optimizations='disabled', threads=args.threads,
                  tapped_graph='runtime-lowered' if args.tap_executed_graph else 'source',
                  lowered_graph_sha256=lowered_sha256,
                  execution_placement=placement,
                  input_sha256=sha256(args.inputs), capture_sha256=sha256(args.output / 'tensors.npz'),
                  instrumentation_max_absolute_error=trace_error, nodes=nodes,
                  tensors={name: dict(shape=list(value.shape), dtype=str(value.dtype))
                           for name, value in tensor_outputs.items()})
    (args.output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Captured {len(tensor_outputs)} tensors ({precision}) in {args.output}')


if __name__ == '__main__':
    main()
