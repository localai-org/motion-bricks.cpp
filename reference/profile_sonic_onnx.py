#!/usr/bin/env python3
"""Benchmark the pinned SONIC ONNX graphs with CPUExecutionProvider only.

Uses fixed output buffers and I/O binding. Loading, validation and warmup
are excluded from latency; input binding and encoder-to-decoder copying are
included. Intended to accompany profile_sonic_cpu.py on the same CPU masks.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import struct
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--cpus', required=True)
    p.add_argument('--iterations', type=int, default=2205)
    p.add_argument('--warmup', type=int, default=100)
    p.add_argument('--optimization', choices=('all', 'disabled'), default='all')
    p.add_argument('--encoder-scope', choices=('original', 'g1'), default='original')
    p.add_argument('--spinning', choices=('0', '1'), default='0')
    p.add_argument('--period-ms', type=float, default=0)
    args = p.parse_args()
    cpus = [int(v) for v in args.cpus.split(',')]
    if len(cpus) not in (1, 2) or len(set(cpus)) != len(cpus) or not set(cpus) <= os.sched_getaffinity(0):
        p.error('select one or two allowed CPUs')
    topology = [tuple((Path(f'/sys/devices/system/cpu/cpu{cpu}/topology') / name).read_text().strip()
                      for name in ('physical_package_id', 'core_id')) for cpu in cpus]
    if len(set(topology)) != len(cpus):
        p.error('select distinct physical cores')
    if args.iterations < 1 or args.warmup < 0 or not math.isfinite(args.period_ms) or args.period_ms < 0:
        p.error('invalid iterations/warmup/period')
    os.sched_setaffinity(0, cpus)
    os.environ.update(OMP_NUM_THREADS=str(len(cpus)), OMP_THREAD_LIMIT=str(len(cpus)),
                      OMP_DYNAMIC='FALSE', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    import numpy as np
    import onnxruntime as ort

    hashes = {'encoder': '013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3',
              'decoder': 'c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed'}
    sessions, graphs = {}, {}
    encoder_output = 'encoded_tokens'
    for component, expected in hashes.items():
        path = args.policy / f'model_{component}.onnx'
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            p.error(f'unsupported/corrupt {component} ONNX')
        options = ort.SessionOptions()
        options.intra_op_num_threads = len(cpus)
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = (ort.GraphOptimizationLevel.ORT_ENABLE_ALL if args.optimization == 'all'
                                             else ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
        options.add_session_config_entry('session.intra_op.allow_spinning', args.spinning)
        options.add_session_config_entry('session.inter_op.allow_spinning', args.spinning)
        import onnx
        model = onnx.load(path, load_external_data=False)
        if component == 'encoder' and args.encoder_scope == 'g1':
            branch_output = '/quantizer/Cast_output_0'
            model = onnx.utils.Extractor(onnx.shape_inference.infer_shapes(model)).extract_model(['obs_dict'], [branch_output])
            model.graph.initializer.append(onnx.helper.make_tensor('profile_output_shape', onnx.TensorProto.INT64, [2], [1, 64]))
            model.graph.node.append(onnx.helper.make_node('Reshape', [branch_output, 'profile_output_shape'], [encoder_output]))
            del model.graph.output[:]
            model.graph.output.append(onnx.helper.make_tensor_value_info(encoder_output, onnx.TensorProto.FLOAT, [1, 64]))
            onnx.checker.check_model(model)
        serialized = model.SerializeToString()
        graphs[component] = dict(sha256=hashlib.sha256(serialized).hexdigest(), nodes=len(model.graph.node),
                                 gemm=sum(n.op_type == 'Gemm' for n in model.graph.node),
                                 matmul=sum(n.op_type == 'MatMul' for n in model.graph.node))
        sessions[component] = ort.InferenceSession(serialized, sess_options=options, providers=['CPUExecutionProvider'])
    blob = args.fixture.read_bytes()
    if len(blob) < 12 or blob[:8] != b'MBSONIC1':
        p.error('invalid fixture header')
    count, = struct.unpack_from('<I', blob, 8)
    stride = 1762 + 994 + 4160 + 64 + 7197 + 29 + 29
    if not 1 <= count <= 20000 or len(blob) != 12 + count * stride * 4:
        p.error('invalid fixture sample count/length')
    fixture = np.frombuffer(blob, dtype='<f4', offset=12).reshape(count, stride)
    encoder_inputs = fixture[:, :1762].copy().reshape(count, 1, 1762)
    if np.any(encoder_inputs[:, :, 0] != 0):
        p.error('this comparison supports G1 mode-0 observations only')
    decoder_inputs = fixture[:, 1762:2756].copy().reshape(count, 1, 994)
    expected_tokens = fixture[:, 2756 + 4160:2756 + 4160 + 64]
    expected_actions, expected_composed = fixture[:, -58:-29], fixture[:, -29:]
    tokens, actions = np.empty((1, 64), dtype=np.float32), np.empty((1, 29), dtype=np.float32)
    encoder, decoder = sessions['encoder'], sessions['decoder']
    eb, db = encoder.io_binding(), decoder.io_binding()
    eb.bind_output(encoder_output, 'cpu', 0, np.float32, tokens.shape, tokens.ctypes.data)
    db.bind_output('action', 'cpu', 0, np.float32, actions.shape, actions.ctypes.data)

    def encode(e):
        eb.bind_cpu_input('obs_dict', e)
        encoder.run_with_iobinding(eb)

    def decode(d):
        db.bind_cpu_input('obs_dict', d)
        decoder.run_with_iobinding(db)

    # Compare the serving graphs' outputs, without adding diagnostic outputs
    # that could block optimizer transformations or change the timed graphs.
    parity = dict(samples=count, token_mismatches=0, action_failed_samples=0,
                  composed_failed_samples=0, max_abs=0., max_relative_l2=0.)

    def compare(actual, expected, field):
        a, b = actual.reshape(-1).astype(np.float64), expected.astype(np.float64)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise RuntimeError('non-finite parity output')
        delta = a - b
        absolute = float(np.max(np.abs(delta)))
        relative = float(np.sqrt(np.dot(delta, delta) / max(float(np.dot(b, b)), 1e-20)))
        parity['max_abs'] = max(parity['max_abs'], absolute)
        parity['max_relative_l2'] = max(parity['max_relative_l2'], relative)
        parity[field] += int(absolute > 5e-4 or relative > 1e-5)

    for i in range(count):
        encode(encoder_inputs[i])
        parity['token_mismatches'] += int(np.count_nonzero(tokens.reshape(-1) != expected_tokens[i]))
        decode(decoder_inputs[i])
        compare(actions, expected_actions[i], 'action_failed_samples')
        decoder_inputs[i, :, :64] = tokens
        decode(decoder_inputs[i])
        compare(actions, expected_composed[i], 'composed_failed_samples')
    parity['pass'] = not any(parity[k] for k in ('token_mismatches', 'action_failed_samples', 'composed_failed_samples'))
    # Restore captured histories/tokens; benchmark composition overwrites tokens.
    decoder_inputs[:, 0, :] = fixture[:, 1762:2756]
    enc_ns, dec_ns, pair_ns, lateness = [], [], [], []
    misses = 0
    now = time.perf_counter_ns
    for i in range(args.warmup + args.iterations):
        if i == args.warmup:
            usage_start = resource.getrusage(resource.RUSAGE_SELF)
            wall_start = now()
        if i >= args.warmup and args.period_ms:
            deadline = wall_start + (i - args.warmup) * args.period_ms * 1e6
            remaining = (deadline - now()) / 1e9
            if remaining > 0:
                time.sleep(remaining)
        e, d = encoder_inputs[i % count], decoder_inputs[i % count]
        t0 = now()
        encode(e)
        t1 = now()
        d[:, :64] = tokens
        t2 = now()
        decode(d)
        t3 = now()
        if i >= args.warmup:
            enc_ns.append(t1 - t0)
            dec_ns.append(t3 - t2)
            pair_ns.append(t3 - t0)
            if args.period_ms:
                lateness.append(max(0, t0 - deadline))
                misses += t3 > deadline + args.period_ms * 1e6
    wall_ns = now() - wall_start
    usage_end = resource.getrusage(resource.RUSAGE_SELF)
    affinity = {task.name: sorted(os.sched_getaffinity(int(task.name))) for task in Path('/proc/self/task').iterdir()}
    if any(not set(mask) <= set(cpus) for mask in affinity.values()):
        raise RuntimeError('worker escaped requested CPU affinity')

    def stats(values):
        ordered = sorted(values)
        def percentile(q):
            return ordered[max(0, math.ceil(q * len(ordered)) - 1)] / 1e6
        return dict(mean_ms=sum(values) / len(values) / 1e6, p50_ms=percentile(.5),
                    p95_ms=percentile(.95), p99_ms=percentile(.99), max_ms=max(values) / 1e6)

    result = dict(runtime=ort.__version__, runtime_build=ort.get_build_info(),
                  providers={name: s.get_providers() for name, s in sessions.items()},
                  policy_sha256=hashes, fixture_sha256=hashlib.sha256(blob).hexdigest(),
                  encoder_scope=args.encoder_scope, serving_graphs=graphs, onnx_version=onnx.__version__,
                  optimization=args.optimization, spinning=args.spinning, execution='sequential',
                  session_threadpools='separate encoder and decoder pools',
                  cpus=cpus, physical_cores=topology, intra_op_threads=len(cpus), inter_op_threads=1,
                  worker_affinity=affinity, iterations=args.iterations, warmup=args.warmup,
                  period_ms=args.period_ms, parity=parity, encoder=stats(enc_ns), decoder=stats(dec_ns),
                  pair=stats(pair_ns), wall_seconds=wall_ns / 1e9,
                  cpu_seconds=(usage_end.ru_utime + usage_end.ru_stime - usage_start.ru_utime - usage_start.ru_stime),
                  pairs_over_20ms=sum(v > 20e6 for v in pair_ns), action_checksum=float(actions.sum()))
    if args.period_ms:
        result.update(start_lateness=stats(lateness), deadline_misses=misses)
    print(json.dumps(result, indent=2))
    return 0 if parity['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
