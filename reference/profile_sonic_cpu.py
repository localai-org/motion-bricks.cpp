#!/usr/bin/env python3
"""Linux CPU-only SONIC latency harness; uses local weights and MBSONIC1 captures.

Pins the process before loading the backend (workers inherit affinity). All
fixture I/O and allocation happen before warmup. Timings include ctypes/API
overhead; parity checking is deliberately separate. JSON goes to stdout.
"""
import argparse
import ctypes as c
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import struct
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--cpus', required=True, help='One or two logical CPU IDs on distinct physical cores')
    p.add_argument('--batch', type=int, default=1, help='Independent SONIC requests per API call')
    p.add_argument('--iterations', type=int, default=6615)
    p.add_argument('--warmup', type=int, default=100)
    p.add_argument('--period-ms', type=float, default=0, help='0 for burst; 20 for 50 Hz arrivals')
    args = p.parse_args()
    cpus = [int(v) for v in args.cpus.split(',')]
    if len(cpus) not in (1, 2) or len(set(cpus)) != len(cpus):
        p.error('select one or two distinct CPUs')
    if not set(cpus) <= os.sched_getaffinity(0):
        p.error('CPU selection is outside allowed affinity')
    topology = []
    for cpu in cpus:
        root = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        topology.append(tuple((root / name).read_text().strip()
                              for name in ('physical_package_id', 'core_id')))
    if len(set(topology)) != len(cpus):
        p.error('select distinct physical cores, not SMT siblings')
    if (args.iterations < 1 or args.warmup < 0 or not 1 <= args.batch <= 64
            or not math.isfinite(args.period_ms) or args.period_ms < 0):
        p.error('invalid iteration/warmup/period')
    os.sched_setaffinity(0, cpus)
    os.environ.update(OMP_NUM_THREADS=str(len(cpus)), OMP_THREAD_LIMIT=str(len(cpus)),
                      OMP_DYNAMIC='FALSE', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    blob = args.fixture.read_bytes()
    if len(blob) < 12 or blob[:8] != b'MBSONIC1':
        p.error('invalid MBSONIC1 fixture')
    count, = struct.unpack_from('<I', blob, 8)
    # Encoder/decoder observations, five encoder preactivations, tokens,
    # seven decoder preactivations, actions, composed actions.
    stride = 4 * (1762 + 994 + 2048 + 1024 + 512 + 512 + 64 + 64
                  + 2048 + 2048 + 1024 + 1024 + 512 + 512 + 29 + 29 + 29)
    if not 1 <= count <= 20000 or len(blob) != 12 + count * stride:
        p.error('invalid fixture length/sample count')
    enc_type, dec_type = c.c_float * 1762, c.c_float * 994
    samples = [(enc_type.from_buffer_copy(blob, 12 + i * stride),
                dec_type.from_buffer_copy(blob, 12 + i * stride + 1762 * 4)) for i in range(count)]
    if args.batch > 1:
        enc_batch_type = c.c_float * (1762 * args.batch)
        dec_batch_type = c.c_float * (994 * args.batch)
        batched = []
        for i in range(count):
            enc_batch, dec_batch = enc_batch_type(), dec_batch_type()
            for item in range(args.batch):
                enc, dec = samples[(i * args.batch + item) % count]
                c.memmove(c.byref(enc_batch, item * 1762 * c.sizeof(c.c_float)), enc, c.sizeof(enc))
                c.memmove(c.byref(dec_batch, item * 994 * c.sizeof(c.c_float)), dec, c.sizeof(dec))
            batched.append((enc_batch, dec_batch))
        samples = batched
    tokens, actions = (c.c_float * (64 * args.batch))(), (c.c_float * (29 * args.batch))()
    lib = c.CDLL(str(args.library.resolve()))
    error = c.create_string_buffer(1024)
    signatures = {
        'mb_runtime_options_create': [c.POINTER(c.c_void_p), c.c_void_p, c.c_uint64],
        'mb_runtime_options_set_device': [c.c_void_p, c.c_uint32, c.c_void_p, c.c_uint64],
        'mb_runtime_options_set_threads': [c.c_void_p, c.c_uint32, c.c_void_p, c.c_uint64],
        'mb_sonic_load': [c.c_char_p, c.c_void_p, c.POINTER(c.c_void_p), c.c_void_p, c.c_uint64],
        'mb_sonic_encode': [c.c_void_p, c.c_void_p, c.c_uint64, c.c_void_p, c.c_uint64, c.c_void_p, c.c_uint64],
        'mb_sonic_decode': [c.c_void_p, c.c_void_p, c.c_uint64, c.c_void_p, c.c_uint64, c.c_void_p, c.c_uint64],
        'mb_sonic_encode_batch': [c.c_void_p, c.c_void_p, c.c_uint64, c.c_void_p, c.c_uint64, c.c_uint32,
                                  c.c_void_p, c.c_uint64],
        'mb_sonic_decode_batch': [c.c_void_p, c.c_void_p, c.c_uint64, c.c_void_p, c.c_uint64, c.c_uint32,
                                  c.c_void_p, c.c_uint64],
    }
    for name, signature in signatures.items():
        getattr(lib, name).argtypes = signature
        getattr(lib, name).restype = c.c_uint32
    for name in ('mb_sonic_free', 'mb_runtime_options_free'):
        getattr(lib, name).argtypes = [c.c_void_p]
        getattr(lib, name).restype = None

    def check(status):
        if status:
            raise RuntimeError(error.value.decode())

    options, model = c.c_void_p(), c.c_void_p()
    enc_ns, dec_ns, pair_ns, start_lateness_ns = [], [], [], []
    deadline_misses = 0
    encode = lib.mb_sonic_encode if args.batch == 1 else lib.mb_sonic_encode_batch
    decode = lib.mb_sonic_decode if args.batch == 1 else lib.mb_sonic_decode_batch
    now = time.perf_counter_ns
    try:
        check(lib.mb_runtime_options_create(c.byref(options), error, len(error)))
        check(lib.mb_runtime_options_set_device(options, 1, error, len(error)))  # MB_DEVICE_CPU
        check(lib.mb_runtime_options_set_threads(options, len(cpus), error, len(error)))
        check(lib.mb_sonic_load(os.fsencode(args.model), options, c.byref(model), error, len(error)))
        for i in range(args.warmup + args.iterations):
            if i == args.warmup:
                usage_start = resource.getrusage(resource.RUSAGE_SELF)
                wall_start = now()
            if i >= args.warmup and args.period_ms:
                deadline = wall_start + (i - args.warmup) * args.period_ms * 1e6
                remaining = (deadline - now()) / 1e9
                if remaining > 0:
                    time.sleep(remaining)
            enc, dec = samples[i % count]
            t0 = now()
            if args.batch == 1:
                check(encode(model, enc, 1762, tokens, 64, error, len(error)))
            else:
                check(encode(model, enc, 1762 * args.batch, tokens, 64 * args.batch,
                             args.batch, error, len(error)))
            t1 = now()
            for item in range(args.batch):
                c.memmove(c.byref(dec, item * 994 * c.sizeof(c.c_float)),
                          c.byref(tokens, item * 64 * c.sizeof(c.c_float)), 64 * c.sizeof(c.c_float))
            t2 = now()
            if args.batch == 1:
                check(decode(model, dec, 994, actions, 29, error, len(error)))
            else:
                check(decode(model, dec, 994 * args.batch, actions, 29 * args.batch,
                             args.batch, error, len(error)))
            t3 = now()
            if i >= args.warmup:
                enc_ns.append(t1 - t0)
                dec_ns.append(t3 - t2)
                pair_ns.append(t3 - t0)
                if args.period_ms:
                    start_lateness_ns.append(max(0, t0 - deadline))
                    deadline_misses += t3 > deadline + args.period_ms * 1e6
        wall_ns = now() - wall_start
        usage_end = resource.getrusage(resource.RUSAGE_SELF)
        worker_affinity = {task.name: sorted(os.sched_getaffinity(int(task.name)))
                           for task in Path('/proc/self/task').iterdir()}
        if any(not set(mask) <= set(cpus) for mask in worker_affinity.values()):
            raise RuntimeError('worker escaped requested CPU affinity')
    finally:
        lib.mb_sonic_free(model)
        lib.mb_runtime_options_free(options)

    def stats(values):
        ordered = sorted(values)
        def percentile(q):
            return ordered[max(0, math.ceil(q * len(ordered)) - 1)] / 1e6
        return dict(mean_ms=sum(values) / len(values) / 1e6, p50_ms=percentile(.5),
                    p95_ms=percentile(.95), p99_ms=percentile(.99), max_ms=max(values) / 1e6)

    result = dict(library=str(args.library), model=str(args.model), fixture=str(args.fixture),
                  model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                  fixture_sha256=hashlib.sha256(blob).hexdigest(), machine=platform.uname()._asdict(),
                  cpus=cpus, physical_cores=topology, threads=len(cpus), worker_affinity=worker_affinity,
                  experiment_environment={name: os.getenv(name) for name in
                                          ('LD_PRELOAD', 'SONIC_CPU_EXPERIMENT', 'SONIC_CPU_EXPERIMENT_SCOPE')},
                  openmp_environment={name: os.getenv(name) for name in
                                      ('OMP_NUM_THREADS', 'OMP_THREAD_LIMIT', 'OMP_DYNAMIC',
                                       'OMP_WAIT_POLICY', 'GOMP_SPINCOUNT', 'OMP_PROC_BIND', 'OMP_PLACES')},
                  batch=args.batch, iterations=args.iterations, warmup=args.warmup, fixture_samples=count,
                  period_ms=args.period_ms, encoder=stats(enc_ns), decoder=stats(dec_ns), pair=stats(pair_ns),
                  aggregate_pairs_per_second=args.batch * len(pair_ns) * 1e9 / sum(pair_ns),
                  wall_seconds=wall_ns / 1e9,
                  cpu_seconds=(usage_end.ru_utime + usage_end.ru_stime - usage_start.ru_utime - usage_start.ru_stime),
                  voluntary_context_switches=usage_end.ru_nvcsw - usage_start.ru_nvcsw,
                  involuntary_context_switches=usage_end.ru_nivcsw - usage_start.ru_nivcsw,
                  pairs_over_20ms=sum(v > 20e6 for v in pair_ns), action_checksum=sum(actions))
    if args.period_ms:
        result.update(start_lateness=stats(start_lateness_ns), deadline_misses=deadline_misses)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
