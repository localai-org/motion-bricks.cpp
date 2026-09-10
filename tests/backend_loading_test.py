"""Exercise CPU module discovery through the public C ABI in fresh processes.

No third-party Python packages are required. Can also check an installed library:
  python3 tests/backend_loading_test.py --library PREFIX/lib/libmotionbricks.so \
    --model generated/sonic/ggml/sonic-g1.gguf
"""
import argparse
import concurrent.futures
import ctypes as c
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def child(args):
    lib = c.CDLL(str(args.library))
    signatures = {
        'mb_runtime_options_create': [c.POINTER(c.c_void_p), c.c_void_p, c.c_uint64],
        'mb_runtime_options_set_threads': [c.c_void_p, c.c_uint32, c.c_void_p, c.c_uint64],
        'mb_runtime_options_set_backend_directory': [c.c_void_p, c.c_char_p, c.c_void_p, c.c_uint64],
        'mb_sonic_load': [c.c_char_p, c.c_void_p, c.POINTER(c.c_void_p), c.c_void_p, c.c_uint64],
        'mb_sonic_encode': [c.c_void_p, c.c_void_p, c.c_uint64, c.c_void_p, c.c_uint64, c.c_void_p, c.c_uint64],
        'mb_sonic_decode': [c.c_void_p, c.c_void_p, c.c_uint64, c.c_void_p, c.c_uint64, c.c_void_p, c.c_uint64],
    }
    for name, signature in signatures.items():
        getattr(lib, name).argtypes = signature
        getattr(lib, name).restype = c.c_uint32
    for name in ('mb_sonic_free', 'mb_runtime_options_free'):
        getattr(lib, name).argtypes = [c.c_void_p]
        getattr(lib, name).restype = None

    def infer():
        error = c.create_string_buffer(1024)
        options, model = c.c_void_p(), c.c_void_p()

        def check(status):
            if status:
                raise RuntimeError(error.value.decode())

        try:
            check(lib.mb_runtime_options_create(c.byref(options), error, len(error)))
            check(lib.mb_runtime_options_set_threads(options, 1, error, len(error)))
            if args.retry_missing:
                check(lib.mb_runtime_options_set_backend_directory(options, b'/nonexistent-motionbricks-backends', error, len(error)))
                status = lib.mb_sonic_load(os.fsencode(args.model), options, c.byref(model), error, len(error))
                if status != 6 or model.value:  # MB_BACKEND_UNAVAILABLE
                    raise RuntimeError(f'missing modules must return backend-unavailable, got {status}')
                args.retry_missing = False
            directory = os.fsencode(args.directory) if args.directory else None
            check(lib.mb_runtime_options_set_backend_directory(options, directory, error, len(error)))
            check(lib.mb_sonic_load(os.fsencode(args.model), options, c.byref(model), error, len(error)))
            enc, dec = (c.c_float * 1762)(), (c.c_float * 994)()
            tokens, actions = (c.c_float * 64)(), (c.c_float * 29)()
            check(lib.mb_sonic_encode(model, enc, len(enc), tokens, len(tokens), error, len(error)))
            dec[:64] = tokens[:]
            check(lib.mb_sonic_decode(model, dec, len(dec), actions, len(actions), error, len(error)))
            if not all(math.isfinite(x) for x in [*tokens, *actions]):
                raise RuntimeError('non-finite inference output')
            return list(actions)
        finally:
            lib.mb_sonic_free(model)
            lib.mb_runtime_options_free(options)

    if args.concurrent:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: infer(), range(2)))
    else:
        results = [infer()]
    # A second model must reuse the selected module after the first is freed.
    if any(result != results[0] for result in results) or infer() != results[0]:
        raise RuntimeError('repeated/concurrent loads produced different outputs')
    print(json.dumps({'actions': results[0]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--concurrent', action='store_true')
    parser.add_argument('--retry-missing', action='store_true')
    args = parser.parse_args()
    args.library, args.model = args.library.resolve(), args.model.resolve()
    if args.child:
        child(args)
        return
    modules = sorted(args.library.parent.glob('libggml-cpu-*.so'))
    if not modules:
        raise RuntimeError('no packaged CPU variants found')
    # Score functions are compiled without ISA flags and safe on older CPUs.
    scored = []
    for module in modules:
        backend = c.CDLL(str(module))
        backend.ggml_backend_score.restype = c.c_int
        scored.append((backend.ggml_backend_score(), module))
    best_score = max(score for score, _ in scored)
    if best_score <= 0:
        raise RuntimeError('no compatible CPU variant packaged')
    best = {module.name for score, module in scored if score == best_score}
    with tempfile.TemporaryDirectory(prefix='motionbricks-backend-test-') as temporary:
        root = Path(temporary)
        env = dict(os.environ, OMP_NUM_THREADS='1', OMP_THREAD_LIMIT='2', OMP_DYNAMIC='FALSE')
        env.pop('GGML_BACKEND_PATH', None)
        command = [sys.executable, str(Path(__file__).resolve()), '--child', '--library', str(args.library), '--model', str(args.model)]

        def run(extra, expected):
            result = subprocess.run(command + extra, cwd=root, env=env, text=True, capture_output=True)
            if result.returncode:
                raise RuntimeError(f'backend child failed ({result.returncode}):\n{result.stdout}\n{result.stderr}')
            loads = [line for line in result.stderr.splitlines() if 'loaded CPU backend from ' in line]
            if len(loads) != 1 or not any(loads[0].endswith(name) for name in expected):
                raise RuntimeError(f'incorrect module selection/reload: {result.stderr}')
            print(loads[0])
            return json.loads(result.stdout)['actions']

        run(['--concurrent'], best)
        run(['--retry-missing'], best)
        # Exercise fallback packaging, independent of this machine's newer ISA.
        baseline = args.library.parent / 'libggml-cpu-x64.so'
        if baseline.exists():
            fallback = root / 'baseline'
            fallback.mkdir()
            shutil.copy2(baseline, fallback / baseline.name)
            run(['--directory', str(fallback)], {baseline.name})
    print('CPU discovery, repeat/concurrent loads, missing-path retry and fallback passed')


if __name__ == '__main__':
    main()
