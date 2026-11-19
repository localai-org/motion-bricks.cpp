#!/usr/bin/env python3
"""Regenerate the pinned sm_120 instruction tables used by SOMA CPU emulation.

Requires an sm_120 NVIDIA GPU, its driver, CUDA NVRTC, and NumPy. This exhaustively
measures instructions; it does not use model outputs or fit motion fixtures.
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import numpy as np

PINNED = {
    'rsqrt': '002e5ef8ce6e36e6beec010a9e941a82c6a8ba997f273b3ab05fbe5d1eb86ba2',
    'exp2': '11d89bff841e859226e57a64735f294cd2cc2b77665f25b2b953ea3ff83fcbc5',
}


def checked(lib, name, *args):
    status = getattr(lib, name)(*args)
    if status:
        raise RuntimeError(f'{name}: status {status}')


class Probe:
    def __init__(self, nvrtc_path, device_index, kind):
        self.nv, self.cu = C.CDLL(str(nvrtc_path.resolve())), C.CDLL('libcuda.so.1')
        self.program, self.context, self.module, self.function = [C.c_void_p() for _ in range(4)]
        self.device = C.c_int()
        self.dx, self.dy = C.c_uint64(), C.c_uint64()
        self.retained = False
        try:
            checked(self.cu, 'cuInit', 0)
            checked(self.cu, 'cuDeviceGet', C.byref(self.device), device_index)
            major, minor = C.c_int(), C.c_int()
            checked(self.cu, 'cuDeviceComputeCapability', C.byref(major), C.byref(minor), self.device)
            if (major.value, minor.value) != (12, 0):
                raise ValueError('Pinned calibration requires compute capability 12.0 (sm_120)')
            checked(self.cu, 'cuDevicePrimaryCtxRetain', C.byref(self.context), self.device)
            self.retained = True
            checked(self.cu, 'cuCtxSetCurrent', self.context)
            operation = 'y[i]=rsqrtf(x[i]);' if kind == 'rsqrt' else 'asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y[i]) : "f"(x[i]));'
            self.source = ('extern "C" __global__ void probe(const float * x,float * y,int n){'
                           'int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){' + operation + '}}').encode()
            checked(self.nv, 'nvrtcCreateProgram', C.byref(self.program), self.source, b'probe.cu', 0, None, None)
            options = (C.c_char_p * 1)(b'--gpu-architecture=sm_120')
            status = self.nv.nvrtcCompileProgram(self.program, 1, options)
            if status:
                size = C.c_size_t()
                checked(self.nv, 'nvrtcGetProgramLogSize', self.program, C.byref(size))
                log = C.create_string_buffer(size.value)
                checked(self.nv, 'nvrtcGetProgramLog', self.program, log)
                raise RuntimeError(log.value.decode())
            size = C.c_size_t()
            checked(self.nv, 'nvrtcGetCUBINSize', self.program, C.byref(size))
            cubin = C.create_string_buffer(size.value)
            checked(self.nv, 'nvrtcGetCUBIN', self.program, cubin)
            self.cubin = cubin.raw
            checked(self.cu, 'cuModuleLoadData', C.byref(self.module), cubin)
            checked(self.cu, 'cuModuleGetFunction', C.byref(self.function), self.module, b'probe')
            self.count = 1 << 23
            for pointer in (self.dx, self.dy):
                checked(self.cu, 'cuMemAlloc_v2', C.byref(pointer), C.c_size_t(self.count * 4))
        except BaseException:
            self.close()
            raise

    def run(self, inputs):
        assert inputs.dtype == np.float32 and inputs.size == self.count and inputs.flags.c_contiguous
        output = np.empty_like(inputs)
        checked(self.cu, 'cuMemcpyHtoD_v2', self.dx, C.c_void_p(inputs.ctypes.data), C.c_size_t(inputs.nbytes))
        count = C.c_int(inputs.size)
        arguments = (C.c_void_p * 3)(C.addressof(self.dx), C.addressof(self.dy), C.addressof(count))
        checked(self.cu, 'cuLaunchKernel', self.function, (inputs.size + 255) // 256, 1, 1,
                256, 1, 1, 0, None, arguments, None)
        checked(self.cu, 'cuCtxSynchronize')
        checked(self.cu, 'cuMemcpyDtoH_v2', C.c_void_p(output.ctypes.data), self.dy, C.c_size_t(output.nbytes))
        return output

    def close(self):
        # Best-effort cleanup also runs when initialization or compilation fails.
        for pointer in (self.dx, self.dy):
            if pointer.value:
                self.cu.cuMemFree_v2(pointer)
                pointer.value = 0
        if self.module.value:
            self.cu.cuModuleUnload(self.module)
            self.module.value = None
        if self.retained:
            self.cu.cuDevicePrimaryCtxRelease_v2(self.device)
            self.retained = False
        if self.program.value:
            self.nv.nvrtcDestroyProgram(C.byref(self.program))


def generate(kind, args):
    root = args.output / ('rsqrt-mantissa-table' if kind == 'rsqrt' else 'exp2-fraction-table')
    root.mkdir(parents=True, exist_ok=False)
    path = root / f'{kind}-sm120-corrections.bin'
    probe = Probe(args.nvrtc, args.device, kind)
    rows = []
    try:
        (root / 'probe.cu').write_bytes(probe.source)
        (root / 'probe.cubin').write_bytes(probe.cubin)
        with path.open('wb') as stream:
            for exponent in (range(0, 2) if kind == 'rsqrt' else range(-24, 0)):
                first = (exponent + 127) << 23
                inputs = np.arange(first, first + (1 << 23), dtype=np.uint32).view(np.float32)
                actual = probe.run(inputs)
                baseline = (1 / np.sqrt(inputs.astype(np.float64)) if kind == 'rsqrt'
                            else np.exp2(inputs.astype(np.float64))).astype(np.float32)
                delta = actual.view(np.int32).astype(np.int64) - baseline.view(np.int32)
                if delta.min() < -2 or delta.max() > 1:
                    raise ValueError(f'{kind}: instruction differences do not fit pinned packing')
                codes = (delta + 2).astype(np.uint8).reshape(-1, 4)
                packed = codes[:, 0] | codes[:, 1] << 2 | codes[:, 2] << 4 | codes[:, 3] << 6
                stream.write(packed.tobytes())
                rows.append(dict(exponent=exponent, counts={str(i): int(np.count_nonzero(delta == i)) for i in (-2, -1, 0, 1)}))
                print(kind, exponent, flush=True)
    finally:
        probe.close()
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = dict(format=f'motionbricks-{kind}-correction-table-v1', architecture='sm_120',
        sha256=sha, pinned_sha256=PINNED[kind], matches_pin=sha == PINNED[kind], bytes=path.stat().st_size,
        input_interval='[1,4)' if kind == 'rsqrt' else '[2^-24,1)',
        baseline='float32(1/sqrt(float64(input)))' if kind == 'rsqrt' else 'float32(exp2(float64(input)))',
        packing='four 2-bit delta+2 codes per byte, least significant first', rows=rows,
        source_sha256=hashlib.sha256(probe.source).hexdigest(), cubin_sha256=hashlib.sha256(probe.cubin).hexdigest(),
        scope='hardware instruction calibration, not model weights')
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if not manifest['matches_pin']:
        raise ValueError(f'{kind}: generated checksum differs from the pinned calibration; do not use it')
    print(kind, sha, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nvrtc', type=Path, required=True, help='Path to libnvrtc.so.13')
    parser.add_argument('--output', type=Path, required=True, help='Fresh output root; existing table directories are rejected')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--kind', choices=('both', 'rsqrt', 'exp2'), default='both')
    args = parser.parse_args()
    for kind in ('rsqrt', 'exp2') if args.kind == 'both' else (args.kind,):
        generate(kind, args)


if __name__ == '__main__':
    main()
