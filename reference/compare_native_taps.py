#!/usr/bin/env python3
"""Compare native graph snapshots in execution order, including strided views."""
import argparse
import json
from pathlib import Path
import struct
import numpy as np


def read_gguf(path):
    with Path(path).open('rb') as stream:
        def unpack(fmt):
            return struct.unpack('<' + fmt, stream.read(struct.calcsize('<' + fmt)))[0]
        def string():
            return stream.read(unpack('Q')).decode('utf8')
        def value(kind):
            if kind == 8:
                return string()
            if kind == 9:
                element, count = unpack('I'), unpack('Q')
                return [value(element) for _ in range(count)]
            return unpack({0:'B', 1:'b', 2:'H', 3:'h', 4:'I', 5:'i', 6:'f', 7:'?', 10:'Q', 11:'q', 12:'d'}[kind])
        if stream.read(4) != b'GGUF' or unpack('I') != 3:
            raise ValueError('Expected GGUF v3')
        count, fields = unpack('Q'), unpack('Q')
        metadata = {}
        for _ in range(fields):
            name = string()
            metadata[name] = value(unpack('I'))
        tensors = []
        for _ in range(count):
            name = string()
            shape = [unpack('Q') for _ in range(unpack('I'))]
            tensors.append((name, shape, unpack('I'), unpack('Q')))
        alignment = metadata.get('general.alignment', 32)
        start = (stream.tell() + alignment - 1) // alignment * alignment
    result = {}
    for name, shape, kind, offset in tensors:
        dtype = {0:'<f4', 1:'<f2', 26:'<i4'}[kind]
        result[name] = np.memmap(path, dtype=dtype, mode='r', offset=start+offset, shape=tuple(reversed(shape)))
    return metadata, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('actual', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rm, reference = read_gguf(args.reference)
    am, actual = read_gguf(args.actual)
    if reference.keys() != actual.keys():
        raise ValueError('Different graph inventories')
    rows = []
    for key, x in reference.items():
        y = actual[key]
        if x.shape != y.shape or rm[key+'.operation'] != am[key+'.operation'] or rm[key+'.name'] != am[key+'.name']:
            raise ValueError(f'Graph mismatch at {key}')
        finite = np.isfinite(x) & np.isfinite(y)
        if not np.all(finite | (x == y)):
            raise ValueError(f'Unexpected nonfinite values at {key}')
        error = np.abs(x[finite].astype(np.float64) - y[finite].astype(np.float64))
        rows.append(dict(tap=key, name=rm[key+'.name'], operation=rm[key+'.operation'],
                         shape=list(x.shape), max_absolute=float(error.max(initial=0)),
                         rms=float(np.sqrt(np.mean(error**2))) if error.size else 0))
    args.output.write_text(json.dumps(dict(reference=str(args.reference), actual=str(args.actual), taps=rows), indent=2)+'\n')
    for row in rows[:90]:
        print(row['tap'], row['operation'], row['max_absolute'])


if __name__ == '__main__':
    main()
