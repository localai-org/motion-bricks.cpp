#!/usr/bin/env python3
"""Compare native root-selection taps to actual pinned upstream function locals."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from compare_native_taps import read_gguf


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--native',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    manifest=json.loads((args.capture/'manifest.json').read_text())
    pinned=json.loads(Path(__file__).with_name('soma-reference.json').read_text())
    if manifest['reference']!=pinned or manifest['precision']!='fp32':
        raise ValueError('Requires pinned FP32 decision capture')
    def digest(path):
        with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    if digest(args.capture/'tensors.npz')!=manifest['capture_sha256']:
        raise ValueError('Reference checksum mismatch')
    metadata,native=read_gguf(args.native)
    if metadata['precision']!='fp32':raise ValueError('Requires native FP32 output')
    values={};errors={};input_errors={}
    with np.load(args.capture/'tensors.npz') as data:
        for name in ('global_root_values','local_root_values','local_poses'):
            a=native['input.'+name].reshape(-1);b=data['input/'+name].reshape(-1)
            if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError('Invalid input correspondence')
            input_errors[name]=float(np.max(np.abs(a.astype(float)-b.astype(float))))
        if max(input_errors.values())>2e-5:raise ValueError('Native inputs differ from this reference fixture')
        for name,key in [('pose_height_error','avg_height_error_from_pose'),
                         ('root_height_error','avg_height_error_from_root'),('use_root','use_root_module')]:
            expected=data['decision/root_selection/0/'+key].reshape(-1)
            actual=native['root_selection.'+name].reshape(-1)
            if expected.shape!=(1,) or actual.shape!=(1,):raise ValueError('Expected one decision')
            # No target constraints gives upstream 0/0, with a false comparison.
            if np.isnan(expected[0]) or np.isnan(actual[0]):
                if name=='use_root' or not (np.isnan(expected[0]) and np.isnan(actual[0])):
                    raise ValueError('Unexpected nonfinite root decision')
                values[name]={'reference':None,'native':None};errors[name]=0.
            else:
                if not np.isfinite(expected).all() or not np.isfinite(actual).all():
                    raise ValueError('Infinite root decision')
                values[name]={'reference':float(expected[0]),'native':float(actual[0])}
                errors[name]=float(abs(expected.astype(float)[0]-actual.astype(float)[0]))
    passed=errors['use_root']==0 and max(errors['pose_height_error'],errors['root_height_error'])<=1e-5
    report=dict(format='motionbricks-soma-root-selection-parity-v1',passed=passed,values=values,
                absolute_errors=errors,height_error_tolerance_m=1e-5,
                input_absolute_errors=input_errors,
                capture_sha256=manifest['capture_sha256'],native_sha256=digest(args.native),
                source_revision=pinned['source_revision'],backend=metadata['backend'])
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(('PASS' if passed else 'FAIL'),json.dumps(values))
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
