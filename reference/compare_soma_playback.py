#!/usr/bin/env python3
"""Measure native pose-to-pose playback against independently captured upstream playback."""
import argparse
import json
from pathlib import Path
import numpy as np
from compare_native_taps import read_gguf
from compare_soma import summary


def transform_errors(reference, actual):
    if reference.shape != actual.shape or reference.ndim != 3 or reference.shape[-1] != 7:
        raise ValueError('Incompatible playback transform shapes')
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError('Nonfinite playback transforms')
    ref, got = reference.astype(np.float64), actual.astype(np.float64)
    distance = np.linalg.norm(ref[..., :3]-got[..., :3], axis=-1)
    a, b = ref[..., 3:], got[..., 3:]
    na, nb = np.linalg.norm(a, axis=-1, keepdims=True), np.linalg.norm(b, axis=-1, keepdims=True)
    if np.any(na < 1e-8) or np.any(nb < 1e-8):
        raise ValueError('Degenerate playback quaternion')
    dot = np.abs(np.sum((a/na)*(b/nb), axis=-1))
    angle = np.rad2deg(2*np.arccos(np.clip(dot, 0, 1)))
    return distance, angle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True, help='upstream playback capture directory')
    parser.add_argument('--native', type=Path, required=True, help='native runner output GGUF')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.reference/'manifest.json').read_text())
    metadata, native = read_gguf(args.native)
    reference = np.load(args.reference/'tensors.npz')
    length_matches = manifest['frames'] == metadata['soma.frames']
    frames = min(manifest['frames'], metadata['soma.frames'])
    joints = metadata['soma.joints']
    report = dict(source_revision=manifest['source_revision'], reference=str(args.reference),
                  native=str(args.native), backend=metadata['backend'],
                  duration_matches=length_matches, reference_tokens=manifest['tokens'],
                  native_tokens=metadata['soma.tokens'], compared_frames=frames, stages={})
    for name, key in [('decoded', 'decoded_local'), ('corrected', 'corrected_local'), ('world', 'corrected_world')]:
        actual = native[name].reshape(-1, joints, 7)[:frames]
        expected = reference[key][:frames]
        distance, angle = transform_errors(expected, actual)
        report['stages'][name] = dict(position_m=summary(distance), rotation_deg=summary(angle),
            root_position_m=summary(distance[:, 0]), per_frame=dict(root_position_m=distance[:, 0].tolist(),
            max_joint_position_m=distance.max(axis=1).tolist(), max_joint_rotation_deg=angle.max(axis=1).tolist()))
    report['contacts_absolute'] = summary(np.abs(reference['contacts'][:frames]-native['contacts'].reshape(-1,4)[:frames]))
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key!='stages'}, indent=2))
    for key, value in report['stages'].items():
        print(key, 'max position (m):', value['position_m']['max'], 'max rotation (deg):', value['rotation_deg']['max'])
    if not length_matches:
        raise SystemExit('Duration mismatch: common-prefix errors do not establish parity')


if __name__ == '__main__':
    main()
