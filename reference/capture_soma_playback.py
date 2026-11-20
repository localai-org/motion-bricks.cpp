#!/usr/bin/env python3
"""Apply pinned upstream decoding and endpoint correction to captured SOMA features.

This is an oracle for native playback implementation. Applying it to native features
isolates neural error; it does not establish native playback parity.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from compare_native_taps import read_gguf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--poses', type=Path, required=True, help='prepare_soma_inputs output directory')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--capture', type=Path, help='upstream capture directory')
    source.add_argument('--native', type=Path, help='native-boundaries.gguf')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.upstream.resolve()
    pin = json.loads(Path(__file__).with_name('soma-reference.json').read_text())['source_revision']
    revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != pin:
        raise ValueError('Upstream checkout must match pinned source revision')
    changed = subprocess.check_output(['git', '-C', str(root), 'diff', 'HEAD', '--', 'motionbricks', ':!motionbricks/assets'], text=True)
    if changed:
        raise ValueError('Upstream runtime code has local modifications')
    pose_manifest = json.loads((args.poses / 'manifest.json').read_text())
    if pose_manifest['source_revision'] != pin:
        raise ValueError('Pose fixture revision differs')
    if args.native:
        metadata, tensors = read_gguf(args.native)
        tokens = int(metadata['soma.tokens'])
        features = np.array(tensors['output_features']).reshape(1, 4*tokens, 366)
    else:
        capture_manifest=json.loads((args.capture/'manifest.json').read_text())
        if capture_manifest['reference']['source_revision']!=pin:
            raise ValueError('Feature capture revision differs')
        with (args.capture/'tensors.npz').open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=capture_manifest['capture_sha256']:
                raise ValueError('Feature capture checksum mismatch')
        with np.load(args.capture / 'tensors.npz') as data:
            tokens = int(data['output/pred_num_tokens'].item())
            features = data['output/pred_global_motions'].copy()
    if not 6 <= tokens <= 16 or not np.isfinite(features).all():
        raise ValueError('Invalid SOMA output')
    with np.load(args.poses / 'poses.npz') as data:
        poses = {key: data[key] for key in data.files}
    sys.path.insert(0, str(root))
    import warp as wp
    wp.config.kernel_cache_dir = str(Path('.cache/soma-warp').resolve())
    wp.init(); wp.set_device('cpu')
    from motionbricks.model.soma import kernels
    from motionbricks.utils.pose_utils import compute_world_poses
    def array(values, dtype):
        return wp.array(np.ascontiguousarray(values), dtype=dtype, device='cpu')
    joints = len(poses['parents'])
    inverse = np.full(joints, -1, dtype=np.int32)
    if poses['model_to_pose'][0] != 0:
        raise ValueError('Upstream SOMA engine requires Hips at skeleton index zero')
    for model, pose in enumerate(poses['model_to_pose']):
        parent = poses['parents'][pose]
        if parent == -1 or inverse[parent] >= 0:
            inverse[pose] = model
    count = 4*tokens-6
    local = wp.zeros((count, joints), dtype=wp.transform, device='cpu')
    contacts = wp.zeros((count, 4), dtype=wp.float32, device='cpu')
    frame_count = wp.zeros(1, dtype=wp.int32, device='cpu')
    parents = array(poses['parents'], wp.int32)
    wp.launch(kernels.decode_motionbricks_output_kernel, dim=count,
              inputs=[30, joints, parents, array(inverse, wp.int32),
                      array(poses['reference_local'], wp.transform), array(poses['alignment'], wp.transform),
                      wp.transform_identity(), array(features, wp.float32), array([tokens], wp.int32)],
              outputs=[local, contacts, frame_count], device='cpu')
    decoded = local.numpy().copy()
    if int(frame_count.numpy()[0]) != count:
        raise ValueError('Unexpected decoded frame count')
    start = wp.zeros(joints, dtype=wp.transform, device='cpu')
    end = wp.zeros_like(start)
    wp.launch(kernels.compute_endpoint_deltas_kernel, dim=joints,
              inputs=[frame_count, array(poses['local'][:4], wp.transform),
                      array(poses['local'][4:], wp.transform), local], outputs=[start, end], device='cpu')
    wp.launch(kernels.apply_endpoint_deltas_kernel, dim=(count, joints),
              inputs=[frame_count, start, end], outputs=[local], device='cpu')
    world = wp.empty_like(local)
    compute_world_poses(local, parents, world)
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez(args.output / 'tensors.npz', decoded_local=decoded, start_deltas=start.numpy(),
             end_deltas=end.numpy(), corrected_local=local.numpy(), corrected_world=world.numpy(),
             contacts=contacts.numpy())
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from convert_to_gguf import write_component, sha256, kv_string
    values = [('features', features[:, :tokens*4].copy()),
              ('tokens', np.array([tokens], dtype=np.int32)),
              ('parents', poses['parents']), ('pose_to_model', inverse),
              ('reference', poses['reference_local']), ('alignment', poses['alignment']),
              ('context', poses['local'][:4].copy()), ('target', poses['local'][4:].copy()),
              ('decoded', decoded), ('corrected', local.numpy()), ('world', world.numpy()),
              ('start', start.numpy()), ('end', end.numpy()), ('contacts', contacts.numpy())]
    values.extend([('model_to_pose', poses['model_to_pose']), ('input_world', poses['world']),
                   ('target_count', np.array([pose_manifest['target_poses']], dtype=np.int32)),
                   ('heading_joints', np.array([pose_manifest['joint_names'].index(name)
                                               for name in ('RightLeg', 'LeftLeg')], dtype=np.int32))])
    with np.load(args.poses / 'inputs.npz') as data:
        values.extend(('input.'+key, data[key]) for key in data.files)
    provenance=[] if args.native else [kv_string('fixture.capture_sha256',capture_manifest['capture_sha256'])]
    write_component(args.output / 'playback.gguf', 'playback-fixture', values,
                    sha256(args.output / 'tensors.npz'), skeleton='soma30', upstream_revision=pin,
                    extra_metadata=provenance)
    report = dict(source_revision=pin, frames=count, tokens=tokens,
                  feature_source=str(args.native or args.capture), pose_source=str(args.poses),
                  playback_implementation='upstream-warp-cpu', joint_names=pose_manifest['joint_names'])
    (args.output / 'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'Captured upstream playback: {count} frames, {joints} joints -> {args.output}')


if __name__ == '__main__':
    main()
