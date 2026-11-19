#!/usr/bin/env python3
"""Convert the pinned NVIDIA SOMA SafeTensors to native FP32 GGUF components.

The bundle includes feature-space networks and normalization/T-pose support
tensors for the dedicated SOMA API. Applications supply the pose skeleton when
creating a SOMA skeleton handle.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from convert_to_gguf import load_selected, write_component

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reference'))
from fetch_soma import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True, help='fetch_soma.py output directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mixed-onnx', type=Path, help='Pinned official mixed-FP16 export; preserve its folded normalization scales')
    args = parser.parse_args()
    manifest = json.loads((Path(__file__).resolve().parents[1] / 'reference/soma-reference.json').read_text())
    entries = manifest['checkpoints'] + [next(x for x in manifest['source_assets']
                                            if x['path'].endswith('soma_model_rig.npz'))]
    for entry in entries:
        if not verify(args.reference / entry['path'], entry):
            raise ValueError(f"Missing or corrupt reference artifact: {entry['path']}")
    components = {}
    expected = {'pose': (209, 136520976), 'root': (174, 40418385), 'vq-decoder': (51, 12202221)}
    for role, count in expected.items():
        marker = '/vqvae-' if role == 'vq-decoder' else '-' + role + '-'
        entry = next(x for x in entries if marker in x['path'] and x['path'].endswith('.safetensors'))
        if role == 'vq-decoder':
            select = lambda n: n.startswith('pose_net.decoder.') or n == 'pose_net.quantizer.vq._codebook.embed'
            prefix = 'pose_net.'
        else:
            select = lambda n: n.startswith('mmm_net.') and n != 'mmm_net.initted'
            prefix = 'mmm_net.'
        values = load_selected(args.reference / entry['path'], select, prefix)
        inventory = (len(values), sum(int(v.size) for _, v in values))
        if inventory != count:
            raise ValueError(f'{role} inventory {inventory} differs from pinned {count}')
        components[role] = write_component(args.output / (role + '.gguf'), role, values, entry['sha256'],
            general_name='NVIDIA MotionBricks SOMA', skeleton='soma30',
            upstream_revision=manifest['source_revision'], joint_names=())
    support = []
    for name in ('mean', 'std'):
        entry = next(x for x in entries if x['path'].endswith('/' + name + '.npy'))
        value = np.load(args.reference / entry['path'], allow_pickle=False)
        if value.shape != (370,) or not np.isfinite(value).all():
            raise ValueError(f'Unexpected SOMA normalization {name}')
        if name == 'std' and np.any(value < 0):
            raise ValueError('SOMA normalization standard deviations must be nonnegative')
        support.append(('motion_' + name, value.astype(np.float32)))
    mixed_entry = None
    if args.mixed_onnx:
        import onnx
        mixed_entry = next(x for x in manifest['source_assets'] if x['path'].endswith('mmm_v6_soma_fp16.onnx'))
        if not verify(args.mixed_onnx, mixed_entry):
            raise ValueError('Mixed ONNX checksum differs from the pinned export')
        model = onnx.load(args.mixed_onnx)
        constants = {node.name: onnx.numpy_helper.to_array(node.attribute[0].t)
                     for node in model.graph.node if node.op_type == 'Constant' and
                     node.attribute[0].type == onnx.AttributeProto.TENSOR}
        # Explicit global/local/body scales in the pinned normalization graph.
        scales = [constants[name] for name in ('/Constant_112', '/Constant_181', '/Constant_303')]
        if any(v.dtype != np.float32 or v.shape != (n,) for v,n in zip(scales,(5,4,361))):
            raise ValueError('Unexpected exported normalization scale layout')
        scale = np.concatenate(scales)
        std = dict(support)['motion_std']
        if not np.isfinite(scale).all() or np.any(scale <= 0) or not np.allclose(scale,np.sqrt(std*std+np.float32(1e-5)),rtol=2e-7,atol=0):
            raise ValueError('Export scales disagree with checkpoint statistics')
        support.append(('mixed_motion_scale', scale))
        entries.append(mixed_entry)
    rig_entry = next(x for x in entries if x['path'].endswith('soma_model_rig.npz'))
    with np.load(args.reference / rig_entry['path'], allow_pickle=False) as rig:
        for name in ('capture_offsets', 'standard_offsets'):
            value = rig[name]
            if value.shape != (30, 3, 3) or not np.isfinite(value).all():
                raise ValueError(f'Unexpected SOMA rig {name}')
            support.append((name, value.astype(np.float32)))
    provenance = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    components['support'] = write_component(args.output / 'support.gguf', 'support', support, provenance,
        general_name='NVIDIA MotionBricks SOMA feature support', skeleton='soma30',
        upstream_revision=manifest['source_revision'], joint_names=())
    output = dict(format='motionbricks-soma-feature-bundle-v1', precision='fp32',
                  upstream_revision=manifest['source_revision'], checkpoint_revision=manifest['checkpoint_revision'],
                  skeleton='soma30', joint_count=30, external_pose_width=267, internal_pose_width=268,
                  global_motion_width=366, local_motion_width=365, codebook_heads=5,
                  codes_per_head=16, code_dimension_per_head=32, components=components,
                  inference_parameter_count=sum(components[x]['parameter_count'] for x in expected))
    if mixed_entry: output['mixed_normalization_source_sha256'] = mixed_entry['sha256']
    (args.output / 'manifest.json').write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    main()
