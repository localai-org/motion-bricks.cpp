#!/usr/bin/env python3
"""Convert independently captured upstream boundaries to native GGUF fixtures."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from convert_to_gguf import sha256, write_component


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.capture / 'manifest.json').read_text())
    pinned = json.loads(Path(__file__).with_name('soma-reference.json').read_text())
    if manifest['reference'] != pinned:
        raise ValueError('Fixture capture is not from the pinned SOMA reference')
    archive = args.capture / 'tensors.npz'
    if manifest['capture_sha256'] != sha256(archive) or manifest['precision'] != 'fp32':
        raise ValueError('Fixture source must be an intact FP32 capture')
    prefix = 'layer/model._mmm_pose_model.mmm_net/0/'
    mapping = {'input.pose_tokens': 'input/0', 'input.root_values': 'input/1',
               'input.pose_cond': 'input/2', 'input.has_pose_cond': 'input/3',
               'input.num_tokens': 'input/4', 'output.pose_logits': 'output/pose_logits'}
    with np.load(archive, allow_pickle=False) as data:
        values = [(name, data[prefix + key]) for name, key in mapping.items()]
        for name in ('global_root_values', 'local_root_values', 'local_poses',
                     'has_global_root_values', 'has_local_root_values', 'has_local_poses', 'allowed_pred_num_tokens'):
            values.append(('full.' + name, data['input/' + name]))
        values.append(('full.output', data['output/pred_global_motions']))
        for name,key in [('use_root','use_root_module'),('pose_error','avg_height_error_from_pose'),
                         ('root_error','avg_height_error_from_root')]:
            source='decision/root_selection/0/'+key
            if source in data:values.append(('decision.'+name,data[source]))
        root_prefix = 'layer/model._mmm_root_model.mmm_net/0/'
        for index, name in enumerate(('global', 'global_mask', 'local', 'local_mask', 'poses', 'pose_mask')):
            values.append(('root.' + name, data[root_prefix + 'input/' + str(index)]))
        values.append(('root.logits', data['layer/model._mmm_root_model.mmm_net._proj_num_token_output_logit/0/output']))
        values.append(('root.motion', data[root_prefix + 'output/pred_global_root_values']))
        decoder = 'layer/model._vqvae_pose_model.decoder/0/'
        count = int(data[prefix + 'input/4'].reshape(-1)[0])
        if data[prefix + 'input/4'].size != 1 or not 6 <= count <= 16:
            raise ValueError('Initial native fixture runner requires one request with 6–16 tokens')
        values.extend([
            ('decoder.quantized', data[decoder + 'input/0'][:, :, :count].copy()),
            ('decoder.external', data[decoder + 'input/1'][:, :count*4].copy()),
            ('decoder.target', data[decoder + 'input/2'][:, :count*4].copy()),
            ('decoder.mask', data[decoder + 'input/3'][:, :count*4].copy()),
            ('decoder.output', data[decoder + 'output'][:, :, :count*4].transpose(0, 2, 1).copy()),
        ])
    write_component(args.output / 'pose.gguf', 'pose-fixture', values, sha256(archive),
                    general_name='Independent upstream SOMA pose boundary', skeleton='soma30',
                    upstream_revision=manifest['reference']['source_revision'])


if __name__ == '__main__':
    main()
