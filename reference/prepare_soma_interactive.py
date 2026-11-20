#!/usr/bin/env python3
"""Export the pinned upstream locomotion targets for the shared demo controller."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.upstream.resolve()
    scripts = Path(__file__).resolve().parent
    pin = json.loads((scripts / 'soma-reference.json').read_text())['source_revision']
    targets = [('idle', 'assets/motions/soma/locomotion/idle/idle_loop_001__A268.bvh', 1.0)]
    documents = [
        ('locomotion_bs1_linear', {'Walk': 'walk', 'Jog': 'jog', 'Run': 'run'}),
        ('locomotion_bs2_directional_crouch', {'Forward': 'crouch'}),
        ('locomotion_style_variations_bs1_linear', {
            'Skipping': 'skipping', 'Vogue': 'vogue', 'Injured Torso': 'injured_torso',
            'Injured Leg': 'injured_leg', 'Zombie': 'walk_zombie', 'Combat': 'combat'}),
    ]
    for document, names in documents:
        path = Path('assets/smart_primitives/soma') / (document + '.smart_primitive.json')
        data = json.loads(subprocess.check_output(['git', '-C', str(root), 'show', f'{pin}:{path}']))
        found = set()
        for node in data['root']['children']:
            if node['name'] not in names:
                continue
            component, = [c for c in node['components'] if c['type'] == 'target_pose']
            motion = (root / path.parent / component['clip']).resolve().relative_to(root)
            targets.append((names[node['name']], str(motion), component['sample_time_seconds']))
            found.add(node['name'])
        if found != set(names):
            raise ValueError(f'Missing upstream targets: {set(names) - found}')
    export = [sys.executable, str(scripts / 'export_soma_interactive.py')]
    for name, motion, time in targets:
        poses = args.output.parent / (args.output.stem + '-poses') / name
        subprocess.run([sys.executable, str(scripts / 'prepare_soma_inputs.py'),
                        '--upstream', str(root), '--motion', motion,
                        '--target-time', str(time), '--output', str(poses)], check=True)
        export += ['--style', f'{name}={poses}']
    subprocess.run(export + ['--output', str(args.output)], check=True)


if __name__ == '__main__':
    main()
