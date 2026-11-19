#!/usr/bin/env python3
"""Export verified upstream pose fixtures as interactive SOMA target styles.

Each --style is NAME=POSE_DIRECTORY from prepare_soma_inputs.py. Only poses,
skeleton data and provenance are exported; interactive motion is generated live.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--style',action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); output=None
    tuning=json.loads(Path(__file__).with_name('soma-interactive-tuning.json').read_text())
    for spec in args.style:
        name,path=spec.split('=',1); path=Path(path)
        manifest=json.loads((path/'manifest.json').read_text())
        data=np.load(path/'poses.npz',allow_pickle=False)
        names=manifest['joint_names']; right=names.index('RightLeg');left=names.index('LeftLeg')
        skeleton=dict(names=names,parents=data['parents'].tolist(),reference=data['reference_local'].reshape(-1).tolist(),
                      mapping=data['model_to_pose'].tolist(),right_leg=right,left_leg=left)
        if output is None:
            output=dict(format='motionbricks-soma-interactive-v1',skeleton=skeleton,styles=[],source_revision=manifest['source_revision'])
        elif output['skeleton']!=skeleton or output['source_revision']!=manifest['source_revision']:
            raise ValueError('Styles must use the same pinned skeleton and source')
        local=data['local'][4:].copy();world=data['world'][4:]
        # Hip lateral vector defines forward in the horizontal plane.
        lateral=world[0,right,:3]-world[0,left,:3]
        heading=float(np.arctan2(lateral[2],-lateral[0]))
        speed=float(np.linalg.norm(world[-1,0,[0,2]]-world[0,0,[0,2]])*10)
        settings=tuning['styles'].get(name,dict(speed=speed,velocity_scale=1,target_horizon=1))
        if manifest['source_revision']!=tuning['source_revision']:
            raise ValueError('Style tuning does not match pinned source')
        output['styles'].append(dict(name=name,**settings,
            heading=heading,local=local.reshape(-1).tolist(),world=world[0,:,:3].reshape(-1).tolist(),
            source_motion=manifest['motion'],source_sha256=manifest['motion_sha256'],
            source_times=manifest['times'][4:]))
    if not any(s['name']=='idle' for s in output['styles']):raise ValueError('An idle style is required')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,separators=(',',':'))+'\n')

if __name__=='__main__':main()
