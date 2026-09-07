#!/usr/bin/env python3
"""Write a bounded native physics configuration from verified upstream data.

Gains/scales/defaults come from the actual instrumented controller, not the
simulator YAML's unused nominal gains. Geometry comes from the validated
MotionBricks G1 converter. Output contains numbers, never executable code.
"""
import argparse
import json
from pathlib import Path
import struct
import numpy as np
from export_sonic_motion import HingeConverter, read_motion, sha256

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--motion',type=Path,required=True)
    p.add_argument('--motion-xml',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): p.error('output exists')
    summary=json.loads((args.run/'summary.json').read_text())
    if summary['failure'] or summary['provenance']['source_revision']!='a0732b642c0333077e127a2f56ab0014c196bca4': raise ValueError('invalid upstream run')
    if sha256(args.motion_xml)!='5d76cf92f00dd49d6eb9fae38d7d38e46886848b602ac691051e886c3bcccfb1': raise ValueError('wrong G1 XML')
    obj,_,_=read_motion(args.motion)
    converter=HingeConverter(args.motion_xml,obj['metadata']['joints'])
    row=json.loads((args.run/'boundaries.jsonl').read_text().splitlines()[0])
    with np.load(args.run/'physics.npz',allow_pickle=False) as data:
        index=np.flatnonzero(data['time']>0)[10]
        kp,kd=data['kp'][index],data['kd'][index]
        limits=data['torque_limits'][data['body_actuator_indices']]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('wb') as f:
        f.write(b'MBSPHY01')
        for i in range(29):
            f.write(struct.pack('<I',converter.indices[i]))
            values=[*converter.axes[i],*converter.offsets[i].reshape(-1),row['default_angles'][i],row['action_scale'][i],kp[i],kd[i],limits[i]]
            f.write(struct.pack('<17d',*values))
    report=dict(schema=1,config_sha256=sha256(args.output),motion_xml_sha256=sha256(args.motion_xml),
                run_provenance=summary['provenance'],hardware_joint_names=converter.names,
                attribution='Numeric controller parameters from NVIDIA GR00T-WholeBodyControl original SONIC release; no policy source copied.')
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(args.output)

if __name__=='__main__': main()
