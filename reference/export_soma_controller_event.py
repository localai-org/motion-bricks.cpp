#!/usr/bin/env python3
"""Export an independent controller event for encoder/layer diagnosis."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import warp as wp
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from convert_to_gguf import write_component,kv_f32,kv_u32,kv_string
from check_soma_api_chain import digest

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--capture',type=Path,required=True)
p.add_argument('--event',type=int,required=True)
p.add_argument('--native-event',type=Path,help='Optional recorded native controller context/target, in runtime space')
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
m=json.loads((a.capture/'manifest.json').read_text())
s=json.loads((a.capture/'controller-assets.json').read_text())
if digest(a.capture/'capture.npz')!=m['capture_sha256']:raise ValueError('Capture checksum mismatch')
d=np.load(a.capture/'capture.npz');e=m['events'][a.event];prefix=f'event_{a.event:04d}_'
a.output.mkdir(parents=True,exist_ok=False)
inputs={k[len(prefix+'input_'):]:d[k] for k in d.files if k.startswith(prefix+'input_')}
np.savez(a.output/'inputs.npz',**inputs)
inverse=wp.transform_inverse(wp.transform(*s['model_to_runtime']))
def model_space(poses):
    result=poses.copy()
    for pose in result:pose[0]=np.array(wp.mul(inverse,wp.transform(*pose[0])))
    return result
context=d[prefix+'context'];target=d[prefix+'target']
if a.native_event:
    feedback=json.loads(a.native_event.read_text())
    context=np.array(feedback['context'],np.float32).reshape(context.shape)
    target=np.array(feedback['target'],np.float32).reshape(target.shape)
mapping=d['model_to_pose'].astype(np.int32)
reverse=np.full(len(d['parents']),-1,dtype=np.int32)
for model,pose in enumerate(mapping):
    parent=d['parents'][pose]
    if parent==-1 or reverse[parent]>=0:reverse[pose]=model
values=[('parents',d['parents'].astype(np.int32)),('pose_to_model',reverse),('model_to_pose',mapping),
        ('reference',d['reference']),('context',model_space(context)),('target',model_space(target)),
        ('target_count',np.array([4],np.uint32)),('heading_joints',np.array([s['names'].index(n) for n in ('RightLeg','LeftLeg')],np.uint32))]
values.extend(('input.'+k,v) for k,v in inputs.items())
write_component(a.output/'input.gguf','playback-fixture',values,m['capture_sha256'],skeleton='soma30',upstream_revision=m['source_revision'],extra_metadata=[kv_f32('fixture.heading_angle',float(np.arctan2(*e['heading']))),kv_u32('fixture.allowed_token_bits',sum(1<<(n-6) for n in e['allowed'])),kv_string('fixture.capture_sha256',m['capture_sha256'])])
(a.output/'event.json').write_text(json.dumps(dict(event=a.event,**e),indent=2)+'\n')
