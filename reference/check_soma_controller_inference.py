#!/usr/bin/env python3
"""Replay every independent interactive generation through the native public API."""
import argparse
import json
from pathlib import Path
import numpy as np
import warp as wp
from check_soma_api_chain import SomaAPI, digest
from compare_soma_playback import transform_errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('capture', 'library', 'bundle', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--backend', choices=('cpu','vulkan'), default='cpu')
    a = p.parse_args()
    manifest = json.loads((a.capture/'manifest.json').read_text())
    assets = json.loads((a.capture/'controller-assets.json').read_text())
    if digest(a.capture/'capture.npz') != manifest['capture_sha256']:
        raise ValueError('Capture checksum mismatch')
    precision=manifest.get('oracle',{}).get('precision','fp32')
    if precision=='mixed-fp16' and a.backend!='vulkan':
        p.error('Interactive mixed-FP16 replay currently requires the qualified Vulkan mode')
    data = np.load(a.capture/'capture.npz')
    contract = json.loads(Path(__file__).with_name('soma-controller-tolerances.json').read_text())
    limits = contract['inference_replay']
    legacy_limits = json.loads(Path(__file__).with_name('soma-chain-tolerances.json').read_text())
    inverse = wp.transform_inverse(wp.transform(*assets['model_to_runtime']))
    def model_space(poses):
        result = poses.copy()
        for pose in result:
            pose[0] = np.array(wp.mul(inverse, wp.transform(*pose[0])))
        return result
    api = SomaAPI(a.library, a.bundle, a.backend)
    api.set_skeleton(dict(parents=data['parents'], reference=data['reference'],
        model_to_pose=data['model_to_pose'], heading_joints=[assets['names'].index(n) for n in ('RightLeg','LeftLeg')]))
    rows=[]
    try:
        for i, event in enumerate(manifest['events']):
            prefix=f'event_{i:04d}_'
            context, target = model_space(data[prefix+'context']), model_space(data[prefix+'target'])
            allowed, heading = sum(1 << (n-6) for n in event['allowed']), np.arctan2(*event['heading'])
            before = api.generate(context,target,0,allowed,heading) if precision=='mixed-fp16' else None
            actual=api.generate(context,target,3 if precision=='mixed-fp16' else 0,allowed,heading)
            restored = True
            repeatable = True
            if before is not None:
                after = api.generate(context,target,0,allowed,heading)
                restored = all(np.array_equal(before[k],after[k]) for k in before)
                repeated = api.generate(context,target,3,allowed,heading)
                repeatable = all(np.array_equal(actual[k],repeated[k]) for k in actual)
            expected=model_space(data[prefix+'motion'])
            count=min(len(expected), len(actual['corrected']))
            distance, angle=transform_errors(expected[:count], actual['corrected'][:count])
            contact=float(np.max(np.abs(data[prefix+'contacts'][:count]-actual['contacts'][:count])))
            row=dict(index=i,tick=event['tick'],tokens_match=actual['tokens']==event['tokens'],
                position_m=float(distance.max()), rotation_deg=float(angle.max()), contacts_absolute=contact,
                contact_classification_match=bool(np.array_equal(data[prefix+'contacts'][:count]>=.5,actual['contacts'][:count]>=.5)))
            row['fp32_restored']=restored
            row['repeatable']=repeatable
            row['passed']=bool(restored and repeatable and row['tokens_match'] and row['contact_classification_match'] and row['position_m']<=limits['position_m'] and row['rotation_deg']<=limits['rotation_deg'] and contact<=limits['contacts_absolute'])
            row['legacy_contact_limit_passed']=contact<=legacy_limits['contacts_absolute']
            rows.append(row)
            print(row,flush=True)
    finally:
        api.close()
    result=dict(format='motionbricks-controller-inference-parity-v1', source_revision=manifest['source_revision'],
        capture_sha256=manifest['capture_sha256'], library_sha256=digest(a.library), backend=a.backend,
        precision=precision, fp32_scope_restoration_checked=precision=="mixed-fp16", limits=limits, events=rows, passed=all(r['passed'] for r in rows))
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    raise SystemExit(0 if result['passed'] else 1)


if __name__=='__main__': main()
