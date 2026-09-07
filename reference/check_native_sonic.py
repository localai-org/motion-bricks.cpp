#!/usr/bin/env python3
"""Independent native live-loop boundary check, not a rollout equality claim.

Rebuild the live reference using the already upstream-validated Python hinge
converter, then inspect actual C++ observation/history/action/PD buffers.
Run the official unmodified ONNX on the same observations. Physics is free to
diverge from asynchronous upstream DDS runs; same-state inference is not.
"""
import argparse
import json
from pathlib import Path
import struct

import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation, Slerp
from convert_sonic_gguf import load_models
from export_sonic_motion import HingeConverter, HARDWARE_TO_ISAAC as order, read_motion, physical_fk, sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['recording','motion','motion-xml','scene','config','policy','output']:
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('output exists')
    run=json.loads(args.recording.read_text())
    for name in ['motion','scene','config']:
        if run['inputs'][name]['sha256']!=sha256(getattr(args,name)):raise ValueError('recording input hash mismatch: '+name)
    obj,roots,quats=read_motion(args.motion)
    # The public C ABI receives F32, not the JSON decimal double intermediates.
    roots=np.asarray(obj['roots'],np.float32).reshape(-1,3).astype(float)
    quats=np.asarray(obj['rotations'],np.float32).reshape(-1,34,4).astype(float)
    converter=HingeConverter(args.motion_xml,obj['metadata']['joints'])
    pose,_=converter.convert(roots,quats)
    source=np.arange(len(roots))/30
    unwrapped=np.unwrap(pose[:,7:],axis=0)
    slerp=Slerp(source,Rotation.from_quat(pose[:,[4,5,6,3]]))
    def sample(times):
        t=np.clip(times,0,source[-1])
        r=np.array([np.interp(t,source,pose[:,k]) for k in range(3)]).T
        q=slerp(t).as_quat()[:,[3,0,1,2]]
        angles=np.array([np.interp(t,source,unwrapped[:,k]) for k in range(29)]).T
        return np.concatenate([r,q,angles],axis=1)
    cfg=args.config.read_bytes()
    if cfg[:8]!=b'MBSPHY01' or len(cfg)!=4068:raise ValueError('config format')
    values=np.array([struct.unpack_from('<17d',cfg,12+140*j) for j in range(29)])
    defaults,scales,kp,kd,limits=values[:,12:].T
    options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
    options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sessions={key:ort.InferenceSession(model.SerializeToString(),sess_options=options,providers=['CPUExecutionProvider']) for key,model in load_models(args.policy).items()}
    checks={}
    def check(name,a,b,atol=3e-6,rtol=None):
        a,b=np.asarray(a),np.asarray(b)
        if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise AssertionError(name+': shape/nonfinite')
        error=float(np.max(np.abs(a-b)))
        relative=float(np.linalg.norm(a-b)/max(float(np.linalg.norm(b)),1e-12))
        item=checks.setdefault(name,dict(max_abs=0.,relative_l2=0.,abs_tolerance=atol,relative_tolerance=rtol))
        item['max_abs']=max(item['max_abs'],error);item['relative_l2']=max(item['relative_l2'],relative)
        if error>atol or (rtol is not None and relative>rtol):raise AssertionError(f'{name} at {index}: abs={error}, relative={relative}')
    history=[];last=np.zeros(29,np.float32)
    rows=run['rows']
    reference_pose=sample(np.arange(1,len(rows)+1)/50)
    expected_positions=physical_fk(args.scene,converter.names,reference_pose)[0][...,[1,2,0]]
    for index,row in enumerate(rows):
        enc,dec,actions,targets,state,pd=[np.array(a) for a in row['trace']]
        q,dq,base,angular=state[:29],state[29:58],state[58:62],state[62:65]
        base_rotation=Rotation.from_quat(base.astype(np.float32)[[1,2,3,0]])
        h=dict(q=(q.astype(np.float32).astype(float)-defaults)[order],dq=dq.astype(np.float32)[order],angular=angular.astype(np.float32),action=last.copy(),gravity=base_rotation.inv().apply([0,0,-1]))
        history.append(h)
        future=index/50+np.arange(10)*.1
        current,next_pose=sample(future),sample(future+.02)
        wanted_enc=np.zeros(1762,np.float32)
        # Wrap interpolated hinge branches into the current interval, as C++
        # does per segment. Differences in 2*pi coordinates are not policy-equivalent.
        for array,times in [(current,future),(next_pose,future+.02)]:
            frame=np.clip((times*30).astype(int),0,len(pose)-1)
            array[:,7:]+=np.round((pose[frame,7:]-array[:,7:])/(2*np.pi))*2*np.pi
        wanted_enc[4:294]=current[:,7:][:,order].reshape(-1)
        wanted_enc[294:584]=((next_pose[:,7:]-current[:,7:])*50)[:,order].reshape(-1)
        ref_rotation=Rotation.from_quat(current[:,[4,5,6,3]])
        wanted_enc[601:661]=(base_rotation.inv()*ref_rotation).as_matrix()[...,:2].reshape(-1)
        check('encoder_live_reference',enc,wanted_enc,1e-5)
        zero=dict(q=np.zeros(29),dq=np.zeros(29),angular=np.zeros(3),action=np.zeros(29),gravity=[0,0,1])
        past=([zero]*max(0,10-len(history))+history[-10:])
        wanted_dec=np.concatenate([dec[:64],*[np.array([h[key] for h in past]).reshape(-1) for key in ['angular','q','dq','action','gravity']]]).astype(np.float32)
        check('decoder_state_history',dec,wanted_dec,3e-6)
        tokens=sessions['encoder'].run(None,{'obs_dict':enc.astype(np.float32)[None]})[0].reshape(-1)
        check('tokens_exact',dec[:64],tokens,0)
        wanted_actions=sessions['decoder'].run(None,{'obs_dict':dec.astype(np.float32)[None]})[0].reshape(-1)
        check('actions_same_state',actions,wanted_actions,5e-4,1e-5)
        check('motor_targets',targets,(defaults+scales*actions[np.argsort(order)]).astype(np.float32),0)
        for substep in pd.reshape(-1,87):
            expected=np.clip(kp*(targets-substep[:29])-kd*substep[29:58],-limits,limits)
            check('clipped_pd_torque',substep[58:],expected,1e-10)
        check('reference_fk',np.array(row['reference']).reshape(30,3),expected_positions[index],2e-6)
        last=actions.copy()
    report=dict(schema=1,result='pass',samples=len(rows),recording_sha256=sha256(args.recording),checks=checks,
                tracking_outcome=run['failure'],note='Boundary/neural parity does not assert trajectory equality or universal physical feasibility.')
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
