#!/usr/bin/env python3
"""Capture real clip poses and feature encoding using NVIDIA's Warp runtime on CPU."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream',type=Path,required=True)
    parser.add_argument('--motion',required=True,help='BVH path relative to the upstream checkout')
    parser.add_argument('--context-time',type=float,default=0.0)
    parser.add_argument('--context-poses',type=Path,help='Override context with four local poses in an NPY array')
    parser.add_argument('--target-time',type=float,default=1.0)
    parser.add_argument('--target-poses',type=int,choices=(1,2,3,4),default=4)
    parser.add_argument('--target-local-poses',type=Path,help='Override four target local poses for explicitly labelled stress fixtures')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    pin=json.loads(Path(__file__).with_name('soma-reference.json').read_text())['source_revision']
    root=args.upstream.resolve(); path=(root/args.motion).resolve()
    relative=path.relative_to(root)
    pointer=subprocess.check_output(['git','-C',str(root),'show',f'{pin}:{relative.as_posix()}'],text=True)
    expected=next(line.split('sha256:')[1] for line in pointer.splitlines() if line.startswith('oid '))
    with path.open('rb') as stream: actual=hashlib.file_digest(stream,'sha256').hexdigest()
    if actual!=expected:raise ValueError('Clip differs from pinned upstream LFS identity')
    sys.path.insert(0,str(root))
    import warp as wp
    wp.config.kernel_cache_dir=str(Path('.cache/soma-warp').resolve())
    wp.init();wp.set_device('cpu')
    from motionbricks.io.bvh import load_bvh
    from motionbricks.utils.pose_utils import compute_world_poses
    from motionbricks.model.soma import kernels
    from motionbricks.model.soma.contract import SOMA_N1_HEADING_JOINTS
    from motionbricks.model.kernels import write_motionbricks_conditions_kernel
    from motionbricks.io.usd import load_skeleton_from_usd
    skeleton_path=root/'motionbricks/assets/human/soma_base_skel_minimal_no_root.usd'
    pointer=subprocess.check_output(['git','-C',str(root),'show',f'{pin}:motionbricks/assets/human/soma_base_skel_minimal_no_root.usd'],text=True)
    skeleton_sha=next(line.split('sha256:')[1] for line in pointer.splitlines() if line.startswith('oid '))
    with skeleton_path.open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=skeleton_sha:
            raise ValueError('SOMA USD skeleton differs from pinned LFS identity')
    skeleton=load_skeleton_from_usd(skeleton_path,'/OUTPUT/c_skeleton_grp/Hips')
    clip=load_bvh(path,skeleton)
    contract=json.loads((root/'motionbricks/assets/onnx/soma/soma_n1.json').read_text())
    names=contract['joint_names']; skeleton=clip.skeleton
    mapping=np.array([skeleton.joint_index(name) for name in names],np.int32)
    if np.any(mapping<0) or mapping[0]!=0:raise ValueError('Clip lacks required SOMA joints or Hips root')
    poses=wp.empty((8,skeleton.num_joints),dtype=wp.transform,device='cpu')
    times=[args.context_time+i/30 for i in range(4)]+[args.target_time+min(i,args.target_poses-1)/30 for i in range(4)]
    if min(times)<0 or max(times)>(clip.num_frames-1)/clip.sample_rate:
        raise ValueError('Requested poses are outside the clip')
    for i,t in enumerate(times):clip.sample(t,poses[i])
    if args.context_poses:
        context=np.load(args.context_poses,allow_pickle=False)
        if context.shape!=(4,skeleton.num_joints,7) or not np.isfinite(context).all():
            raise ValueError('Invalid chained context poses')
        wp.copy(poses[:4],wp.array(context.astype(np.float32),dtype=wp.transform,device='cpu'))
    if args.target_local_poses:
        target=np.load(args.target_local_poses,allow_pickle=False)
        if target.shape!=(4,skeleton.num_joints,7) or not np.isfinite(target).all():
            raise ValueError('Invalid target override poses')
        wp.copy(poses[4:],wp.array(target.astype(np.float32),dtype=wp.transform,device='cpu'))
    world=wp.empty_like(poses);compute_world_poses(poses,skeleton.parent_indices,world)
    alignment=wp.empty(1,dtype=wp.transform,device='cpu')
    right,left=(skeleton.joint_index(name) for name in SOMA_N1_HEADING_JOINTS)
    if min(right,left)<0:raise ValueError('Missing hip heading joints')
    wp.launch(kernels.compute_guide_alignment_kernel,dim=1,
              inputs=[right,left,wp.vec2(0.,1.),world],outputs=[alignment],device='cpu')
    buffers={name:wp.zeros(shape,dtype=wp.float32,device='cpu') for name,shape in
             [('global_root_values',(1,8,5)),('local_root_values',(1,8,4)),('local_poses',(1,8,267))]}
    wp.launch(kernels.encode_motionbricks_inputs_kernel,dim=8,
              inputs=[30,right,left,wp.array(mapping,dtype=wp.int32),world,alignment],
              outputs=list(buffers.values()),device='cpu')
    inputs={name:value.numpy() for name,value in buffers.items()}
    conditions={name:wp.zeros((1,count),dtype=wp.int32,device='cpu') for name,count in
                [('has_global_root_values',8),('has_local_root_values',8),('has_local_poses',8),('allowed_pred_num_tokens',11)]}
    wp.launch(write_motionbricks_conditions_kernel,dim=11,inputs=[args.target_poses,(1<<11)-1],
              outputs=list(conditions.values()),device='cpu')
    inputs.update({name:value.numpy() for name,value in conditions.items()})
    args.output.mkdir(parents=True,exist_ok=False)
    np.savez(args.output/'inputs.npz',**inputs)
    np.savez(args.output/'poses.npz',local=poses.numpy(),world=world.numpy(),alignment=alignment.numpy(),
             parents=skeleton.parent_indices.numpy(),model_to_pose=mapping,
             reference_local=skeleton.reference_local_transforms.numpy())
    report=dict(source_revision=pin,motion=args.motion,motion_sha256=actual,skeleton_sha256=skeleton_sha,warp_version=wp.__version__,
                reference_device='cpu',times=times,target_poses=args.target_poses,
                joint_names=skeleton.joint_names,model_joint_names=names)
    if args.context_poses:
        report['context_override']=str(args.context_poses)
        report['context_sha256']=hashlib.sha256(args.context_poses.read_bytes()).hexdigest()
    if args.target_local_poses:
        report['target_override']=str(args.target_local_poses)
        report['target_sha256']=hashlib.sha256(args.target_local_poses.read_bytes()).hexdigest()
        report['fixture_kind']='synthetic-target-stress'
    (args.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'Captured upstream motion inputs: {args.output}')


if __name__=='__main__':main()
