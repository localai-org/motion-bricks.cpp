#!/usr/bin/env python3
"""Compare the adapter to pinned, actual upstream geometry operations.

Uses the real G1Skeleton34, GlobalRootGlobalJoints and mujoco_qpos_converter,
not the shape-only neural fixture representation. Native local rotations are
composed into world rotations and packed into the real representation's named
fields. No converter output is reused as expected input for native parity.

Humanoid_Batch's six numerical interpolation/velocity methods are executed
verbatim from its verified AST, avoiding imports of unrelated mesh/training
dependencies (Open3D/Isaac). This tests those methods, not the full training FK
pipeline. Physical FK/CSV consumption are checked separately in C++/MuJoCo.
"""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import scipy.ndimage as filters
import torch

from export_sonic_motion import HingeConverter, read_motion, resample, velocities, sha256
from fetch_sonic import SOURCE_REVISION


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream-root', type=Path, required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--export', type=Path, required=True)
    p.add_argument('--reader-output', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    upstream = args.upstream_root
    paths = ['motionbricks/motionbricks/helper/mujoco_helper.py',
             'motionbricks/assets/skeletons/g1/g1.xml',
             'gear_sonic/utils/motion_lib/torch_humanoid_batch.py']
    for path in paths:
        expected = subprocess.check_output(['git', '-C', str(upstream), 'show', f'{SOURCE_REVISION}:{path}'])
        if (upstream/path).read_bytes() != expected:
            raise ValueError(f'upstream source differs from pinned revision: {path}')
    sys.path[:0] = [str(upstream/'motionbricks'), str(upstream)]
    from motionbricks.helper.mujoco_helper import mujoco_qpos_converter, global_mats_to_local_mats
    from motionbricks.motionlib.core.skeletons.g1 import G1Skeleton34
    from motionbricks.motionlib.core.motion_reps.dual_root_global_joints import GlobalRootGlobalJoints
    from motionbricks.motionlib.core.utils.rotations import matrix_to_cont6d, quaternion_to_matrix
    import gear_sonic.isaac_utils.rotations as upstream_rotations

    obj, roots, quats = read_motion(args.input)
    skel = G1Skeleton34(load=False)
    if skel.bone_order_names != [j['name'] for j in obj['metadata']['joints']] or skel.joint_parents.tolist() != [j['parent'] for j in obj['metadata']['joints']]:
        raise ValueError('native skeleton differs from upstream G1Skeleton34')
    skel.register_buffer('neutral_joints', torch.tensor([j['position'] for j in obj['metadata']['joints']], dtype=torch.float32))
    rep = GlobalRootGlobalJoints(fps=30, skeleton=skel, name='global_root_global_joints')
    xml = upstream/paths[1]
    upstream_converter = mujoco_qpos_converter(rep, str(xml))
    adapter = HingeConverter(xml, obj['metadata']['joints'])
    report = {'schema': 1, 'source_revision': SOURCE_REVISION, 'input_sha256': sha256(args.input),
              'source_hashes': {path: sha256(upstream/path) for path in paths}, 'checks': {}}

    def check(name, actual, expected, tolerance):
        actual, expected = np.asarray(actual), np.asarray(expected)
        if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise AssertionError(f'{name}: invalid shape or non-finite data')
        maximum = float(np.max(np.abs(actual-expected))) if actual.size else 0.
        report['checks'][name] = {'max_abs_error': maximum, 'tolerance': tolerance, 'shape': list(actual.shape)}
        if maximum > tolerance:
            raise AssertionError(f'{name}: {maximum} exceeds {tolerance}')

    def pack(root, globals_):
        # These are the only fields read by the real upstream converter;
        # root heading is valid identity but the full root rotation is in 6D.
        features = torch.zeros((1, len(root), rep.motion_rep_dim), dtype=torch.float32)
        features[..., rep.indices['global_root_pos']] = torch.as_tensor(root, dtype=torch.float32)[None]
        features[..., rep.indices['global_root_heading'][0]] = 1
        features[..., rep.indices['global_rot_data']] = matrix_to_cont6d(globals_).reshape(1, len(root), -1)
        return features

    local = quaternion_to_matrix(torch.tensor(quats[..., [3, 0, 1, 2]], dtype=torch.float32))
    global_rot = local.clone()
    for i, parent in enumerate(skel.joint_parents.tolist()):
        if parent >= 0:
            global_rot[:, i] = global_rot[:, parent] @ local[:, i]
    reference = upstream_converter.convert_motion_features_to_mujoco_qpos(pack(roots, global_rot), rep, False, True)[0].numpy()
    actual, _ = adapter.convert(roots, quats)
    check('native_root_translation', actual[:, :3], reference[:, :3], 2e-6)
    check('native_hinge_angles', actual[:, 7:], reference[:, 7:], 3e-6)
    check('native_root_rotation_matrix', Rotation.from_quat(actual[:, [4,5,6,3]]).as_matrix(),
          Rotation.from_quat(reference[:, [4,5,6,3]]).as_matrix(), 3e-6)

    # Upstream inverse makes independent legal hinge probes. Test each hinge
    # with both signs, simultaneous bends, and root translation/roll/pitch/yaw.
    probes = np.zeros((62, 36), dtype=np.float32)
    probes[:, 3] = 1
    probes[:, 2] = .8
    for i in range(29):
        probes[2*i+1, 7+i], probes[2*i+2, 7+i] = .3, -.3
    probes[59:, :3] = [[1,2,.8],[-2,1,.9],[0,-1,.7]]
    probes[59:, 3:7] = Rotation.from_euler('xyz', [[.1,.2,.8],[-.2,.1,-1.4],[0,0,3.1]]).as_quat()[:, [3,0,1,2]]
    probes[61, 7:] = np.linspace(-.2,.2,29)
    pos, rot = upstream_converter.convert_mujoco_qpos_to_motion_transforms(torch.from_numpy(probes)[None])
    local_probe = global_mats_to_local_mats(rot, skel)[0].numpy()
    probe_quats = Rotation.from_matrix(local_probe.reshape(-1,3,3)).as_quat().reshape(-1,34,4)
    converted, residual = adapter.convert(pos[0,:,0].numpy(), probe_quats)
    check('all_hinge_probes', converted[:, 7:], probes[:, 7:], 3e-6)
    check('root_translation_probes', converted[:, :3], probes[:, :3], 2e-6)
    check('root_rotation_probes', Rotation.from_quat(converted[:, [4,5,6,3]]).as_matrix(),
          Rotation.from_quat(probes[:, [4,5,6,3]]).as_matrix(), 3e-6)
    check('legal_hinge_projection_residual', residual, np.zeros_like(residual), 3e-6)

    # Execute original numeric methods without changing their AST. No stubs
    # stand in for math; dependencies are upstream rotation functions.
    tree = ast.parse((upstream/paths[2]).read_text())
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Humanoid_Batch')
    selected = {'_lerp', '_slerp', '_compute_frame_blend', 'interploate_pose',
                '_compute_velocity', '_compute_angular_velocity'}
    klass.body = [n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name in selected]
    if len(klass.body) != len(selected):
        raise ValueError('upstream numerical methods changed')
    namespace = dict(torch=torch, np=np, filters=filters, **{n: getattr(upstream_rotations,n) for n in
                     ('slerp','quat_identity_like','quat_mul_norm','quat_inverse','quat_angle_axis')})
    exec(compile(ast.Module(body=[klass], type_ignores=[]), str(upstream/paths[2]), 'exec'), namespace)
    humanoid = namespace['Humanoid_Batch']()
    # Root + 29 hinge rotations in MuJoCo world coordinates, WXYZ.
    mj_axes = adapter.axes @ np.array([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])
    pose = Rotation.from_rotvec((actual[:,7:,None]*mj_axes).reshape(-1,3)).as_quat().reshape(-1,29,4)[..., [3,0,1,2]]
    pose = np.concatenate([actual[:,None,3:7], pose], axis=1)
    interp_q, interp_p = humanoid.interploate_pose(torch.tensor(pose[None],dtype=torch.float32),
                                                 torch.tensor(actual[None,:,:3],dtype=torch.float32),30,50)
    times, sampled = resample(actual)
    check('resampled_root_translation', sampled[:,:3], interp_p[0].numpy(), 5e-6)
    root_actual = Rotation.from_quat(sampled[:,[4,5,6,3]]).as_matrix()
    root_expected = Rotation.from_quat(interp_q[0,:,0].numpy()[:,[1,2,3,0]]).as_matrix()
    interp_angles = Rotation.from_quat(interp_q[0,:,1:].numpy().reshape(-1,4)[:,[1,2,3,0]]).as_rotvec().reshape(-1,29,3)
    interp_angles = np.sum(interp_angles*mj_axes,axis=-1)
    # Upstream slerp substitutes a time-INDEPENDENT midpoint for sin(theta)
    # < .001, or the first sample when the F32 dot rounds to >=1. SciPy uses
    # the requested time. Bound that intentional difference analytically:
    # <= .001 rad for midpoint (half a <.002 rad step), with F32 margin.
    # Everywhere outside those branches, retain the tight parity assertion.
    source_pose = torch.tensor(pose[None],dtype=torch.float32)
    upstream_times = torch.arange(0,(len(actual)-1)/30,.02,dtype=torch.float32)
    i0,i1,_ = humanoid._compute_frame_blend(upstream_times,(len(actual)-1)/30,len(actual))
    dot = torch.sum(source_pose[0,i0]*source_pose[0,i1],dim=-1).abs()
    all_fallback = ((torch.sqrt(1-dot*dot) < .001) | (dot >= 1)).numpy()
    root_fallback,fallback = all_fallback[:,0],all_fallback[:,1:]
    check('resampled_root_regular_slerp',root_actual[~root_fallback],root_expected[~root_fallback],3e-5)
    check('resampled_root_upstream_midpoint_difference',root_actual[root_fallback],root_expected[root_fallback],.0011)
    check('resampled_hinges_regular_slerp',sampled[:,7:][~fallback],interp_angles[~fallback],3e-5)
    check('resampled_hinges_upstream_midpoint_difference',sampled[:,7:][fallback],interp_angles[fallback],.0011)
    report['intentional_resampling_difference'] = {
        'upstream_small_angle_fallback_samples':int(fallback.sum()),
        'upstream_root_small_angle_fallback_samples':int(root_fallback.sum()),
        'description':'Keep time-correct SLERP; upstream uses midpoint for tiny changes regardless of time fraction.'}
    export = np.load(args.export/'adapter.npz', allow_pickle=False)
    check('exported_qpos', sampled, export['qpos'], 1e-12)
    check('exported_timestamps', times, export['time'], 1e-12)
    pos, quat = export['body_pos'], export['body_quat']
    joint_vel, lin, ang = velocities(sampled[:,7:],pos,quat)
    expected_lin = humanoid._compute_velocity(torch.tensor(pos[None],dtype=torch.float64), .02)[0]
    expected_ang = humanoid._compute_angular_velocity(torch.tensor(quat[None,...,[1,2,3,0]],dtype=torch.float64), .02)[0]
    check('body_linear_velocity',lin,expected_lin.numpy(),1e-10)
    check('body_angular_velocity',ang,expected_ang.numpy(),3e-5)
    from export_sonic_motion import HARDWARE_TO_ISAAC
    reader = json.loads(args.reader_output.read_text())
    check('upstream_reader_joint_order', reader['hardware_to_isaac_gather'], HARDWARE_TO_ISAAC, 0)
    if reader['frames'] != len(times):
        raise AssertionError('upstream reader changed frame count')
    for field in ('joint_pos','joint_vel','body_pos','body_quat','body_lin_vel','body_ang_vel'):
        expected_csv = np.loadtxt(args.export/(field+'.csv'),delimiter=',',skiprows=1)
        check('upstream_reader_'+field,np.asarray(reader[field]).reshape(expected_csv.shape),expected_csv,1e-12)
    report['resampled_frames'] = len(times)
    report['result'] = 'pass'
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
