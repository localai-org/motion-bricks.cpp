#!/usr/bin/env python3
"""Export recorded native G1 motion to upstream SONIC's 50 Hz CSV format.

NumPy/SciPy implementation, independently checked by validate_sonic_adapter.py.
The hinge projection follows NVIDIA's motionbricks/helper/mujoco_helper.py;
sampling/velocity conventions follow gear_sonic/utils/motion_lib/
torch_humanoid_batch.py at the revision in fetch_sonic.py. No neural inference,
joint clamping, pose fitting or automatic root-height correction is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation, Slerp

# Gather hardware/MuJoCo body angles into IsaacLab order. Named by direction
# rather than upstream's confusingly named pair of inverse arrays.
HARDWARE_TO_ISAAC = np.array([0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22,
                            4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26,
                            20, 27, 21, 28])
MJ_TO_MOTION = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_motion(path):
    obj = json.loads(Path(path).read_text())
    n = obj.get('frames', 0)
    if obj.get('schema') != 1 or obj.get('fps') != 30 or obj.get('joints') != 34 or not 3 <= n <= 18000:
        raise ValueError('expected a schema-1 native G1 recording, 3..18000 frames at 30 FPS')
    joints = obj['metadata']['joints']
    if len(joints) != 34 or len({j['name'] for j in joints}) != 34:
        raise ValueError('invalid skeleton names/count')
    if joints[0]['parent'] != -1 or any(not 0 <= j['parent'] < i for i, j in enumerate(joints) if i):
        raise ValueError('skeleton must have one root and topologically ordered parents')
    roots = np.asarray(obj['roots'], dtype=np.float64).reshape(n, 3)
    quats = np.asarray(obj['rotations'], dtype=np.float64).reshape(n, 34, 4)
    if not np.isfinite(roots).all() or not np.isfinite(quats).all():
        raise ValueError('non-finite native motion')
    norm = np.linalg.norm(quats, axis=-1)
    if np.max(np.abs(norm - 1)) > 1e-3:
        raise ValueError('native rotations must be unit XYZW quaternions')
    return obj, roots, quats / norm[..., None]


class HingeConverter:
    def __init__(self, xml_path, joints):
        tree = ET.parse(xml_path)
        names = [j['name'] for j in joints]
        defaults = {e.get('class'): e.find('joint').get('axis')
                    for e in tree.findall('.//default')
                    if e.get('class') and e.find('joint') is not None}
        parents = {child: parent for parent in tree.iter() for child in parent}
        hinges = tree.find('worldbody').findall('.//joint')
        if len(hinges) != 29 or names[0] != 'pelvis_skel':
            raise ValueError('expected the MotionBricks 29-hinge G1 XML and pelvis root')
        self.names, self.indices, axes, offsets = [], [], [], []
        for hinge in hinges:
            name = hinge.get('name')
            index = names.index(name.replace('_joint', '_skel'))
            parent_body = parents[parents[hinge]]
            parent_joint = parent_body.find('joint')
            expected_parent = parent_joint.get('name').replace('_joint', '_skel') if parent_joint is not None else 'pelvis_skel'
            if names[joints[index]['parent']] != expected_parent:
                raise ValueError(f'skeleton/XML parent mismatch at {name}')
            axis = np.fromstring(hinge.get('axis') or defaults[hinge.get('class')], sep=' ')
            if axis.shape != (3,) or not any(np.array_equal(axis, e) for e in np.eye(3)):
                raise ValueError('only upstream positive coordinate-axis hinges are supported')
            quat = np.fromstring(parents[hinge].get('quat', '1 0 0 0'), sep=' ')
            rest = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
            offsets.append(MJ_TO_MOTION @ rest.T @ MJ_TO_MOTION.T)
            axes.append(MJ_TO_MOTION @ axis)
            self.names.append(name)
            self.indices.append(index)
        self.axes, self.offsets = np.array(axes), np.array(offsets)

    def convert(self, roots, quats):
        local = Rotation.from_quat(quats.reshape(-1, 4)).as_matrix().reshape(len(roots), 34, 3, 3)
        hinge = self.offsets @ local[:, self.indices]
        xyz = np.stack([np.arctan2(hinge[..., 2, 1], hinge[..., 2, 2]),
                        np.arctan2(hinge[..., 0, 2], hinge[..., 0, 0]),
                        np.arctan2(hinge[..., 1, 0], hinge[..., 1, 1])], axis=-1)
        angles = np.sum(xyz * self.axes, axis=-1)
        root_rot = MJ_TO_MOTION.T @ local[:, 0] @ MJ_TO_MOTION
        root_quat = Rotation.from_matrix(root_rot).as_quat()[:, [3, 0, 1, 2]]
        qpos = np.concatenate([roots @ MJ_TO_MOTION, root_quat, angles], axis=-1)
        projected = Rotation.from_rotvec((angles[..., None] * self.axes).reshape(-1, 3)).as_matrix().reshape(hinge.shape)
        residual = Rotation.from_matrix((projected.swapaxes(-1, -2) @ hinge).reshape(-1, 3, 3)).magnitude()
        return qpos, residual.reshape(len(roots), 29)


def resample(qpos, source_fps=30, target_fps=50):
    # Upstream samples [0, (N-1)/fps), not N/fps, excluding the last timestamp.
    # Keep original and resampled timestamps explicit; never stretch time to
    # make a rounded frame count match a desired duration.
    src = np.arange(len(qpos), dtype=np.float64) / source_fps
    dst = np.arange(0, src[-1], 1 / target_fps, dtype=np.float64)
    roots = np.stack([np.interp(dst, src, qpos[:, i]) for i in range(3)], axis=-1)
    root_quat = Slerp(src, Rotation.from_quat(qpos[:, [4, 5, 6, 3]]))(dst).as_quat()[:, [3, 0, 1, 2]]
    # SLERP of single-axis rotations is shortest-arc interpolation of angles.
    unwrapped = np.unwrap(qpos[:, 7:], axis=0)
    angles = np.stack([np.interp(dst, src, unwrapped[:, i]) for i in range(29)], axis=-1)
    return dst, np.concatenate([roots, root_quat, angles], axis=-1)


def velocities(angles, body_pos, body_quat, fps=50):
    diff = np.diff(angles, axis=0) * fps
    # Deliberately reproduce upstream's penultimate-difference terminal row
    # (fk_batch uses dof_vel[:, -2:-1]), including its unusual endpoint rule.
    joint_vel = np.concatenate([diff, diff[-2:-1]], axis=0)
    linear = gaussian_filter1d(np.gradient(body_pos, axis=0) * fps, 2, axis=0, mode='nearest')
    rotations = Rotation.from_quat(body_quat.reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix().reshape(*body_quat.shape[:-1], 3, 3)
    delta = rotations[1:] @ rotations[:-1].swapaxes(-1, -2)
    angular = np.zeros_like(body_pos)
    angular[:-1] = Rotation.from_matrix(delta.reshape(-1, 3, 3)).as_rotvec().reshape(delta.shape[:2] + (3,)) * fps
    angular = gaussian_filter1d(angular, 2, axis=0, mode='nearest')
    return joint_vel, linear, angular


def physical_fk(scene, names, qpos):
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
    if any(i < 0 for i in ids) or len(set(ids)) != 29:
        raise ValueError('physical robot does not have the same 29 named body joints')
    addresses = model.jnt_qposadr[ids]
    body_ids = np.r_[1, model.jnt_bodyid[np.asarray(ids)[HARDWARE_TO_ISAAC]]]
    pos, quat = [], []
    for frame in qpos:
        data.qpos[:] = model.qpos0
        data.qpos[:7] = frame[:7]
        data.qpos[addresses] = frame[7:]
        mujoco.mj_kinematics(model, data)
        pos.append(data.xpos[body_ids].copy())
        quat.append(data.xquat[body_ids].copy())
    return np.array(pos), np.array(quat), [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in body_ids]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--motion-xml', type=Path, required=True)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New clip directory inside a motions directory')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists')
    obj, roots, quats = read_motion(args.input)
    converter = HingeConverter(args.motion_xml, obj['metadata']['joints'])
    original, residual = converter.convert(roots, quats)
    times, qpos = resample(original)
    positions, rotations, body_names = physical_fk(args.scene, converter.names, qpos)
    joint_vel, linear, angular = velocities(qpos[:, 7:], positions, rotations)
    arrays = dict(joint_pos=qpos[:, 7:][:, HARDWARE_TO_ISAAC],
                  joint_vel=joint_vel[:, HARDWARE_TO_ISAAC], body_pos=positions,
                  body_quat=rotations, body_lin_vel=linear, body_ang_vel=angular)
    if any(not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('export contains non-finite values')
    args.output.mkdir(parents=True)
    for name, array in arrays.items():
        flat = array.reshape(len(times), -1)
        np.savetxt(args.output / (name + '.csv'), flat, delimiter=',', fmt='%.9f',
                   header=','.join(f'{name}_{i}' for i in range(flat.shape[1])), comments='')
    (args.output / 'metadata.txt').write_text('Body part indexes:\n[' + ' '.join(map(str, range(30))) + ']\n\n' +
                                            f'Total timesteps: {len(times)}\n')
    np.savez_compressed(args.output / 'adapter.npz', source_time=np.arange(len(roots))/30,
                        source_qpos=original, time=times, qpos=qpos, projection_residual=residual,
                        body_pos=positions, body_quat=rotations)
    manifest = dict(schema=1, fps=50, frames=len(times), source_fps=30, source_frames=len(roots),
                    source_last_time=(len(roots)-1)/30, last_time=float(times[-1]),
                    input_sha256=sha256(args.input), motion_xml_sha256=sha256(args.motion_xml),
                    scene_sha256=sha256(args.scene), exporter_sha256=sha256(__file__),
                    hardware_joint_names=converter.names, isaac_body_names=body_names,
                    hardware_to_isaac_gather=HARDWARE_TO_ISAAC.tolist(),
                    hinge_projection_max_rad=float(residual.max()),
                    hinge_projection_rms_rad=float(np.sqrt(np.mean(residual**2))),
                    endpoint_convention='upstream half-open time grid; final joint velocity uses penultimate difference',
                    body_velocities='upstream gradient/relative-quaternion differences, Gaussian sigma=2 nearest',
                    physical_fk='unmodified simulation scene; zero hand angles; no pose fitting/clamping',
                    files={p.name: sha256(p) for p in sorted(args.output.iterdir())})
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
