"""Offline geometry/timing/rejection tests. No models, assets or GPU needed."""
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from export_sonic_motion import HingeConverter, MJ_TO_MOTION, read_motion, resample, velocities


class AdapterTests(unittest.TestCase):
    def test_axes_and_rest_offsets(self):
        with tempfile.TemporaryDirectory() as directory:
            xml = ET.Element('mujoco')
            body = ET.SubElement(ET.SubElement(xml,'worldbody'),'body',name='pelvis')
            joints = [{'name':'pelvis_skel','parent':-1}]
            mats = np.tile(np.eye(3),(1,34,1,1))
            angles = np.linspace(-.7,.7,29)
            for i in range(29):
                axis = np.eye(3)[i%3]
                rest = Rotation.from_euler('zyx',[.2,-.1,.05]).as_matrix()
                quat = Rotation.from_matrix(rest).as_quat()[[3,0,1,2]]
                body = ET.SubElement(body,'body',name=f'j{i}',quat=' '.join(map(str,quat)))
                ET.SubElement(body,'joint',name=f'j{i}_joint',axis=' '.join(map(str,axis)))
                joints.append({'name':f'j{i}_skel','parent':i})
                mats[0,i+1] = MJ_TO_MOTION @ rest @ Rotation.from_rotvec(angles[i]*axis).as_matrix() @ MJ_TO_MOTION.T
            joints += [{'name':f'virtual{i}','parent':0} for i in range(4)]
            path = Path(directory)/'robot.xml'
            ET.ElementTree(xml).write(path)
            converter = HingeConverter(path,joints)
            q = Rotation.from_matrix(mats.reshape(-1,3,3)).as_quat().reshape(1,34,4)
            out,residual = converter.convert(np.array([[1,2,3]]),q)
            np.testing.assert_allclose(out[0,:3],[3,1,2])
            np.testing.assert_allclose(out[0,7:],angles,atol=1e-14)
            self.assertLess(residual.max(),1e-14)
            joints[5]['parent'] = 0
            with self.assertRaisesRegex(ValueError,'parent mismatch'):
                HingeConverter(path,joints)

    def test_time_grid_sign_continuity_and_constant_speed(self):
        qpos = np.zeros((31,36))
        qpos[:,0] = np.arange(31)/30 * 2
        qpos[:,3] = 1
        qpos[::2,3] = -1
        qpos[:,7] = np.arange(31)/30 * .4
        times,result = resample(qpos)
        self.assertEqual(len(times),50)
        self.assertAlmostEqual(times[-1],.98)
        np.testing.assert_allclose(result[:,0],times*2,atol=1e-14)
        np.testing.assert_allclose(result[:,7],times*.4,atol=1e-14)
        self.assertTrue(np.all(np.abs(result[:,3]) == 1))

    def test_upstream_terminal_velocity_rule(self):
        angles = (np.arange(5)**2)[:,None]
        pos = np.zeros((5,1,3))
        quat = np.tile([1,0,0,0],(5,1,1))
        joint,linear,angular = velocities(angles,pos,quat)
        np.testing.assert_array_equal(joint[:,0],[50,150,250,350,250])
        np.testing.assert_array_equal(linear,0)
        np.testing.assert_array_equal(angular,0)

    def test_reject_bad_native_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'motion.json'
            obj = dict(schema=1,fps=30,joints=34,frames=3,
                       metadata={'joints':[{'name':str(i),'parent':i-1} for i in range(34)]},
                       roots=[0]*9,rotations=list(np.tile([0,0,0,1],(3*34,1)).ravel().astype(float)))
            path.write_text(json.dumps(obj))
            read_motion(path)
            obj['rotations'][0] = 3
            path.write_text(json.dumps(obj))
            with self.assertRaisesRegex(ValueError,'unit'):
                read_motion(path)
            obj['rotations'][0] = float('nan')
            path.write_text(json.dumps(obj))
            with self.assertRaisesRegex(ValueError,'non-finite'):
                read_motion(path)


if __name__ == '__main__':
    unittest.main()
