#!/usr/bin/env python3
"""Measure contact-point slip from recorded MuJoCo qpos/qvel, without stepping.

Contact geometry is recomputed at the saved post-integration state. Foot/world
contacts with separation <=0 are counted; no guessed foot-height or velocity
threshold substitutes for contact. Tangential point speed uses J(q)*qvel,
independently checked with finite differences of the moving rigid-body point.
This is geometric contact slip, not a force-weighted stance metric.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from export_sonic_motion import sha256
from analyze_sonic import active_physics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--scene',type=Path,required=True)
    args = p.parse_args()
    summary = json.loads((args.run/'summary.json').read_text())
    play = next(e for e in summary['events'] if e['name']=='play_requested')
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data,probe = mujoco.MjData(model),mujoco.MjData(model)
    feet = [model.body(name).id for name in ('left_ankle_roll_link','right_ankle_roll_link')]
    speeds,counts = [],[]
    fd_errors = []
    with np.load(args.run/'physics.npz',allow_pickle=False) as recording:
        active = np.flatnonzero(active_physics(recording, play))
        if len(active)==0 or not np.isfinite(recording['qpos']).all() or not np.isfinite(recording['qvel']).all():
            raise ValueError('missing/invalid physical recording')
        for index in active:
            data.qpos[:] = recording['qpos'][index]
            data.qvel[:] = recording['qvel'][index]
            mujoco.mj_fwdPosition(model,data)
            count = [0,0]
            for contact in data.contact:
                bodies = [int(model.geom_bodyid[g]) for g in contact.geom]
                body = next((b for b in feet if b in bodies),None)
                if body is None or 0 not in bodies or contact.dist > 0:
                    continue
                jac = np.zeros((3,model.nv))
                mujoco.mj_jac(model,data,jac,None,contact.pos,body)
                velocity = jac@data.qvel
                normal = contact.frame[:3]
                tangent = velocity-normal*np.dot(normal,velocity)
                speeds.append(float(np.linalg.norm(tangent)))
                count[feet.index(body)] += 1
                if len(fd_errors)<100 and index%10==0:
                    local = data.xmat[body].reshape(3,3).T@(contact.pos-data.xpos[body])
                    probe.qpos[:] = data.qpos
                    mujoco.mj_integratePos(model,probe.qpos,data.qvel,1e-6)
                    mujoco.mj_kinematics(model,probe)
                    moved = probe.xpos[body]+probe.xmat[body].reshape(3,3)@local
                    fd_errors.append(float(np.max(np.abs((moved-contact.pos)/1e-6-velocity))))
            counts.append(count)
    if not speeds or len(fd_errors)<10 or max(fd_errors)>1e-4:
        raise ValueError(f'insufficient contacts or failed Jacobian finite-difference check: {fd_errors[:10]}')
    report = dict(schema=1,physics_sha256=sha256(args.run/'physics.npz'),scene_sha256=sha256(args.scene),
                  active_physics_samples=len(active),contact_point_samples=len(speeds),
                  foot_contact_fraction=np.mean(np.array(counts)>0,axis=0).tolist(),
                  no_foot_contact_fraction=float(np.mean(np.sum(counts,axis=1)==0)),
                  tangential_speed_mean_mps=float(np.mean(speeds)),
                  tangential_speed_p95_mps=float(np.quantile(speeds,.95)),
                  tangential_speed_max_mps=float(np.max(speeds)),
                  jacobian_fd_probes=len(fd_errors),jacobian_fd_max_error_mps=max(fd_errors),
                  interpretation='Recomputed post-state penetrating foot/world contact-point speed; not force-weighted; no physics steps or pose changes.')
    (args.run/'contacts.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
