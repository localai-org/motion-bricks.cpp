#!/usr/bin/env python3
"""Evaluate a full physical run, rejecting boundary/safety failures.

Tracking errors are reported, not fitted away or called numerical parity.
This gate establishes functioning physical playback, not production-quality
tracking of arbitrary motions. Adapter validation remains a preceding gate.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from export_sonic_motion import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--clip',type=Path,required=True)
    p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--title',required=True)
    args=p.parse_args()
    source=Path(__file__).parent
    report={'schema':1,'result':'fail','criterion':'Unassisted complete playback with valid finite state, clean sanitizer/controller exit, checked controller and PD boundaries; tracking quality reported separately.'}
    try:
        summary=json.loads((args.run/'summary.json').read_text())
        if summary['failure'] or summary['controller_exit'] != 0:
            raise ValueError(f"physical run failed: {summary['failure']}")
        events={e['name']:e for e in summary['events']}
        if 'playback_window_finished' not in events:
            raise ValueError('requested playback did not complete')
        for tool,extra in (
                ('check_sonic_boundaries.py',['--clip',str(args.clip)]),
                ('sonic_contacts.py',['--scene',str(args.scene)]),
                ('sonic_playback.py',['--scene',str(args.scene),'--output',str(args.output),'--title',args.title])):
            subprocess.run([sys.executable,str(source/tool),'--run',str(args.run),*extra],check=True)
        diagnostics=json.loads((args.run/'diagnostics.json').read_text())
        if diagnostics['boundaries']['active_rows']<2:
            raise ValueError('no active policy playback')
        report.update(result='pass',diagnostics=diagnostics,
                      artifacts_sha256={name:sha256(args.run/name) for name in
                                        ('summary.json','provenance.json','physics.npz','boundaries.jsonl','diagnostics.json')},
                      playback_sha256=sha256(args.output),
                      evaluator_sha256=sha256(__file__))
    except Exception as error:
        report['failure']=str(error)
        raise
    finally:
        (args.run/'acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'result':report['result'],'run':str(args.run)}))


if __name__=='__main__': main()
