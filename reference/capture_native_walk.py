#!/usr/bin/env python3
"""Record a separate MotionBricks demo session without changing browser sessions.

Preserves full API responses and consumed-frame boundaries, not just a rendered
trajectory. The server must already be running with the desired model/backend.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import urllib.parse
import urllib.request


def local_identity(pid, url, style):
    """Bind a local HTTP capture to its listening PID, argv and artifact hashes."""
    endpoint = urllib.parse.urlparse(url)
    if endpoint.hostname not in ('127.0.0.1','localhost','::1'):
        raise ValueError('--server-pid requires a loopback URL on the same host')
    process=Path('/proc')/str(pid)
    sockets=set()
    for descriptor in (process/'fd').iterdir():
        try: sockets.add(os.readlink(descriptor))
        except FileNotFoundError: pass # An unrelated HTTP connection closed.
    owned=False
    for kind in ('tcp','tcp6'):
        for line in (process/'net'/kind).read_text().splitlines()[1:]:
            fields=line.split()
            if fields[3]=='0A' and int(fields[1].split(':')[1],16)==(endpoint.port or 80) and f'socket:[{fields[9]}]' in sockets:
                owned=True
    if not owned: raise ValueError('server PID does not own the requested listening port')
    argv=(process/'cmdline').read_bytes().decode().strip('\0').split('\0')
    cwd=Path(os.readlink(process/'cwd'))
    def option(name):
        return argv[argv.index(name)+1]
    def path_option(name):
        path=Path(option(name))
        return path if path.is_absolute() else cwd/path
    library=path_option('-library')
    stat=library.stat()
    if not any(str(library) in line and int(line.split()[4])==stat.st_ino for line in (process/'maps').read_text().splitlines()):
        raise ValueError('library on disk is not the mapped server library')
    paths={'library':library,'server':process/'exe','style':path_option('-styles')/(style+'.mbstyle')}
    for path in sorted(path_option('-model').glob('*.gguf')):
        paths['model/'+path.name]=path
    if len([k for k in paths if k.startswith('model/')])!=4:
        raise ValueError('expected four MotionBricks model GGUF files')
    if '-kimodo-dir' in argv:
        root=path_option('-kimodo-dir')
        for path in sorted(root.rglob('animation.glb')):
            paths['kimodo/'+str(path.relative_to(root))]=path
    hashes={}
    for name,path in paths.items():
        with path.open('rb') as stream: hashes[name]=hashlib.file_digest(stream,'sha256').hexdigest()
    return {'pid':pid,'argv':argv,'device':option('-device'),
            'sampling':option('-sampling') if '-sampling' in argv else 'gumbel',
            'files_sha256':hashes,'verification':'listener PID, mapped library inode, file hashes before/after capture'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--style", default="walk")
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument('--server-pid',type=int,help='Verify local listening PID and model/library/style identities')
    parser.add_argument('--pattern',choices=['straight','stop-turn','kimodo-return'],default='straight')
    parser.add_argument('--kimodo-id',help='Explicit G1 clip ID for kimodo-return')
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 1 <= args.seconds <= 60:
        parser.error("seconds must be between 1 and 60")
    if not 0 <= args.seed < 2**64 - 120:
        parser.error("seed is outside supported range")
    if args.output.exists():
        parser.error("output already exists")
    if args.pattern=='kimodo-return' and not args.kimodo_id:
        parser.error('kimodo-return requires --kimodo-id')
    if (args.pattern=='stop-turn' and args.seconds<10) or (args.pattern=='kimodo-return' and args.seconds<4):
        parser.error('duration is too short to include all requested command phases')
    identity=local_identity(args.server_pid,args.url,args.style) if args.server_pid else None

    def request(route, body=None):
        req = urllib.request.Request(args.url.rstrip("/") + route,
                                     data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as response:
            return json.load(response)

    meta = request("/api/meta")
    if meta.get("fps") != 30 or len(meta.get("joints", [])) != 34:
        raise ValueError("expected live MotionBricks G1 metadata")
    initial = request("/api/session", {"style": args.style})
    session = initial["session"]
    total = math.ceil(args.seconds * 30)
    roots, rotations, plans = [], [], []
    inserted=False
    prefetched=None
    # Replan at the demo's 16-frame cadence. Each new plan starts at the
    # advanced agent cursor; no extra interpolation/blending is added here.
    while len(roots) // 3 < total:
        frame=len(roots)//3
        if args.pattern=='kimodo-return' and frame==96 and not inserted:
            previous=plans[-1]['response']['motion']
            advance=plans[-1]['consumed_frames']
            command={'session':session,'clip':args.kimodo_id,'advance':advance,
                     'current_root':previous['roots'][advance*3:advance*3+3],
                     'current_rotations':previous['rotations'][advance*136:(advance+1)*136]}
            response=request('/api/kimodo/start',command)
            authored=response['motion']
            # The final authored sample is the first exit-plan sample: share
            # that timestamp, rather than duplicating a stationary frame.
            consumed=authored['frames']-1
            plans.append({'start_frame':frame,'consumed_frames':consumed,'route':'/api/kimodo/start',
                          'request':command,'response':response})
            roots.extend(authored['roots'][:consumed*3])
            rotations.extend(authored['rotations'][:consumed*136])
            command={'session':session,'style':args.style,'move':[0,1],'facing':[0,1],'seed':args.seed+len(plans)}
            prefetched=(command,request('/api/kimodo/finish',command))
            inserted=True
            total=max(total,len(roots)//3+120) # At least four native seconds after the exit.
            frame=len(roots)//3
        moving,facing=[0,1],[0,1]
        if args.pattern=='stop-turn':
            if 96<=frame<144: moving=[0,0]
            elif 144<=frame<272: moving,facing=[1,0],[1,0]
            elif frame>=272: moving,facing=[0,0],[1,0]
        command = {"session": session, "style": args.style, "move": [0, 1],
                   "facing": [0, 1], "seed": args.seed + len(plans),
                   "advance": 0 if not plans else plans[-1]["consumed_frames"]}
        command.update(move=moving,facing=facing)
        route='/api/plan'
        if prefetched is not None:
            command,response=prefetched
            prefetched=None
            route='/api/kimodo/finish'
        else:
            response = request(route, command)
        motion = response["motion"]
        if motion["joints"] != 34 or not 24 <= motion["frames"] <= 64:
            raise ValueError("unexpected native plan shape")
        if len(motion["roots"]) != motion["frames"] * 3 or len(motion["rotations"]) != motion["frames"] * 34 * 4:
            raise ValueError("truncated native plan")
        if not all(math.isfinite(v) for v in motion["roots"] + motion["rotations"]):
            raise ValueError("non-finite native motion")
        consumed = min(16, total - len(roots) // 3)
        plans.append({"start_frame": len(roots) // 3, "consumed_frames": consumed,
                      "route":route,"request": command, "response": response})
        roots.extend(motion["roots"][:consumed * 3])
        rotations.extend(motion["rotations"][:consumed * 34 * 4])
        print(f"recorded {len(roots)//3}/{total} frames", flush=True)
    result = {"schema": 1, "producer": "motion-bricks.cpp demo native API",
              "metadata": meta, "initial_response": initial, "plans": plans,
              "fps": 30, "frames": total, "joints": 34,
              "roots": roots, "rotations": rotations,
              "pattern":args.pattern,"runtime_identity":identity,
              "runtime_identity_note": "Use --server-pid for verified local runtime provenance."}
    if identity is not None and local_identity(args.server_pid,args.url,args.style)!=identity:
        raise ValueError('runtime identity changed during capture; refusing an ambiguous record')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
