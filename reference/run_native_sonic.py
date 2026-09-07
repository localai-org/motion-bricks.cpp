#!/usr/bin/env python3
"""Headless smoke/recording test for the native SONIC/MuJoCo C ABI.

Only stdlib is needed: neither inference nor simulation uses Python packages.
"""
import argparse
import ctypes as c
import json
from pathlib import Path
import time

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--motion',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','vulkan'],default='cpu')
    p.add_argument('--trace',action='store_true',help='include exact policy/motor boundary buffers for independent validation')
    args=p.parse_args()
    if args.output.exists(): p.error('output exists')
    lib=c.CDLL(str(args.library.resolve())); error=c.create_string_buffer(1024)
    signatures={'mb_runtime_options_create':[c.POINTER(c.c_void_p),c.c_void_p,c.c_uint64],
                'mb_runtime_options_set_device':[c.c_void_p,c.c_uint32,c.c_void_p,c.c_uint64],
                'mb_runtime_options_set_threads':[c.c_void_p,c.c_uint32,c.c_void_p,c.c_uint64],
                'mb_sonic_load':[c.c_char_p,c.c_void_p,c.POINTER(c.c_void_p),c.c_void_p,c.c_uint64],
                'mb_physics_create':[c.c_void_p,c.c_char_p,c.c_char_p,c.POINTER(c.c_void_p),c.c_void_p,c.c_uint64],
                'mb_physics_start':[c.c_void_p,c.c_void_p,c.c_uint64,c.c_void_p,c.c_uint64,c.c_void_p,c.c_uint64],
                'mb_physics_step':[c.c_void_p,c.c_void_p,c.c_uint64,c.c_void_p,c.c_uint64,c.c_uint32,c.c_double,c.c_void_p,c.c_uint64,c.c_void_p,c.c_uint64,c.c_void_p,c.c_uint64],
                'mb_physics_status':[c.c_void_p,c.POINTER(c.c_double),c.POINTER(c.c_uint32),c.POINTER(c.c_uint32),c.c_void_p,c.c_uint64]}
    signatures['mb_physics_trace']=[c.c_void_p,c.c_uint32,c.c_void_p,c.c_uint64,c.POINTER(c.c_uint64),c.c_void_p,c.c_uint64]
    for name,types in signatures.items(): getattr(lib,name).argtypes=types;getattr(lib,name).restype=c.c_uint32
    for name in ['mb_physics_free','mb_sonic_free','mb_runtime_options_free']:getattr(lib,name).argtypes=[c.c_void_p]
    def call(name,*values):
        if getattr(lib,name)(*values,error,1024): raise RuntimeError(error.value.decode())
    options,sonic,physics=c.c_void_p(),c.c_void_p(),c.c_void_p()
    rows=[]; elapsed=0;failure=None
    try:
        call('mb_runtime_options_create',c.byref(options));call('mb_runtime_options_set_device',options,1 if args.device=='cpu' else 2);call('mb_runtime_options_set_threads',options,4)
        call('mb_sonic_load',str(args.model).encode(),options,c.byref(sonic));call('mb_physics_create',sonic,str(args.scene).encode(),str(args.config).encode(),c.byref(physics))
        obj=json.loads(args.motion.read_text());print('source keys',list(obj),flush=True)
        motion=obj['motion'] if 'motion' in obj else obj
        roots=motion['roots']; rotations=motion['rotations']; frames=len(roots)//3
        root=(c.c_float*len(roots))(*roots);rot=(c.c_float*len(rotations))(*rotations)
        call('mb_physics_start',physics,root,3,rot,136)
        actual=(c.c_float*90)();reference=(c.c_float*90)();stamp=c.c_double();fallen=c.c_uint32();contacts=c.c_uint32()
        start=time.monotonic()
        for i in range(int((frames-1)/30*50)):
            call('mb_physics_step',physics,root,len(roots),rot,len(rotations),frames,i/50,actual,90,reference,90)
            call('mb_physics_status',physics,c.byref(stamp),c.byref(fallen),c.byref(contacts))
            rows.append(dict(time=stamp.value,actual=list(actual),reference=list(reference),fallen=bool(fallen.value),contacts=contacts.value))
            if args.trace:
                fields=[]
                for field in range(6):
                    count=c.c_uint64();call('mb_physics_trace',physics,field,None,0,c.byref(count))
                    values=(c.c_double*count.value)();call('mb_physics_trace',physics,field,values,count.value,c.byref(count))
                    fields.append(list(values))
                rows[-1]['trace']=fields
            if fallen.value: failure='not tracked: robot fell with current controller/robot/material configuration';break
        elapsed=time.monotonic()-start
    finally:
        lib.mb_physics_free(physics);lib.mb_sonic_free(sonic);lib.mb_runtime_options_free(options)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    import hashlib
    def digest(path):
        with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
    inputs={name:dict(path=str(getattr(args,name)),sha256=digest(getattr(args,name))) for name in ['library','model','scene','config','motion']}
    args.output.write_text(json.dumps(dict(schema=1,device=args.device,failure=failure,elapsed_seconds=elapsed,inputs=inputs,rows=rows)))
    print(json.dumps(dict(samples=len(rows),sim_seconds=rows[-1]['time'],wall_seconds=elapsed,failure=failure)))
    if failure: raise SystemExit(1)

if __name__=='__main__': main()
