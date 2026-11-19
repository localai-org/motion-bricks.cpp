#!/usr/bin/env python3
"""Exercise a persistent public SOMA model against a captured independent chain.

This validates public outputs and duration decisions. Pose tokens and internal
layer taps remain covered by the internal chain harness, not this API test.
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import numpy as np
from compare_native_taps import read_gguf
from compare_soma_playback import transform_errors


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class SomaAPI:
    def __init__(self, library, bundle, backend, rsqrt=None, exp2=None):
        self.lib = C.CDLL(str(library.resolve()))
        p, u, n, text = C.c_void_p, C.c_uint32, C.c_uint64, C.c_char_p
        signatures = {
            'runtime_options_create': [C.POINTER(p), text, n],
            'runtime_options_set_device': [p,u,text,n],
            'runtime_options_set_threads': [p,u,text,n],
            'runtime_options_set_model_kind': [p,u,text,n],
            'model_load': [text,p,C.POINTER(p),text,n],
            'model_set_arithmetic_calibration': [p,text,text,text,n],
            'inference_request_create': [C.POINTER(p),text,n],
            'inference_request_set_skeleton': [p,u,p,p,n,p,n,u,u,text,n],
            'inference_request_set_boundary_transforms': [p,u,p,n,text,n],
            'inference_request_set_precision': [p,u,text,n],
            'inference_request_set_heading': [p,C.c_float,text,n],
            'inference_request_set_mask': [p,u,p,n,text,n],
            'model_infer': [p,p,C.POINTER(p),text,n],
            'motion_get_frame_count': [p,C.POINTER(n),text,n],
            'motion_get_joint_count': [p,C.POINTER(n),text,n],
            'motion_get_token_count': [p,C.POINTER(u),text,n],
            'motion_copy_field': [p,u,p,n,C.POINTER(n),text,n],
        }
        for name, arguments in signatures.items():
            fn = getattr(self.lib, 'mb_' + name)
            fn.argtypes, fn.restype = arguments, C.c_uint32
        for name in ('runtime_options','model','inference_request','motion'):
            fn = getattr(self.lib, 'mb_' + name + '_free')
            fn.argtypes, fn.restype = [p], None
        self.error=C.create_string_buffer(2048)
        self.model,self.request=p(),p()
        options=p()
        try:
            self.call('runtime_options_create',C.byref(options))
            self.call('runtime_options_set_device',options,{'cpu':1,'vulkan':2}[backend])
            self.call('runtime_options_set_threads',options,1)
            self.call('runtime_options_set_model_kind',options,1)
            self.call('model_load',str(bundle).encode(),options,C.byref(self.model))
            self.call('inference_request_create',C.byref(self.request))
            if rsqrt is not None:
                self.call('model_set_arithmetic_calibration',self.model,str(rsqrt).encode(),str(exp2).encode())
        except BaseException:
            self.close()
            raise
        finally:
            self.lib.mb_runtime_options_free(options)

    def call(self,name,*args):
        self.check(getattr(self.lib,'mb_'+name)(*args,self.error,len(self.error)))

    def check(self,status):
        if status:raise RuntimeError(f'Inference API status {status}: {self.error.value.decode()}')

    def close(self):
        self.lib.mb_inference_request_free(self.request)
        self.lib.mb_model_free(self.model)
        self.request,self.model=C.c_void_p(),C.c_void_p()

    def set_skeleton(self,fixture):
        parents=np.ascontiguousarray(fixture['parents'],dtype=np.int32)
        reference=np.ascontiguousarray(fixture['reference'],dtype=np.float32)
        mapping=np.ascontiguousarray(fixture['model_to_pose'],dtype=np.uint32)
        heading=np.asarray(fixture['heading_joints']).reshape(-1)
        self.call('inference_request_set_skeleton',self.request,parents.size,parents.ctypes.data,
            reference.ctypes.data,reference.size,mapping.ctypes.data,mapping.size,int(heading[0]),int(heading[1]))

    def generate(self,context,target,precision,allowed=2047,heading=None):
        if not 0<allowed<=2047:raise ValueError('invalid duration mask')
        for boundary,values in enumerate((context,target)):
            values=np.ascontiguousarray(values,dtype=np.float32)
            self.call('inference_request_set_boundary_transforms',self.request,boundary,values.ctypes.data,values.size)
        mask=np.array([(allowed>>i)&1 for i in range(11)],dtype=np.uint32)
        self.call('inference_request_set_mask',self.request,3,mask.ctypes.data,11)
        self.call('inference_request_set_precision',self.request,precision)
        self.call('inference_request_set_heading',self.request,0 if heading is None else heading)
        motion=C.c_void_p()
        self.call('model_infer',self.model,self.request,C.byref(motion))
        try:
            frames,joints,tokens=C.c_uint64(),C.c_uint64(),C.c_uint32()
            for name,value in [('frame',frames),('joint',joints),('token',tokens)]:
                self.call('motion_get_'+name+'_count',motion,C.byref(value))
            result={'tokens':tokens.value}
            for field,name in enumerate(('corrected','world','features','contacts')):
                count=C.c_uint64()
                self.call('motion_copy_field',motion,field,None,0,C.byref(count))
                values=np.empty(count.value,dtype=np.float32)
                self.call('motion_copy_field',motion,field,values.ctypes.data,values.size,C.byref(count))
                shape=(frames.value,joints.value,7) if field<2 else (-1,366 if field==2 else 4)
                result[name]=values.reshape(shape)
            return result
        finally:self.lib.mb_motion_free(motion)


def compare(actual, expected, limits):
    metrics, passed = {}, actual['tokens'] == int(np.asarray(expected['tokens']).item())
    for name in ('corrected', 'world'):
        reference = np.asarray(expected[name]).reshape(-1, actual[name].shape[1], 7)
        length_matches = len(reference) == len(actual[name])
        count = min(len(reference), len(actual[name]))
        distance, angle = transform_errors(reference[:count], actual[name][:count])
        metrics[name] = dict(position_m=float(distance.max()), rotation_deg=float(angle.max()),
            root_position_m=float(distance[:, 0].max()), frames_match=length_matches,
            per_frame=dict(root_position_m=distance[:, 0].tolist(),
                           max_joint_position_m=distance.max(axis=1).tolist(),
                           max_joint_rotation_deg=angle.max(axis=1).tolist()))
        passed &= length_matches and distance.max() <= limits['position_m'] and angle.max() <= limits['rotation_deg']
    contacts = np.asarray(expected['contacts']).reshape(-1, 4)
    count = min(len(contacts), len(actual['contacts']))
    error = np.abs(contacts[:count] - actual['contacts'][:count])
    metrics['contacts_absolute'] = float(error.max())
    passed &= len(contacts) == len(actual['contacts']) and np.isfinite(error).all() and error.max() <= limits['contacts_absolute']
    metrics['duration_matches'] = actual['tokens'] == int(np.asarray(expected['tokens']).item())
    return dict(passed=bool(passed), metrics=metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('library', 'bundle', 'reference-chain', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--backend', choices=('cpu', 'vulkan'), default='cpu')
    parser.add_argument('--rsqrt-table', type=Path)
    parser.add_argument('--exp2-table', type=Path)
    parser.add_argument('--runtime-artifact', type=Path, action='append', default=[])
    parser.add_argument('--load-cpu-first', action='store_true', help='Exercise CPU-first backend discovery before Vulkan loading')
    args = parser.parse_args()
    report = json.loads((args.reference_chain / 'report.json').read_text())
    contract = json.loads(Path(__file__).with_name('soma-reference.json').read_text())
    limits_path = Path(__file__).with_name('soma-chain-tolerances.json')
    limits = json.loads(limits_path.read_text())
    if report['format'] != 'motionbricks-soma-chain-v1' or report['source_revision'] != contract['source_revision']:
        parser.error('Expected a pinned independent SOMA chain')
    if report['precision'] not in ('fp32', 'mixed-fp16') or not report['steps']:
        parser.error('Unsupported precision')
    mixed = report['precision'] == 'mixed-fp16'
    if mixed and args.backend == 'cpu' and (args.rsqrt_table is None or args.exp2_table is None):
        parser.error('Mixed-FP16 public API requires CPU and both calibration tables')
    if args.backend == 'vulkan' and (args.rsqrt_table is not None or args.exp2_table is not None):
        parser.error('Vulkan mixed-FP16 does not use CPU calibration tables')
    if not mixed and (args.rsqrt_table is not None or args.exp2_table is not None):
        parser.error('Calibration tables are only used by mixed-FP16 checks')
    args.output.mkdir(parents=True, exist_ok=False)
    runtime_hashes = {str(path.resolve()): digest(path) for path in [args.library, *args.runtime_artifact]}
    cpu_first = SomaAPI(args.library, args.bundle, 'cpu') if args.load_cpu_first else None
    try:
        api = SomaAPI(args.library, args.bundle, args.backend, args.rsqrt_table, args.exp2_table)
    finally:
        if cpu_first is not None:
            cpu_first.close()
    results, previous, skeleton_identity, reference_previous = [], None, None, None
    try:
        for index, step in enumerate(report['steps']):
            if step['index'] != index:
                raise ValueError('Nonsequential chain steps')
            source = Path(step['reference_step'])
            fixture_path = source / 'playback/playback.gguf'
            if digest(fixture_path) != step['reference_fixture_sha256']:
                raise ValueError('Independent playback fixture checksum mismatch')
            capture = json.loads((source / 'capture/manifest.json').read_text())
            if (capture['reference'] != contract or capture['precision'] != report['precision'] or
                digest(source / 'capture/tensors.npz') != step['reference_capture_sha256'] or
                capture['capture_sha256'] != step['reference_capture_sha256'] or
                (mixed and capture.get('provider') != 'CUDAExecutionProvider')):
                raise ValueError('Independent neural capture provenance mismatch')
            playback_manifest = json.loads((source / 'playback/manifest.json').read_text())
            if (playback_manifest['source_revision'] != contract['source_revision'] or
                    playback_manifest['playback_implementation'] != 'upstream-warp-cpu'):
                raise ValueError('Playback must use the pinned upstream implementation')
            _, fixture = read_gguf(fixture_path)
            with np.load(source / 'capture/tensors.npz') as tensors:
                reference_tokens = int(tensors['output/pred_num_tokens'].item())
            if (int(np.asarray(fixture['tokens']).item()) != reference_tokens or
                    playback_manifest['tokens'] != reference_tokens or
                    playback_manifest['frames'] != 4 * reference_tokens - 6):
                raise ValueError('Playback duration differs from neural capture')
            identity = tuple(np.asarray(fixture[key]).tobytes() for key in
                             ('parents', 'reference', 'model_to_pose', 'heading_joints'))
            if skeleton_identity is None:
                skeleton_identity = identity
                api.set_skeleton(fixture)
            elif identity != skeleton_identity:
                raise ValueError('Skeleton changed within the reference chain')
            joints = np.asarray(fixture['parents']).size
            context = np.asarray(fixture['context']).reshape(4, joints, 7)
            if reference_previous is not None and not np.array_equal(context, reference_previous):
                raise ValueError('Reference chain did not feed back its corrected context')
            reference_previous = np.asarray(fixture['corrected']).reshape(-1, joints, 7)[-4:].copy()
            target = np.asarray(fixture['target']).reshape(-1, joints, 7)[:int(np.asarray(fixture['target_count']).item())]
            row = dict(index=index, reference_step=str(source), fixture_sha256=digest(fixture_path), comparisons={})
            arrays = {}
            for mode in ('replay', 'closed_loop'):
                actual_context = previous if mode == 'closed_loop' and previous is not None else context
                fp32_before = api.generate(actual_context, target, 0) if mixed else None
                actual = api.generate(actual_context, target, (3 if args.backend == 'vulkan' else 2) if mixed else 0)
                if mixed:
                    fp32_after = api.generate(actual_context, target, 0)
                    if any(not np.array_equal(fp32_before[key], fp32_after[key]) for key in fp32_before):
                        raise ValueError('Mixed-FP16 request changed subsequent FP32 results')
                row['comparisons'][mode] = compare(actual, fixture, limits)
                arrays[mode + '_context'] = actual_context.copy()
                for key, value in actual.items():
                    arrays[mode + '_' + key] = value
                if mode == 'closed_loop':
                    previous = actual['corrected'][-4:].copy()
            path = args.output / f'step-{index:03d}.npz'
            np.savez(path, **arrays)
            row['outputs_sha256'] = digest(path)
            results.append(row)
            print(f'Public API step {index}: ' + str(all(x['passed'] for x in row['comparisons'].values())), flush=True)
    finally:
        api.close()
    if any(digest(Path(path)) != value for path, value in runtime_hashes.items()):
        raise ValueError('Runtime artifact changed during public API validation')
    output = dict(format='motionbricks-soma-public-api-chain-v1', source_revision=contract['source_revision'],
        precision=report['precision'], backend=args.backend, library_sha256=digest(args.library),
        fp32_scope_restoration_checked=mixed,
        loaded_cpu_first=args.load_cpu_first,
        runtime_artifacts=runtime_hashes,
        bundle_manifest_sha256=digest(args.bundle / 'manifest.json'),
        reference_report_sha256=digest(args.reference_chain / 'report.json'),
        tolerances_sha256=digest(limits_path), steps=results,
        pose_tokens='not exposed by public API; validated separately by internal chain captures',
        passed=all(x['passed'] for row in results for x in row['comparisons'].values()))
    if mixed and args.backend == 'cpu':
        output.update(rsqrt_table_sha256=digest(args.rsqrt_table), exp2_table_sha256=digest(args.exp2_table))
    (args.output / 'report.json').write_text(json.dumps(output, indent=2) + '\n')
    raise SystemExit(0 if output['passed'] else 1)


if __name__ == '__main__':
    main()
