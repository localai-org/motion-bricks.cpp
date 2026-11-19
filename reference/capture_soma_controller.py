#!/usr/bin/env python3
"""Record the upstream controller with its FP32 PyTorch or mixed-FP16 ONNX oracle.

Only the TensorRT inference transport is replaced. Encoding, decoding, endpoint
correction, character/state machine, targets, playback and history are upstream.
No native library is loaded into this reference process.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--checkpoints', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=55)
    p.add_argument('--onnx-fp16', type=Path, help='Official pinned mixed-FP16 graph; runs with ONNX Runtime CUDA')
    args = p.parse_args()
    pin = json.loads(Path(__file__).with_name('soma-reference.json').read_text())
    root = args.upstream.resolve()
    if subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip() != pin['source_revision']:
        raise ValueError('Upstream revision mismatch')
    if subprocess.check_output(['git', '-C', str(root), 'diff', '--name-only', 'HEAD', '--', 'app', 'motionbricks', 'tools/onnx', ':(exclude)**/assets/**'], text=True).strip():
        raise ValueError('Modified upstream source')
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['PYGLET_HEADLESS'] = 'true'
    import pyglet
    pyglet.options['headless'] = True
    import warp as wp
    wp.config.kernel_cache_dir = str(Path('.cache/soma-warp').resolve())
    wp.init(); wp.set_device('cpu')
    sys.path.insert(0, str(root))
    import torch
    from motionbricks_onnx import checkpoints, checksums
    from motionbricks_onnx.models.soma import model as source
    from motionbricks.model.soma.engine import MotionBricksSOMAEngine
    from motionbricks.model.config import MotionBricksInferenceConfig
    from motionbricks.model.contract import ModelRig, valid_frame_count
    from motionbricks.io.usd import load_skeleton_from_usd
    from app.runtime.common.clip_registry import ClipRegistry
    from app.runtime.common.io.smart_primitive_document import SmartPrimitiveAssetPool
    from app.runtime.common.rig import model_to_runtime
    from app.runtime.common.motion import LocomotionCommand
    from app.runtime.soma.character import SOMACharacter
    from app.runtime.soma.presets import DEFAULT_LOCOMOTION_MODES, DEFAULT_IDLE_CLIPS, DEFAULT_CROUCH_IDLE_CLIPS
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.mha.set_fastpath_enabled(False)
    release = checksums.CHECKPOINTS['soma']
    weights = checkpoints.checkpoint_root(args.checkpoints, release)
    oracle_identity = dict(precision='fp32', backend='upstream-fp32-pytorch-and-warp-cpu')
    session = None
    if args.onnx_fp16:
        expected = next(x for x in pin['source_assets'] if x['path'].endswith('_fp16.onnx'))
        if digest(args.onnx_fp16) != expected['sha256']:
            raise ValueError('Mixed-FP16 graph checksum mismatch')
        import onnxruntime as ort
        sys.path.insert(0,str(Path(__file__).resolve().parent))
        from capture_soma_onnx import runtime_identity
        os.environ['ORT_CUDA_GEMM_OPTIONS'] = '0'
        ort.preload_dlls()
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(args.onnx_fp16), options, providers=['CUDAExecutionProvider'])
        if session.get_providers()[0] != 'CUDAExecutionProvider':
            raise ValueError('FP16 reference fell back from CUDA')
        oracle_identity = dict(precision='mixed-fp16', backend='upstream-mixed-fp16-onnx-cuda-and-warp-cpu',
            graph_sha256=expected['sha256'], runtime=runtime_identity('CUDAExecutionProvider',session.get_provider_options()))
    else:
        model = source.build_soma(checkpoints.load_sources(weights, release)).eval()
    tick = 0
    events = []
    arrays = {}

    class OracleEngine(MotionBricksSOMAEngine):
        def __init__(self):
            self._model_to_runtime = model_to_runtime(ModelRig.SOMA)
            self._runtime_to_model = wp.transform_inverse(self._model_to_runtime)
            self._alignment = wp.empty(1, dtype=wp.transform)
            self._generated_frame_count = wp.empty(1, dtype=wp.int32)
            self._initialize_pose_layout()
            self.buffers = {}
            for name, shape in [('global_root_values', (1,8,5)), ('local_root_values',(1,8,4)), ('local_poses',(1,8,267)), ('has_global_root_values',(1,8)), ('has_local_root_values',(1,8)), ('has_local_poses',(1,8)), ('allowed_pred_num_tokens',(1,11))]:
                self.buffers[name] = wp.zeros(shape, dtype=wp.int32 if name.startswith(('has_', 'allowed_')) else wp.float32)
        @property
        def device(self): return wp.get_device('cpu')
        @property
        def stream(self): return None
        def _ensure_open(self): pass
        def device_tensor(self, name): return self.buffers[name]
        def close(self): pass
        def generate_animation(self, context, target, output, *, config=None):
            config = config or MotionBricksInferenceConfig()
            self._prepare_generation(context, target, output)
            self._encode_guide_poses(context, target, config)
            inputs = [self.buffers[n].numpy().copy() for n in source.INPUT_NAMES]
            if session is None:
                with torch.inference_mode(): results = [x.numpy().copy() for x in model(*[torch.from_numpy(x) for x in inputs])]
            else:
                results = session.run(list(source.OUTPUT_NAMES), dict(zip(source.INPUT_NAMES,inputs)))
            for name, value in zip(source.OUTPUT_NAMES, results):
                a = value.copy()
                self.buffers[name] = wp.array(a, dtype=wp.int32 if a.dtype.kind in 'iu' else wp.float32)
            self._decode_generated_poses(output)
            if config.correct_generated_diff: self._post_process_generated_poses(context, target, output)
            tokens = int(self.buffers['pred_num_tokens'].numpy().item())
            output.num_frames = valid_frame_count(tokens)
            index = len(events)
            prefix = f'event_{index:04d}'
            for name, a in [('context',context.poses.numpy()), ('target',target.poses.numpy()), ('motion',output.poses.numpy()[:output.num_frames]), ('contacts',output.foot_contacts.numpy()[:output.num_frames])]:
                arrays[prefix+'_'+name] = a.copy()
            for name, value in zip(source.INPUT_NAMES, inputs): arrays[prefix+'_input_'+name] = value
            events.append(dict(tick=tick, tokens=tokens, frames=output.num_frames, allowed=list(config.allowed_token_counts), heading=list(config.heading_alignment)))
            print(f'generate {index}: tick={tick} tokens={tokens}', flush=True)
            return output

    skeleton = load_skeleton_from_usd(root/'motionbricks/assets/human/soma_base_skel_minimal_no_root.usd', '/OUTPUT/c_skeleton_grp/Hips')
    registry = ClipRegistry(ModelRig.SOMA, skeleton=skeleton)
    engine = OracleEngine()
    pool = SmartPrimitiveAssetPool(registry, skeleton.reference_local_transforms, engine._model_to_runtime)
    # One documented idle clip per stance makes stop/restart reproducible without
    # replacing upstream's idle-transition or playback policy.
    character = SOMACharacter(engine, asset_pool=pool,
        idle_paths=DEFAULT_IDLE_CLIPS[:1], crouch_idle_paths=DEFAULT_CROUCH_IDLE_CLIPS[:1])
    def save_array(name, value):
        data = np.ascontiguousarray(value, dtype='<f4')
        data.tofile(args.output/(name+'.f32'))
        return dict(file=name+'.f32', shape=list(data.shape), sha256=digest(args.output/(name+'.f32')))
    mode_assets = []
    for definition in DEFAULT_LOCOMOTION_MODES:
        state = character.state_machine.crouching_locomotion_state if definition.name == 'crouch_directional_2d' else character.state_machine.standing_locomotion_state
        runtime = state.mode_runtimes[definition.name]
        mode_assets.append(dict(name=definition.name, label=definition.label, directional=definition.blend_space_type.value=='directional_2d',
            max_speed=definition.max_speed, default_speed=definition.default_speed, minimum_speed=definition.minimum_speed,
            velocity_scale=definition.velocity_scale, position_scale=definition.position_scale,
            early=definition.transition_early_by_frames, allowed=list(definition.allowed_token_counts),
            values=[list(v) for _,v in definition.target_values], targets=save_array(definition.name, runtime.target_poses.numpy())))
    clips=[]
    for name, state in [('standing',character.state_machine.standing_idle_state), ('crouching',character.state_machine.crouching_idle_state)]:
        clip=state.clips[0]
        clips.append(dict(name=name, rate=clip.animation.sample_rate, duration=clip.duration,
            source_ground=list(clip.source_ground), poses=save_array(name,clip.animation.poses.numpy()[:clip.animation.num_frames])))
    asset_doc=dict(format='motionbricks-controller-assets-v1', source_revision=pin['source_revision'],
        joints=skeleton.num_joints, names=skeleton.joint_names, parents=skeleton.parent_indices.numpy().tolist(),
        reference=skeleton.reference_local_transforms.numpy().reshape(-1).tolist(),
        model_to_pose=engine._model_to_pose.numpy().tolist(), model_to_runtime=list(engine._model_to_runtime),
        modes=mode_assets, clips=clips)
    (args.output/'controller-assets.json').write_text(json.dumps(asset_doc,separators=(',',':'))+'\n')
    arrays['initial_pose'] = character.current_pose.numpy().copy()
    original_generate = character.player.generate
    def generate(target, params, **kw):
        result = original_generate(target, params, **kw)
        events[-1]['transition_early_by_frames'] = params.transition_early_by_frames
        events[-1]['state'] = character.state_machine.state.name
        return result
    character.player.generate = generate
    # Fixed 50 Hz input stream: acceleration, turn/reversal, every style,
    # directional strafing, crouch transitions, and stop/restart.
    segments = [
        (0, 'linear_1d', None, [1,0,0], False),
        (1, 'linear_1d', [1,0,0], [1,0,0], False),
        (4, 'linear_1d', [1,1,0], [1,0,0], False),
        (7, 'linear_1d', [-1,0,0], [1,0,0], True),
        (10, 'directional_2d', [0,1,0], [1,0,0], False),
        (13, 'crouch_directional_2d', [-1,0,0], [1,0,0], False),
        (16, 'crouch_directional_2d', None, [1,0,0], False),
        (18, 'linear_1d', [1,0,0], [1,0,0], False),
        (21, 'skipping_1d', [1,0,0], [1,0,0], False),
        (25, 'vogue_1d', [0,1,0], [0,1,0], False),
        (29, 'injured_torso_1d', [0,1,0], [0,1,0], False),
        (33, 'injured_leg_1d', [1,0,0], [1,0,0], False),
        (37, 'zombie_1d', [1,0,0], [1,0,0], False),
        (41, 'combat_1d', [-1,0,0], [-1,0,0], False),
        (45, 'combat_1d', None, [-1,0,0], False),
        (48, 'linear_1d', [.4,0,0], [1,0,0], False),
        (52, 'linear_1d', None, [1,0,0], False),
    ]
    modes = {m.name for m in DEFAULT_LOCOMOTION_MODES}
    if any(s[1] not in modes for s in segments): raise ValueError(f'Invalid mode names; expected {modes}')
    records, poses = [], []
    for tick in range(round(args.seconds*50)):
        segment = max((s for s in segments if s[0] <= tick/50), key=lambda s:s[0])
        _, mode, move, facing, fast = segment
        command = LocomotionCommand(np.array(facing,np.float32), None if move is None else np.array(move,np.float32), fast, mode)
        before = len(events)
        character.step(.02, command)
        m = character.movement
        records.append(dict(tick=tick, mode=mode, move=move, facing=facing, fast=fast,
            state=character.state_machine.state.name, generated=len(events)>before,
            speed=m.desired_speed, velocity=m.desired_velocity.tolist(), current_facing=m.current_facing.tolist(),
            ground=character.character_ground_position.tolist(), actual_velocity=character.actual_velocity.tolist(),
            playback_time=character.player.playback_time, transition=character.player.playback_duration))
        poses.append(character.current_pose.numpy().copy())
    arrays['poses'] = np.array(poses)
    arrays['parents'] = skeleton.parent_indices.numpy()
    arrays['reference'] = skeleton.reference_local_transforms.numpy()
    arrays['model_to_runtime'] = np.array(engine._model_to_runtime)
    arrays['model_to_pose'] = engine._model_to_pose.numpy()
    binary_arrays = {name:save_array(name, value) for name,value in arrays.items()}
    np.savez_compressed(args.output/'capture.npz', **arrays)
    manifest = dict(format='motionbricks-controller-capture-v1', source_revision=pin['source_revision'],
        reference_backend=oracle_identity['backend'], oracle=oracle_identity, idle_selection='first clip per stance', arrays=binary_arrays, dt=.02, frames=records, events=events,
        capture_sha256=digest(args.output/'capture.npz'), segments=segments,
        modes=[dict(name=m.name, max_speed=m.max_speed, default_speed=m.default_speed, minimum_speed=m.minimum_speed,
                    velocity_scale=m.velocity_scale, position_scale=m.position_scale, allowed=list(m.allowed_token_counts),
                    transition_early_by_frames=m.transition_early_by_frames) for m in DEFAULT_LOCOMOTION_MODES])
    (args.output/'manifest.json').write_text(json.dumps(manifest,separators=(',',':'))+'\n')


if __name__ == '__main__': main()
