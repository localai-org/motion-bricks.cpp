#!/usr/bin/env python3
"""Capture the pinned upstream SOMA model, without reimplementing its inference.

Inputs are the seven ONNX-interface tensors in an NPZ archive. Captures contain
module inputs/outputs plus non-module representation and orchestration taps.
This is the FP32 PyTorch oracle; mixed-FP16 must use the upstream ONNX graph.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class TensorCapture:
    def __init__(self):
        self.tensors = {}
        self.calls = Counter()

    def record(self, name, value):
        if isinstance(value, torch.Tensor):
            # Copy immediately: upstream mutates batch dictionaries and tensor slices.
            self.tensors[name] = value.detach().cpu().numpy().copy()
        elif isinstance(value, np.ndarray):
            self.tensors[name] = value.copy()
        elif isinstance(value, dict):
            for key, child in value.items():
                self.record(f'{name}/{key}', child)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                self.record(f'{name}/{index}', child)

    def wrap(self, stack, obj, attribute, label):
        original = getattr(obj, attribute)
        # Bound methods may come from the class, not the instance dictionary.
        owned = attribute in vars(obj)

        def wrapped(*args, **kwargs):
            index = self.calls[label]
            self.calls[label] += 1
            prefix = f'{label}/{index}'
            self.record(prefix + '/input', args)
            self.record(prefix + '/kwargs', kwargs)
            result = original(*args, **kwargs)
            self.record(prefix + '/output', result)
            return result

        setattr(obj, attribute, wrapped)
        if owned:
            stack.callback(setattr, obj, attribute, original)
        else:
            stack.callback(delattr, obj, attribute)

    def capture_locals(self, stack, function, label, names):
        """Copy selected locals at function return without replacing its arithmetic."""
        code=getattr(function,'__func__',function).__code__
        previous=sys.getprofile()
        if previous is not None and not callable(previous):
            raise ValueError('Cannot compose capture with the installed profiler')
        def profile(frame,event,arg):
            if previous is not None:previous(frame,event,arg)
            if event=='return' and frame.f_code is code:
                index=self.calls[label];self.calls[label]+=1
                for name in names:
                    if name in frame.f_locals:
                        self.record(f'{label}/{index}/{name}',frame.f_locals[name])
        sys.setprofile(profile)
        stack.callback(sys.setprofile,previous)

    def attach(self, stack, model):
        # Disable fused MHA/transformer paths so layer hooks see the real execution.
        fastpath = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)
        stack.callback(torch.backends.mha.set_fastpath_enabled, fastpath)
        for name, module in model.named_modules():
            label = 'layer/' + (name or 'model')
            self.wrap(stack, module, 'forward', label)
            # The wrapper records only when forward is actually used.

        inference = model.model
        self.capture_locals(stack,inference._decode_motions_from_predicted_root_and_pose_tokens,
                            'decision/root_selection',('avg_height_error_from_pose',
                            'avg_height_error_from_root','use_root_module','target_frame_inds',
                            'target_global_height','pred_global_height_from_pose_module',
                            'pred_global_height_from_root_module'))
        for name in ('predict', '_extract_initial_root_info', '_predict_root_trajectories',
                     '_predict_pose_tokens', '_decode_motions_from_predicted_root_and_pose_tokens',
                     '_reapply_initial_root_info'):
            self.wrap(stack, inference, name, 'boundary/' + name)

        visited = set()
        for name in ('global_motion_rep', 'local_motion_rep', 'motion_rep'):
            rep = getattr(inference, name)
            for obj, label in ((rep, name), (getattr(rep, 'dual_rep', None), name + '.dual')):
                if obj is None or id(obj) in visited:
                    continue
                visited.add(id(obj))
                for method in ('normalize', 'unnormalize', 'local_to_global', 'global_to_local'):
                    if hasattr(obj, method):
                        self.wrap(stack, obj, method, 'representation/' + label + '/' + method)

        for name, module in model.named_modules():
            for method in ('get_codes_from_indices', 'forward_decoder'):
                if hasattr(module, method):
                    self.wrap(stack, module, method, 'boundary/' + name + '/' + method)

        for module_name, functions in (
            ('motionbricks_onnx.models.soma.model', ('change_t_pose_global_mats_onnx_compatible',)),
            ('motionbricks_onnx.models.soma.inference', ('global_root_to_local_onnx_compatible',)),
        ):
            module = importlib.import_module(module_name)
            for function in functions:
                self.wrap(stack, module, function, 'geometry/' + function)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    manifest = json.loads(Path(__file__).with_name('soma-reference.json').read_text())
    actual_revision = subprocess.check_output(
        ['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if actual_revision != manifest['source_revision']:
        raise ValueError('Upstream revision differs from pinned parity target')
    # A source-only checkout may lack a configured LFS clean filter. The materialized
    # rig is checked by content hash below, rather than Git's pointer comparison.
    if subprocess.check_output(['git', '-C', str(args.upstream), 'status', '--porcelain',
                                '--untracked-files=no', '--', 'tools/onnx',
                                ':(exclude)tools/onnx/assets/soma_model_rig.npz'], text=True).strip():
        raise ValueError('Upstream exporter has modified tracked files')
    from motionbricks_onnx import checkpoints, checksums
    from motionbricks_onnx.models.soma import model as source
    if Path(source.__file__).resolve() != args.upstream.resolve() / 'tools/onnx/models/soma/model.py':
        raise ValueError('Installed exporter does not come from --upstream')
    rig = args.upstream / 'tools/onnx/assets/soma_model_rig.npz'
    expected = next(x for x in manifest['source_assets'] if x['path'].endswith('soma_model_rig.npz'))
    if sha256(rig) != expected['sha256']:
        raise ValueError('SOMA rig differs from pinned reference')
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    release = checksums.CHECKPOINTS['soma']
    root = checkpoints.checkpoint_root(args.checkpoints, release)
    model = source.build_soma(checkpoints.load_sources(root, release)).eval()
    with np.load(args.inputs, allow_pickle=False) as archive:
        if set(archive.files) != set(source.INPUT_NAMES):
            raise ValueError('Input archive must contain exactly the seven SOMA inputs')
        inputs = [torch.from_numpy(archive[name].copy()) for name in source.INPUT_NAMES]
    capture = TensorCapture()
    capture.record('input', dict(zip(source.INPUT_NAMES, inputs)))
    # Check that tracing itself does not change the observable reference result.
    with torch.inference_mode():
        fastpath_output = model(*inputs)
        # The exporter and layer capture use explicit transformer operations. Compare
        # instrumentation against that same execution path, and report fused-path
        # differences separately rather than attributing them to the hooks.
        torch.backends.mha.set_fastpath_enabled(False)
        baseline = model(*inputs)
        with ExitStack() as stack:
            capture.attach(stack, model)
            outputs = model(*inputs)
    trace_error = {}
    for name, expected_output, output in zip(source.OUTPUT_NAMES, baseline, outputs):
        if not torch.isfinite(output).all():
            raise ValueError(f'Non-finite upstream output: {name}')
        delta = (expected_output.double() - output.double()).abs()
        trace_error[name] = float(delta.max())
        torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
    capture.record('output', dict(zip(source.OUTPUT_NAMES, outputs)))
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez(args.output / 'tensors.npz', **capture.tensors)
    modules = {name or 'model': type(module).__name__ for name, module in model.named_modules()}
    report = dict(schema_version=1, reference=manifest, precision='fp32',
                  torch_version=torch.__version__, numpy_version=np.__version__,
                  threads=args.threads, input_sha256=sha256(args.inputs),
                  capture_sha256=sha256(args.output / 'tensors.npz'),
                  instrumentation_max_absolute_error=trace_error,
                  fused_path_max_absolute_error={name: float((a.double() - b.double()).abs().max())
                      for name, a, b in zip(source.OUTPUT_NAMES, fastpath_output, baseline)},
                  modules=modules, calls=dict(capture.calls),
                  uncalled_modules=[name for name in modules if 'layer/' + name not in capture.calls],
                  tensors={name: dict(shape=list(value.shape), dtype=str(value.dtype))
                           for name, value in capture.tensors.items()})
    (args.output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Captured {len(capture.tensors)} tensors in {args.output}')


if __name__ == '__main__':
    main()
