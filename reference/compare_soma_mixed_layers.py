#!/usr/bin/env python3
"""Inventory native semantic taps against pinned CUDA ONNX mixed-FP16 outputs.

Diagnostic only: no numerical tolerance or parity pass is implied. Shorter
sequences require validated source masks and explicitly mapped temporal axes.
Input identity is not inferred from filenames or shapes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from compare_native_taps import read_gguf


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def correspondence(name, component):
    """Explicit semantic mappings, never choose a node by compatible shape."""
    if component in ('root','pose'):
        for suffix,operation in [('score_product','MatMul_1_output_0'),
                                 ('probabilities','Softmax_output_0'),
                                 ('merged_values','Reshape_12_output_0')]:
            if name.endswith('.self_attn.'+suffix):
                if suffix=='merged_values' and name.startswith('_shared_transformer_model.'):
                    operation='Reshape_9_output_0'
                return (name.removesuffix('.'+suffix),operation),None
    if name.endswith('.in_proj_weight'):
        return None, 'weight conversion, not a layer output'
    if name == '_positioned_input':
        return ('mmm_net_1.Add_2_output_0', None), None
    if component == 'pose' and name == '_proj_local_pose':
        name += '_1'
    op = 'Add_output_0'
    if name in ('_pose_token_emb', '_middle_token_emb', '_proj_input_num_tokens', '_proj_num_valid_positions'):
        op = 'Gather_output_0'
    elif name.endswith('self_attn.out_proj'):
        name = name.removesuffix('.out_proj'); op = 'Gemm_output_0'
    elif name.endswith('self_attn.in_proj_bias'):
        name = name.removesuffix('.in_proj_bias')
    elif 'transformer' in name and name.endswith(('.norm1', '.norm2')):
        op = 'LayerNormalization_output_0'
    elif 'transformer' in name and name.endswith('.dropout'):
        name = name.removesuffix('.dropout'); op = 'Relu_output_0'
    elif name == '_proj_num_token_output_logit':
        op = 'Gemm_output_0'
    elif '.relu_layers.' in name:
        op = 'LeakyRelu_output_0'
    elif name.endswith(('.activation1', '.activation2')):
        op = 'Relu_output_0'
    elif '.model.' in name and name.endswith(('.norm1', '.norm2')):
        return None, 'identity normalization folded out of ONNX; alias not proved'
    elif name.endswith(('.conv1', '.conv2')):
        op = 'Conv_output_0'
    elif '.target_cond_blocks.' in name or '.external_cond_blocks.' in name:
        if int(name.rsplit('.', 1)[1]) % 2: op = 'Relu_output_0'
    elif name.startswith(('decoder.model.', '_conv_output.model.')):
        tail = name.split('.model.', 1)[1]
        if tail in ('0', '2.2', '3.2', '4', '6'): op = 'Conv_output_0'
        elif tail in ('1', '5'): op = 'Relu_output_0'
        elif tail in ('2.1', '3.1'): op = 'Resize_output_0'
    elif name in ('_proj_input.0', '_proj_input.1'):
        if name.endswith('.1'): op = 'Relu_output_0'
        name = '_proj_input.' + name
    return (name, op), None


def temporal_prefix(reference, component, name, actual, expected, count):
    """Only remove known masked temporal padding, never infer axes from sizes."""
    if component == 'pose':
        if name == '_proj_num_valid_positions': return expected, None
        axis = 0; full = 64 if name == '_proj_local_pose' else 16
        valid = count * (full // 16)
        mask_key = 'graph//mmm_net_1/Less_output_0'
    elif component == 'root' and name.startswith('_root_token_transformer_model.'):
        axis = 0; full = 25; valid = 9 + count
        mask_key = 'graph//mmm_net/Less_output_0'
    elif name.startswith(('decoder.', '_conv_output.')):
        mask_key = ('graph//Less_6_output_0' if component == 'decoder'
                    else 'graph//mmm_net/Less_output_0')
        suffix = name.split('.', 1)[1]
        if suffix.startswith('target_cond_blocks.'):
            axis = 0; full = 64; valid = count * 4
        elif suffix.startswith('external_cond_blocks.'):
            block = int(suffix.split('.')[1]); axis = 0 if block % 2 == 0 else 1
            full = 16 * 2**(block // 2); valid = count * 2**(block // 2)
        elif suffix.startswith('model.'):
            parts = suffix.split('.'); block = int(parts[1]); axis = 1
            scale = 1 if block <= 1 else 2 if block == 2 else 4
            if block in (2, 3) and len(parts) > 2 and parts[2] == '0': scale //= 2
            full = 16 * scale; valid = count * scale
        else: return expected, None
    else: return expected, None
    axes = (axis,)
    if (component in ('root', 'pose') and
            name.endswith(('.self_attn.score_product', '.self_attn.probabilities'))):
        if actual.ndim != 3 or expected.ndim != 3:
            return expected, None
        axes = (1, 2)  # Explicit head/query/key layout of attention taps.
    if (actual.ndim != expected.ndim or actual.ndim < 2 or
            any(actual.shape[i] != valid or expected.shape[i] != full for i in axes) or
            any(a != b for i, (a, b) in enumerate(zip(actual.shape, expected.shape)) if i not in axes)):
        return expected, None
    mask = np.asarray(reference[mask_key])
    if mask.shape != (1, 16) or not np.array_equal(mask, (np.arange(16) < count)[None, :]):
        raise ValueError(f'Unexpected upstream temporal mask: {mask_key}')
    selection = [slice(None)] * expected.ndim
    for i in axes: selection[i] = slice(0, valid)
    info = dict(kind='masked_temporal_prefix', valid_length=valid,
                padded_length=full, mask_reference=mask_key)
    info.update(axis=axis) if len(axes) == 1 else info.update(axes=list(axes))
    return expected[tuple(selection)], info


def compare(reference, metadata, native, nodes=()):
    duration = np.asarray(reference['output/pred_num_tokens']).reshape(-1)
    if duration.size != 1 or not 6 <= duration[0] <= 16 or duration[0] != int(duration[0]):
        raise ValueError('Layer mapping requires one valid SOMA duration')
    count = int(duration[0])
    lookup = {}
    candidates = []
    for key in reference.files:
        if not key.startswith('graph//') or '__fp' in key: continue
        path = key[7:]
        if '/' not in path: continue
        module, op = path.rsplit('/', 1)
        module = module.removeprefix('mmm_net/').replace('/', '.')
        pair = (module, op)
        if pair in lookup: raise ValueError(f'Ambiguous ONNX module: {pair}')
        lookup[pair] = key
        # Explicit full graph inventory, including constants/shape operations.
        candidates.append(key)
    producers = {}
    for node in nodes:
        for output in node['outputs']:
            if output in producers: raise ValueError(f'Duplicate ONNX producer: {output}')
            producers[output] = node
    rows = []; used = set(); component = 'root'; execution = 0; previous = -1
    for tap, actual in native.items():
        label = metadata[tap + '.name']
        _, index, name = label.split('/', 2)
        index = int(index)
        if index < previous:
            execution += 1
            if name == '_proj_local_pose': component = 'pose'
        previous = index
        if name.startswith('decoder.'): component = 'decoder'
        row = dict(tap=tap, name=name, component=component, execution=execution,
                   native_operation=metadata[tap + '.operation'])
        pair, reason = correspondence(name, component)
        if reason == 'weight conversion, not a layer output':
            row.update(status='excluded', reason=reason)
            rows.append(row)
            continue
        if pair == ('mmm_net_1.Add_2_output_0', None):
            key = 'graph//mmm_net_1/Add_2_output_0'
        else: key = lookup.get(pair)
        if key is None and '.model.' in name and name.endswith(('.norm1', '.norm2')):
            # Pinned ResConv1DBlock uses Identity normalization. Its exact output
            # is the input to the following activation, including conditioning
            # and casts. Resolve that edge from the captured source graph;
            # never approximate it with the preceding convolution's output.
            activation = name.rsplit('.', 1)[0] + '.activation' + name[-1]
            activation_key = lookup.get((activation, 'Relu_output_0'))
            producer = producers.get(activation_key.removeprefix('graph/') if activation_key else '')
            if producer is not None:
                if producer['op'] != 'Relu' or len(producer['inputs']) != 1:
                    raise ValueError(f'Unexpected folded normalization consumer: {activation}')
                key = 'graph/' + producer['inputs'][0]
                row['correspondence'] = dict(kind='folded_identity_output',
                    consumer=producer['name'], input=producer['inputs'][0],
                    source='tools/onnx/models/layers.py:ResConv1DBlock')
        if key is None or key not in reference:
            row.update(status='unmapped', reason=reason or 'no explicit ONNX correspondence',
                       requested_correspondence=pair)
            rows.append(row); continue
        a = np.squeeze(np.asarray(actual)); b = np.squeeze(reference[key])
        if '.external_cond_blocks.' in name and name.endswith(('.1', '.3')):
            a = a.T
            row['layout_transform'] = 'transpose native time-major activation to channels-major'
        if name == '_pose_token_emb' and a.shape == (count * 5, 32) and b.shape == (16, 5, 32):
            a = a.reshape(count, 5, 32)
            row['layout_transform'] = 'restore token/head axes of flattened embedding gather'
        if a.shape != b.shape:
            row['padded_reference_shape'] = list(b.shape)
            b, prefix = temporal_prefix(reference, component, name, a, b, count)
            if prefix: row['temporal_correspondence'] = prefix
        row.update(reference=key, native_shape=list(a.shape), reference_shape=list(b.shape),
                   reference_dtype=str(b.dtype))
        if a.shape != b.shape:
            row['status'] = 'shape_mismatch'; rows.append(row); continue
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f'Non-finite layer: {label}')
        error = a.astype(np.float64) - b.astype(np.float64)
        row.update(status='compared', values=int(a.size),
                   unequal_values=int(np.count_nonzero(a != b)),
                   max_absolute=float(np.abs(error).max()),
                   rms=float(np.sqrt(np.mean(error * error))))
        used.add(key); rows.append(row)
    return dict(layers=rows, reference_tokens=count, native_taps=len(rows),
                compared_taps=sum(r['status'] == 'compared' for r in rows),
                excluded_weight_conversions=sum(r['status'] == 'excluded' for r in rows),
                semantic_taps=sum(r['status'] != 'excluded' for r in rows),
                semantic_coverage_complete=bool(rows) and all(r['status'] in ('compared', 'excluded') for r in rows)
                    and any(r['status'] == 'compared' for r in rows),
                unique_reference_outputs=len(used),
                unmapped_reference_graph_outputs=[k for k in candidates if k not in used],
                coverage_scope='native named semantic boundaries; not every ONNX graph operation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--native-trace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.reference / 'manifest.json').read_text())
    pin = json.loads(Path(__file__).with_name('soma-reference.json').read_text())
    if manifest['reference'] != pin: raise ValueError('SOMA reference pin mismatch')
    if manifest['precision'] != 'mixed-fp16' or manifest['provider'] != 'CUDAExecutionProvider':
        raise ValueError('Expected official mixed-FP16 CUDA capture')
    archive = args.reference / 'tensors.npz'
    if sha256(archive) != manifest['capture_sha256']: raise ValueError('Capture checksum mismatch')
    metadata, native = read_gguf(args.native_trace)
    if metadata.get('precision') != 'mixed-fp16-experimental':
        raise ValueError('Expected native mixed-FP16 trace')
    with np.load(archive) as reference:
        report = compare(reference, metadata, native, manifest.get('nodes', ()))
    report.update(format='motionbricks-soma-mixed-layer-diagnostic-v1',
                  status='diagnostic; parity not established',
                  input_identity='not verified by this layer-only report',
                  reference=str(args.reference), native_trace=str(args.native_trace),
                  reference_capture_sha256=manifest['capture_sha256'],
                  native_trace_sha256=sha256(args.native_trace),
                  source_revision=pin['source_revision'],
                  reference_runtime_identity=manifest.get('runtime_identity'),
                  native_matmul_operands=metadata.get('arithmetic.matmul_operands'),
                  native_matmul_policy=metadata.get('arithmetic.matmul_policy','unrecorded'),
                  native_root_policy=metadata.get('arithmetic.root_policy','unrecorded'),
                  native_root_projection_policy=metadata.get('arithmetic.root_projection_policy','unrecorded'),
                  native_aligned_block_order=metadata.get('arithmetic.aligned_block_order','unrecorded'),
                  native_nvjet3_policy=metadata.get('arithmetic.nvjet3_policy','unrecorded'),
                  native_nvjet8_policy=metadata.get('arithmetic.nvjet8_policy','unrecorded'),
                  native_pose_scores_policy=metadata.get('arithmetic.pose_scores_policy','unrecorded'),
                  native_root_conv_policy=metadata.get('arithmetic.root_conv_policy','unrecorded'),
                  native_conditioning_policy=metadata.get('arithmetic.conditioning_policy','unrecorded'),
                  native_decoder_conv_policy=metadata.get('arithmetic.decoder_conv_policy','unrecorded'),
                  native_nvjet_split4_policy=metadata.get('arithmetic.nvjet_split4_policy','unrecorded'),
                  native_norm_policy=metadata.get('arithmetic.norm_policy','unrecorded'),
                  rsqrt_table_sha256=metadata.get('arithmetic.rsqrt_table_sha256'),
                  root_softmax_policy=metadata.get('arithmetic.root_softmax_policy','unrecorded'),
                  pose_softmax_policy=metadata.get('arithmetic.pose_softmax_policy','unrecorded'),
                  exp2_table_sha256=metadata.get('arithmetic.exp2_table_sha256'))
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f"Compared {report['compared_taps']}/{report['native_taps']} native taps; "
          f"{report['unique_reference_outputs']} unique reference outputs; diagnostic only")


if __name__ == '__main__': main()
