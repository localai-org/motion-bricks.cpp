#!/usr/bin/env python3
"""Report same-input SOMA capture errors; this does not declare a parity pass.

Compare raw model features on the valid prefix shared by each pair of requests.
Different predicted lengths are explicit mismatches, never silently truncated
into a passing comparison. Rotations use angular distance, not quaternion signs.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size or not np.isfinite(values).all():
        raise ValueError('Empty or non-finite error measurements')
    return dict(max=float(values.max()), mean=float(values.mean()),
                rms=float(np.sqrt(np.mean(values * values))), p95=float(np.quantile(values, .95)))


def duration_decision(logits, allowed, actual_tokens):
    logits=np.asarray(logits).reshape(-1).astype(np.float64)
    allowed=np.asarray(allowed).reshape(-1)
    if (logits.shape!=(12,) or allowed.shape!=(11,) or not np.isfinite(logits).all() or
            not np.isin(allowed,[0,1]).all() or not allowed.any() or not 6<=actual_tokens<=16):
        raise ValueError('Invalid duration decision inputs')
    eligible=np.flatnonzero(allowed)
    # Stable ordering reproduces argmax's first-index tie break. Class 12 is
    # the explicitly disabled out-of-reach class, not a duration choice.
    ranked=eligible[np.argsort(-logits[eligible],kind='stable')]
    choice=int(ranked[0])+6
    runner_up=int(ranked[1])+6 if len(ranked)>1 else None
    return dict(tokens=int(actual_tokens),argmax_tokens=choice,
                selection_consistent=choice==actual_tokens,runner_up_tokens=runner_up,
                winner_margin=float(logits[ranked[0]]-logits[ranked[1]]) if len(ranked)>1 else None,
                allowed_tokens=(eligible+6).tolist(),logits=logits.tolist())


def compare_duration_decisions(reference, native, metadata):
    key='layer/model._mmm_root_model.mmm_net._proj_num_token_output_logit/0/output'
    if key not in reference:key='graph//mmm_net/_proj_num_token_output_logit/Gemm_output_0'
    allowed=reference['input/allowed_pred_num_tokens'].reshape(-1)
    # Older pose-to-pose runners always used all eleven duration choices.
    bits=metadata.get('soma.allowed_token_bits',2047)
    if not isinstance(bits,(int,np.integer)) or not 0<bits<2048:
        raise ValueError('Invalid native duration mask metadata')
    native_allowed=np.array([(bits>>i)&1 for i in range(11)])
    return dict(reference=duration_decision(reference[key],allowed,int(reference['output/pred_num_tokens'].item())),
                native=duration_decision(native['duration_logits'],native_allowed,metadata['soma.tokens']),
                allowed_masks_match=bool(np.array_equal(allowed,native_allowed)))


def rotation_matrices(six):
    first, second = six[..., :3], six[..., 3:]
    first = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-12)
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    second = second / np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-12)
    return np.stack((first, second, np.cross(first, second)), axis=-2)


def compare(reference, actual):
    ref_tokens = reference['output/pred_num_tokens'].reshape(-1)
    got_tokens = actual['output/pred_num_tokens'].reshape(-1)
    ref = reference['output/pred_global_motions'].astype(np.float64)
    got = actual['output/pred_global_motions'].astype(np.float64)
    if ref.shape != got.shape or ref.shape[1:] != (64, 366) or ref_tokens.shape != got_tokens.shape:
        raise ValueError('Incompatible SOMA output shapes')
    for tokens in (ref_tokens, got_tokens):
        if len(tokens) != ref.shape[0] or not np.all((tokens >= 6) & (tokens <= 16) & (tokens == np.floor(tokens))):
            raise ValueError('Invalid predicted token counts')
    requests = []
    for index, (a, b) in enumerate(zip(ref_tokens, got_tokens)):
        count = 4 * int(min(a, b))
        x, y = ref[index, :count], got[index, :count]
        positions_x = x[:, 5:92].reshape(count, 29, 3).copy()
        positions_y = y[:, 5:92].reshape(count, 29, 3).copy()
        # RIC positions are relative to the root's ground position; height is absolute.
        positions_x[:, :, [0, 2]] += x[:, None, [0, 2]]
        positions_y[:, :, [0, 2]] += y[:, None, [0, 2]]
        rx = rotation_matrices(x[:, 92:272].reshape(count, 30, 6))
        ry = rotation_matrices(y[:, 92:272].reshape(count, 30, 6))
        cosine = np.clip((np.sum(rx * ry, axis=(-1, -2)) - 1) / 2, -1, 1)
        angle = np.rad2deg(np.arccos(cosine))
        roots = np.linalg.norm(x[:, :3] - y[:, :3], axis=-1)
        joints = np.linalg.norm(positions_x - positions_y, axis=-1)
        requests.append(dict(request=index, reference_tokens=int(a), actual_tokens=int(b),
                             token_count_matches=bool(a == b), compared_raw_frames=count,
                             features_absolute=summary(np.abs(x-y)), root_position_m=summary(roots),
                             joint_position_m=summary(joints), joint_rotation_deg=summary(angle),
                             contacts_absolute=summary(np.abs(x[:, -4:] - y[:, -4:])),
                             per_frame=dict(root_position_m=roots.tolist(),
                                            max_joint_position_m=joints.max(axis=-1).tolist(),
                                            max_joint_rotation_deg=angle.max(axis=-1).tolist())))
    return dict(token_count_mismatches=int(np.count_nonzero(ref_tokens != got_tokens)), requests=requests)


def compare_pose_decisions(reference, actual):
    keys=('graph//mmm_net/_proj_pose_output_logit/Add_output_0',
          'layer/model._mmm_pose_model.mmm_net/0/output/pose_logits')
    logits=[next((data[key] for key in keys if key in data),None) for data in (reference,actual)]
    if any(value is None for value in logits):return None
    durations=[np.asarray(data['output/pred_num_tokens']).reshape(-1) for data in (reference,actual)]
    if durations[0].shape!=durations[1].shape:raise ValueError('Pose request counts differ')
    arrays=[np.asarray(value).reshape(len(durations[0]),16,5,16) for value in logits]
    rows=[]
    for index,(a,b) in enumerate(zip(*durations)):
        if not 6<=a<=16 or not 6<=b<=16:raise ValueError('Invalid pose duration')
        count=int(min(a,b));left=arrays[0][index,:count];right=arrays[1][index,:count]
        if not np.isfinite(left).all() or not np.isfinite(right).all():raise ValueError('Nonfinite pose logits')
        rows.append(dict(request=index,compared_pose_tokens=count*5,
                         mismatches=int(np.count_nonzero(left.argmax(-1)!=right.argmax(-1)))))
    return dict(requests=rows,mismatches=sum(row['mismatches'] for row in rows),
                compared_pose_tokens=sum(row['compared_pose_tokens'] for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--actual', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifests = [json.loads((path / 'manifest.json').read_text()) for path in (args.reference, args.actual)]
    pin=json.loads(Path(__file__).with_name('soma-reference.json').read_text())
    for path,manifest in zip((args.reference,args.actual),manifests):
        if manifest['reference']!=pin:raise ValueError('Capture is not from the pinned reference')
        with (path/'tensors.npz').open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=manifest['capture_sha256']:
                raise ValueError('Capture checksum mismatch')
    if manifests[0]['input_sha256'] != manifests[1]['input_sha256']:
        raise ValueError('Captures use different inputs; use a separate closed-loop comparison')
    with np.load(args.reference / 'tensors.npz', allow_pickle=False) as reference, np.load(
        args.actual / 'tensors.npz', allow_pickle=False
    ) as actual:
        report = compare(reference, actual)
        report['pose_decisions']=compare_pose_decisions(reference,actual)
    report.update(schema_version=1, reference=str(args.reference), actual=str(args.actual),
                  reference_precision=manifests[0]['precision'], actual_precision=manifests[1]['precision'],
                  capture_sha256=[manifest['capture_sha256'] for manifest in manifests],
                  cuda_gemm_options=[manifest.get('cuda_gemm_options','unrecorded') for manifest in manifests],
                  scope='raw model outputs; shared valid prefix; no acceptance thresholds')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'requests'}, indent=2))


if __name__ == '__main__':
    main()
