#!/usr/bin/env python3
"""Map named native neural boundaries to independent upstream module captures.

Unmapped nodes and layers are explicit. This report never implies complete layer
coverage just because the mapped subset passes.
"""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
from compare_native_taps import read_gguf

PREFIXES={'pose':'model._mmm_pose_model.mmm_net.',
          'root':'model._mmm_root_model.mmm_net.',
          'decoder':'model._vqvae_pose_model.'}


def valid_sequence_prefix(reference, component, name, actual, expected):
    """Map only known masked temporal axes; never infer an axis from shape alone."""
    duration=np.asarray(reference['output/pred_num_tokens']).reshape(-1)
    if duration.size!=1 or not 6<=duration[0]<=16:
        raise ValueError('Layer correspondence requires one valid SOMA duration')
    count=int(duration[0])
    if component=='root' and name.startswith('_root_token_transformer_model'):
        mask_key='layer/'+PREFIXES['root']+'_root_token_transformer_model/0/kwargs/src_key_padding_mask'
        mask=np.asarray(reference[mask_key])
        wanted=(np.arange(25)>=9+count)[None,:]
        axis=0;full=25;valid=9+count
    elif (component=='root' and name.startswith('_conv_output.')) or (component=='decoder' and name.startswith('decoder.')):
        module='_conv_output' if component=='root' else 'decoder'
        mask_key='layer/'+PREFIXES[component]+module+'/0/kwargs/token_mask'
        mask=np.asarray(reference[mask_key]);wanted=(np.arange(16)<count)[None,:]
        suffix=name[len(module)+1:]
        # Linear condition projections are time-major; convolutional blocks and
        # the transposed external-condition activations are channels-major.
        if suffix.startswith('target_cond_blocks.'):
            axis=0;full=64;valid=count*4
        elif suffix.startswith('external_cond_blocks.'):
            block=int(suffix.split('.')[1]);axis=0 if block%2==0 else 1
            full=16*(2**(block//2));valid=count*(2**(block//2))
        elif suffix.startswith('model.'):
            parts=suffix.split('.');block=int(parts[1]);axis=1
            scale=1 if block<=1 else 2 if block==2 else 4
            if block in (2,3) and len(parts)>2 and parts[2]=='0':scale//=2
            full=16*scale;valid=count*scale
        else:return expected,None
    else:return expected,None
    if mask.shape!=wanted.shape or not np.array_equal(mask,wanted):
        raise ValueError(f'Unexpected upstream temporal mask: {mask_key}')
    if actual.ndim!=2 or expected.ndim!=2 or actual.shape[axis]!=valid or expected.shape[axis]!=full:
        return expected,None
    if actual.shape[1-axis]!=expected.shape[1-axis]:return expected,None
    selection=[slice(None),slice(None)];selection[axis]=slice(0,valid)
    return expected[tuple(selection)],dict(kind='masked_temporal_prefix',axis=axis,
        valid_length=valid,padded_length=full,mask_reference=mask_key)


def compare(reference,native_path):
    metadata,native=read_gguf(native_path)
    rows=[];unmapped=[];used=set()
    sequential_component='root';previous_index=-1
    for key,actual in native.items():
        component,index,name=metadata[key+'.name'].split('/',2)
        if component=='soma':
            index=int(index)
            if index<previous_index and name=='_proj_local_pose':
                sequential_component='pose'
            if name.startswith('decoder.'):
                sequential_component='decoder'
            previous_index=index
            component=sequential_component
        if component not in PREFIXES or ' ' in name or not name or name.startswith('node_'):
            continue
        refkey='layer/'+PREFIXES[component]+name+'/0/output'
        if component=='pose' and name=='_positioned_input':
            refkey='layer/'+PREFIXES[component]+'_transformer_model/0/input/0'
        if name.endswith('self_attn.out_proj'):
            refkey='layer/'+PREFIXES[component]+name.removesuffix('.out_proj')+'/0/output/0'
        if refkey not in reference:
            if metadata[key+'.operation']=='ADD':unmapped.append(dict(tap=key,name=name))
            continue
        # Weight-derived semantic labels mark only module results, not inherited view labels.
        if metadata[key+'.operation'] not in ('ADD','RELU','UNARY','LEAKY_RELU','VIEW','GET_ROWS','UPSCALE'):continue
        expected=reference[refkey]
        a=np.squeeze(np.asarray(actual));b=np.squeeze(expected)
        if '.external_cond_blocks.' in name and name.endswith(('.1','.3')):
            a=a.T  # upstream activation follows its frames-to-channels transpose
        if component=='pose' and name=='_pose_token_emb':
            a=a.reshape(b.shape)  # native gathers flattened token/head indices
        correspondence=None
        if a.shape!=b.shape:
            b,correspondence=valid_sequence_prefix(reference,component,name,a,b)
        if a.shape!=b.shape:
            rows.append(dict(tap=key,reference=refkey,status='shape_mismatch',native_shape=list(a.shape),reference_shape=list(b.shape)))
            continue
        if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError(f'Nonfinite layer {refkey}')
        error=np.abs(a.astype(np.float64)-b.astype(np.float64))
        rows.append(dict(tap=key,reference=refkey,status='compared',shape=list(a.shape),
                         max_absolute=float(error.max()),rms=float(np.sqrt(np.mean(error**2)))))
        if correspondence:rows[-1]['correspondence']=correspondence
        used.add(refkey)
        # In evaluation mode these upstream modules are exact aliases. Verify
        # that independently captured outputs agree before recording coverage.
        aliases=[]
        if name.endswith('.norm2') and 'transformer' in name:
            aliases.append(refkey.replace('.norm2/0/output','/0/output'))
        if name.endswith('self_attn.out_proj'):
            aliases.append(refkey.replace('.self_attn/0/output/0','.dropout1/0/output'))
        if name.endswith('.linear2') and 'transformer' in name:
            aliases.append(refkey.replace('.linear2/0/output','.dropout2/0/output'))
        if name.endswith('.forward_projection'):
            aliases.append(refkey.replace('.forward_projection/0/output','/0/output'))
        for last in ('_transformer_model.layers.15.norm2','_shared_transformer_model.layers.5.norm2',
                     '_root_token_transformer_model.layers.1.norm2'):
            if name==last:
                aliases.append('layer/'+PREFIXES[component]+last.split('.layers.')[0]+'/0/output')
        if name=='_proj_input.1':aliases.append(refkey.replace('.1/0/output','/0/output'))
        if name.endswith('.0.model.3'):
            aliases.append(refkey.replace('.model.3/0/output','/0/output'))
        if name=='decoder.model.6':aliases.append('layer/model._vqvae_pose_model.decoder/0/output')
        if name=='_conv_output.model.6':aliases.append('layer/model._mmm_root_model.mmm_net._conv_output/0/output')
        for alias in aliases:
            if alias in reference:
                alias_expected=expected
                # Decoder wrappers apply the final temporal mask after conv6.
                # Prove that exact operation across the entire padded tensor.
                if name in ('decoder.model.6','_conv_output.model.6'):
                    module=name.removesuffix('.model.6')
                    mask_key='layer/'+PREFIXES[component]+module+'/0/kwargs/token_mask'
                    mask=np.repeat(reference[mask_key],4,axis=1)
                    alias_expected=expected*mask[:,None,:]
                if not np.array_equal(reference[alias],alias_expected):raise ValueError(f'Upstream alias changed: {alias}')
                used.add(alias)
                rows.append(dict(rows[-1],reference=alias,alias_of=refkey))
    candidates=[key for key in reference.files if key.startswith('layer/') and '/output' in key
                and any(key.startswith('layer/'+prefix) for prefix in PREFIXES.values())]
    return dict(format='motionbricks-soma-layer-comparison-v1',layers=rows,
                mapped_layers=len(used.intersection(candidates)),
                mapped_input_boundaries=len(used.difference(candidates)),unmapped_native=unmapped,
                unmapped_reference=[key for key in candidates if key not in used])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--native',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--require-complete',action='store_true')
    parser.add_argument('--max-absolute',type=float,help='Explicit absolute error ceiling for every mapped layer')
    args=parser.parse_args()
    manifest=json.loads((args.capture/'manifest.json').read_text())
    with (args.capture/'tensors.npz').open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest()
    if digest!=manifest['capture_sha256']:raise ValueError('Reference capture checksum mismatch')
    native_metadata,_=read_gguf(args.native)
    if native_metadata.get('trace.capture_sha256')!=digest:
        raise ValueError('Native trace was not produced from this reference capture fixture')
    with np.load(args.capture/'tensors.npz') as reference:report=compare(reference,args.native)
    report['capture_sha256']=digest
    report['source_revision']=manifest['reference']['source_revision']
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('Mapped layers:',report['mapped_layers'],'unmapped native:',len(report['unmapped_native']),
          'unmapped reference:',len(report['unmapped_reference']))
    for row in report['layers']:
        if row['status']!='compared' or row['max_absolute']>1e-3:print(row)
    if args.require_complete and (report['unmapped_reference'] or any(row['status']!='compared' for row in report['layers'])):
        raise SystemExit('Incomplete layer correspondence')
    if args.max_absolute is not None:
        if not np.isfinite(args.max_absolute) or args.max_absolute<0:raise ValueError('Invalid error ceiling')
        if any(row['status']!='compared' or row['max_absolute']>args.max_absolute for row in report['layers']):
            raise SystemExit('Layer error ceiling exceeded')


if __name__=='__main__':main()
