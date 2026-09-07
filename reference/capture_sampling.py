#!/usr/bin/env python3
"""Capture NVIDIA's actual Gumbel sampler with fixed uniforms, including edges."""
import argparse
import hashlib
import importlib.util
import json
import struct
import subprocess
from pathlib import Path
from unittest.mock import patch
import torch
from generate_fixtures import UPSTREAM_REVISION

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    relative='motionbricks/motionbricks/motion_backbone/models/sampling.py'
    source=args.upstream_root/relative
    revision=subprocess.check_output(['git','-C',str(args.upstream_root),'rev-parse','HEAD'],text=True).strip()
    if revision!=UPSTREAM_REVISION: raise ValueError('unexpected upstream revision')
    subprocess.run(['git','-C',str(args.upstream_root),'diff','--exit-code','HEAD','--',relative],check=True)
    spec=importlib.util.spec_from_file_location('upstream_sampling',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    torch.manual_seed(12345)
    logits=torch.randn(512,10,dtype=torch.float32)*2
    uniforms=torch.rand_like(logits)
    uniforms[0]=0;uniforms[1]=1;uniforms[2]=0.5;logits[2]=0
    uniforms[3,0]=1e-30;uniforms[3,1]=1e-20
    # Use the upstream function itself, only replacing its random draw.
    def supplied(tensor,*_args,**_kwargs):
        return tensor.copy_(uniforms)
    with patch.object(torch.Tensor,'uniform_',supplied):
        noise=module.gumbel_noise(logits)
        tokens=module.gumbel_sample(logits,temperature=1.0)
    scores=logits+noise
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('wb') as stream:
        stream.write(struct.pack('<8sII',b'MBGSAMP1',512,10))
        for tensor in (logits,uniforms,noise,scores):stream.write(tensor.numpy().astype('<f4').tobytes())
        stream.write(tokens.numpy().astype('<i4').tobytes())
    manifest={'upstream_revision':revision,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'torch':torch.__version__,'temperature':1,'rows':512,'fixture_sha256':hashlib.sha256(args.output.read_bytes()).hexdigest()}
    args.output.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest))

if __name__=='__main__':main()
