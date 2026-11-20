#!/usr/bin/env python3
"""Download the pinned official SOMA weights and ONNX references, verifying bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request


def verify(path, entry):
    if not path.is_file() or path.stat().st_size != entry['bytes']:
        return False
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() == entry['sha256']


def fetch(url, path, entry):
    if verify(path, entry):
        print(f'Verified {path}', flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.partial')
    try:
        print(f'Downloading {path}', flush=True)
        with urllib.request.urlopen(url, timeout=120) as source, partial.open('wb') as target:
            shutil.copyfileobj(source, target)
        if not verify(partial, entry):
            raise ValueError(f'Checksum or size mismatch: {path}')
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--onnx', action='store_true', help='Also fetch both official ONNX precisions')
    args = parser.parse_args()
    manifest = json.loads(Path(__file__).with_name('soma-reference.json').read_text())
    base = f"https://huggingface.co/{manifest['checkpoint_repository']}/resolve/{manifest['checkpoint_revision']}"
    for entry in manifest['checkpoints']:
        fetch(f"{base}/{entry['path']}", args.output / entry['path'], entry)
    for entry in manifest['source_assets']:
        if entry['path'].endswith('.onnx') and not args.onnx:
            continue
        # The rig NPZ, as well as the ONNX models, is stored with Git LFS.
        host = 'media.githubusercontent.com/media'
        url = f"https://{host}/NVlabs/motionbricks/{manifest['source_revision']}/{entry['path']}"
        fetch(url, args.output / entry['path'], entry)


if __name__ == '__main__':
    main()
