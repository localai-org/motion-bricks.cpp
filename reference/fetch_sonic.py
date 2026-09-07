#!/usr/bin/env python3
"""Fetch only the pinned SONIC deployment bundle and physical baseline assets.

Uses public official repositories, never loads pickle checkpoints, and writes
only below --output. Existing files must match their remote identities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from huggingface_hub import hf_hub_download, hf_hub_url, get_hf_file_metadata
import requests

SOURCE_REVISION = "a0732b642c0333077e127a2f56ab0014c196bca4"
MODEL_REVISION = "6733128a3d8a523b1418b06bca3cdf61c8b0987f"
MODEL_REPO = "nvidia/GEAR-SONIC"
SOURCE_REPO = "NVlabs/GR00T-WholeBodyControl"
MODEL_FILES = ("config.json", "LICENSE", "model_encoder.onnx",
               "model_decoder.onnx", "observation_config.yaml")
ASSET_ROOTS = (
    "gear_sonic_deploy/reference/example/walking_quip_360_R_002__A428",
    "gear_sonic/data/robot_model/model_data/g1",
    "gear_sonic_deploy/thirdparty/unitree_sdk2/lib/x86_64",
    "gear_sonic_deploy/thirdparty/unitree_sdk2/thirdparty/lib/x86_64",
)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.upstream_root.resolve()
    output = args.output.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != SOURCE_REVISION:
        raise ValueError(f"expected upstream {SOURCE_REVISION}, got {revision}")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for name in MODEL_FILES:
        print(f"model: {name}", flush=True)
        path = Path(hf_hub_download(MODEL_REPO, name, revision=MODEL_REVISION,
                                   local_dir=output / "policy"))
        identity = get_hf_file_metadata(hf_hub_url(MODEL_REPO, name, revision=MODEL_REVISION))
        if identity.commit_hash != MODEL_REVISION or path.stat().st_size != identity.size:
            raise ValueError(f"model revision/size mismatch: {name}")
        if len(identity.etag) == 64:
            actual_identity = digest(path)
        elif len(identity.etag) == 40:
            content = path.read_bytes()
            actual_identity = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        else:
            raise ValueError(f"unsupported model identity: {name}: {identity.etag}")
        if actual_identity != identity.etag:
            raise ValueError(f"model content identity mismatch: {name}")
        records.append({"path": str(path.relative_to(output)),
                        "bytes": path.stat().st_size, "sha256": digest(path),
                        "source_etag": identity.etag})
    entries = subprocess.check_output(
        ["git", "-C", str(source), "ls-tree", "-r",
         SOURCE_REVISION, "--", *ASSET_ROOTS], text=True).splitlines()
    if not entries:
        raise ValueError("no upstream robot/example assets found")
    for entry in entries:
        attributes, name = entry.split("\t", 1)
        mode = attributes.split()[0]
        # Read the pinned Git object, not potentially modified local content.
        content = subprocess.check_output(
            ["git", "-C", str(source), "show", f"{SOURCE_REVISION}:{name}"])
        path = output / "assets" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if mode == "120000":
            link = content.decode()
            if Path(link).is_absolute() or not (path.parent / link).resolve().is_relative_to(output):
                raise ValueError(f"unsafe upstream asset symlink: {name}")
            if not path.is_symlink() or path.readlink() != Path(link):
                temporary = path.with_name(path.name + ".partial")
                temporary.symlink_to(link)
                temporary.replace(path)
            records.append({"path": str(path.relative_to(output)), "type": "symlink",
                            "target": link, "bytes": len(content),
                            "sha256": hashlib.sha256(content).hexdigest()})
            continue
        if mode not in ("100644", "100755"):
            raise ValueError(f"unsupported asset mode: {name}: {mode}")
        if content.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
            fields = dict(line.split(" ", 1) for line in content.decode().splitlines())
            expected = fields["oid"].removeprefix("sha256:")
            size = int(fields["size"])
            if not (path.is_file() and path.stat().st_size == size
                    and digest(path) == expected):
                print(f"asset: {name} ({size} bytes)", flush=True)
                url = f"https://media.githubusercontent.com/media/{SOURCE_REPO}/{SOURCE_REVISION}/{name}"
                temporary = path.with_name(path.name + ".partial")
                with requests.get(url, stream=True, timeout=(30, 120)) as response:
                    response.raise_for_status()
                    with temporary.open("wb") as stream:
                        for chunk in response.iter_content(1024 * 1024):
                            stream.write(chunk)
                if temporary.stat().st_size != size or digest(temporary) != expected:
                    raise ValueError(f"Git LFS identity mismatch: {name}")
                temporary.replace(path)
        else:
            path.write_bytes(content)
        records.append({"path": str(path.relative_to(output)),
                        "bytes": path.stat().st_size, "sha256": digest(path)})
    manifest = {"schema": 1, "source_repository": SOURCE_REPO,
                "source_revision": SOURCE_REVISION, "model_repository": MODEL_REPO,
                "model_revision": MODEL_REVISION, "files": records}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified {len(records)} files; manifest: {output / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
