#!/usr/bin/env python3
"""Generate a logging-only upstream source variant, recording its exact identity.

The readonly source is always read from the pinned commit. The only inserted
code is sonic_trace.inc after CreatePolicyCommand and its JSON include. Writes
only the generated build's source copy; never edits the original checkout.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from fetch_sonic import SOURCE_REVISION


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    args = p.parse_args()
    relative = 'gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp'
    original = subprocess.check_output(['git','-C',str(args.upstream_root),'show',f'{SOURCE_REVISION}:{relative}']).decode()
    marker = '          auto motor_command_end_time = std::chrono::steady_clock::now();\n'
    if original.count(marker) != 1:
        raise ValueError('upstream trace insertion point changed')
    snippet = Path(__file__).with_name('sonic_trace.inc').read_text()
    generated = '#include <nlohmann/json.hpp>\n' + original.replace(marker,marker+snippet)
    target = args.output_root/'source'/relative
    if target.resolve().is_relative_to(args.upstream_root.resolve()):
        raise ValueError('refusing to modify original upstream source')
    if target.read_text() != generated:
        target.write_text(generated)
    sha = lambda value: hashlib.sha256(value.encode()).hexdigest()
    manifest = dict(schema=1,source_revision=SOURCE_REVISION,original_sha256=sha(original),
                    snippet_sha256=sha(snippet),instrumented_sha256=sha(generated),
                    mutation='JSON include and post-command observation-only logging; SONIC_TRACE_FILE controls output')
    (args.output_root/'instrumentation.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__':
    main()
