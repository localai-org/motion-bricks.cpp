#!/usr/bin/env bash
# Run inside Dockerfile.sonic. The original upstream checkout stays read-only.
set -euo pipefail
if [[ $# != 2 ]]; then
    echo "usage: bash reference/build_sonic.sh UPSTREAM_ROOT SONIC_OUTPUT" >&2
    exit 2
fi
sonic_source=$(realpath "$1")
sonic_output=$(realpath "$2")
sonic_revision=a0732b642c0333077e127a2f56ab0014c196bca4
test "$(git -C "$sonic_source" rev-parse HEAD)" = "$sonic_revision"
mkdir -p "$sonic_output/source"
if [[ ! -d "$sonic_output/source/.archive-complete" ]]; then
    git -C "$sonic_source" -c filter.lfs.process= -c filter.lfs.smudge= \
        -c filter.lfs.required=false archive "$sonic_revision" gear_sonic_deploy |
        tar -x -C "$sonic_output/source"
    mkdir "$sonic_output/source/.archive-complete"
fi
cp -a "$sonic_output/assets/gear_sonic_deploy/thirdparty/." \
    "$sonic_output/source/gear_sonic_deploy/thirdparty/"
python "$(dirname "$0")/instrument_sonic.py" --upstream-root "$sonic_source" --output-root "$sonic_output"
cmake -S "$sonic_output/source/gear_sonic_deploy" -B "$sonic_output/build" -G Ninja \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_BUILD_TYPE=Release \
    '-DCMAKE_CXX_FLAGS_RELEASE=-O1 -g1 -DNDEBUG -fsanitize=address,undefined -fno-omit-frame-pointer -fno-fast-math' \
    '-DTensorRT_FIND_COMPONENTS=nvinfer;nvinfer_plugin;nvonnxparser' \
    -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=/usr/src/googletest
cmake --build "$sonic_output/build" --target g1_deploy_onnx_ref --parallel "${SONIC_BUILD_JOBS:-1}"
