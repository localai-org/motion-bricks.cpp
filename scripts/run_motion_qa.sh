#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
kimodo_dir="${1:-$repo_root/../kimodo.cpp/demo-output}"
artifact_dir="${2:-$repo_root/generated/motion-qa}"

if [[ ! -d "$kimodo_dir" ]]; then
    echo "Kimodo directory does not exist: $kimodo_dir" >&2
    exit 2
fi
# Resolve caller-relative paths before changing into demo/.
kimodo_dir="$(cd "$kimodo_dir" && pwd)"
if [[ "$artifact_dir" != /* ]]; then
    artifact_dir="$PWD/$artifact_dir"
fi

chrome="${MOTIONBRICKS_CHROME:-$(command -v chromium || command -v chromium-browser || command -v google-chrome || true)}"
if [[ -z "$chrome" ]]; then
    echo "Chromium/Chrome was not found; set MOTIONBRICKS_CHROME" >&2
    exit 2
fi

cd "$repo_root/demo"
CGO_ENABLED=0 \
MOTIONBRICKS_LIB="${MOTIONBRICKS_LIB:-$repo_root/build/debug/libmotionbricks.so}" \
MOTIONBRICKS_MODEL="${MOTIONBRICKS_MODEL:-$repo_root/generated/g1-f32}" \
MOTIONBRICKS_STYLES="${MOTIONBRICKS_STYLES:-$repo_root/generated/styles}" \
MOTIONBRICKS_KIMODO_DIR="$kimodo_dir" \
MOTIONBRICKS_MOTION_QA_DIR="$artifact_dir" \
MOTIONBRICKS_MOTION_QA_DEVICE="${MOTIONBRICKS_MOTION_QA_DEVICE:-cpu}" \
MOTIONBRICKS_CHROME="$chrome" \
go test -run '^TestKimodoMotionQA$' -v -count=1
