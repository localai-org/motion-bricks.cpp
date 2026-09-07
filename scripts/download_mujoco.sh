#!/usr/bin/env bash
# Official, pinned SDK for the native demo; the Docker reference stays 3.3.5.
set -euo pipefail
version=3.12.0
case "$(uname -s)/$(uname -m)" in
  Linux/x86_64) platform=linux-x86_64; sha=a9367911e6d5eaeade17c2197304687421c1fc932cdf7bcd4cb8cfaf0374dcb2 ;;
  Linux/aarch64|Linux/arm64) platform=linux-aarch64; sha=08fd5627a2ef7d5a42580c40e014ab2c1a644f082010c584ca361a3ed8cad838 ;;
  *) echo 'Use the official MuJoCo SDK for your platform and set MUJOCO_INCLUDE_DIR/MUJOCO_LIBRARY.' >&2; exit 1 ;;
esac
parent=${1:-generated/sonic}
destination=$parent/mujoco-$version
if [[ -e "$destination" ]]; then
  echo "Refusing to overwrite existing SDK: $destination" >&2
  exit 1
fi
mkdir -p -- "$parent"
sdk_temp=$(mktemp -d "$parent/.mujoco-download.XXXXXXXX")
trap 'rm -rf -- "$sdk_temp"' EXIT
archive=$sdk_temp/sdk.tar.gz
curl --fail --location --retry 2 \
  "https://github.com/google-deepmind/mujoco/releases/download/$version/mujoco-$version-$platform.tar.gz" \
  --output "$archive"
printf '%s  %s\n' "$sha" "$archive" | sha256sum --check --status
tar -xzf "$archive" -C "$sdk_temp" --no-same-owner
mv -- "$sdk_temp/mujoco-$version" "$destination"
printf 'Verified MuJoCo %s SDK: %s\n' "$version" "$destination"
