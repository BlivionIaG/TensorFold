#!/usr/bin/env bash
# Install the HIP headers and the ROCm device libraries a macOS host needs to build TensorFold's gfx900 (Vega) kernels with Homebrew LLVM.
#   tools/vega-macos/fetch-rocm.sh [--wheel PATH]      -> $TF_VEGA_ROOT (default ~/.cache/tensorfold-vega/rocm-<version>)
# Layout (what zig/build/hip.zig expects of a ROCm root, so -Dhipcc=<root>/bin/hipcc is all the build needs):
#   <root>/bin/hipcc                      symlink to tools/vega-macos/hipcc (the wrapper); <root>/bin/.hipVersion lets clang find the HIP wrapper header
#   <root>/include/hip/**                 HIP 7.x headers (hip_runtime.h, amd_detail/...)
#   <root>/lib/llvm/amdgcn/bitcode/*.bc   ocml, ockl, oclc_*, hip, opencl bitcode
# Needs: python3, curl, and `brew install llvm lld`. No sudo. The wheel (416 MB) is verified against pins.env and kept only when --keep is given.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=pins.env
. "$HERE/pins.env"
ROOT=${TF_VEGA_ROOT:-$HOME/.cache/tensorfold-vega/rocm-$ROCM_VERSION}
WHEEL=""; KEEP=0
while [ $# -gt 0 ]; do
	case "$1" in
	--wheel) WHEEL=$2; shift ;;
	--keep) KEEP=1 ;;
	*) echo "fetch-rocm: unknown argument $1" >&2; exit 2 ;;
	esac
	shift
done
if [ -x "$ROOT/bin/hipcc" ] && [ -f "$ROOT/include/hip/hip_runtime.h" ] && [ -f "$ROOT/lib/llvm/amdgcn/bitcode/ocml.bc" ] && [ -f "$ROOT/.pinned" ] && [ "$(cat "$ROOT/.pinned")" = "$ROCM_WHEEL_SHA256" ]; then
	echo "$ROOT"; exit 0
fi
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
if [ -z "$WHEEL" ]; then
	WHEEL=$TMP/$ROCM_WHEEL
	curl -fL --retry 3 -o "$WHEEL" "${ROCM_INDEX}${ROCM_WHEEL}" >&2
fi
GOT=$(shasum -a 256 "$WHEEL" | cut -d' ' -f1)
[ "$GOT" = "$ROCM_WHEEL_SHA256" ] || { echo "fetch-rocm: $WHEEL sha256 $GOT != pinned $ROCM_WHEEL_SHA256" >&2; exit 1; }
rm -rf "$ROOT"; mkdir -p "$ROOT/bin"
python3 - "$WHEEL" "$ROOT" <<'PY'
import os, sys, zipfile
wheel, root = sys.argv[1:3]
z = zipfile.ZipFile(wheel)
for n in z.namelist():
    if n.endswith('/'):
        continue
    if n.startswith('_rocm_sdk_core/include/hip/'):
        out = os.path.join(root, 'include/hip', n[len('_rocm_sdk_core/include/hip/'):])
    elif n.startswith('_rocm_sdk_core/lib/llvm/amdgcn/bitcode/'):
        out = os.path.join(root, 'lib/llvm/amdgcn/bitcode', n.rsplit('/', 1)[1])
    else:
        continue
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'wb') as f:
        f.write(z.read(n))
PY
# clang only adds its __clang_hip_runtime_wrapper.h (the device math: fmaf, expf, ... next to the host libm) once it reads a HIP version here
python3 - "$ROOT" <<'PY'
import re, sys
root = sys.argv[1]
h = open(root + '/include/hip/hip_version.h').read()
v = {k: re.search(r'#define HIP_VERSION_%s\s+(\d+)' % k, h).group(1) for k in ('MAJOR', 'MINOR', 'PATCH')}
open(root + '/bin/.hipVersion', 'w').write(''.join('HIP_VERSION_%s=%s\n' % kv for kv in v.items()) + 'HIP_VERSION_GITHASH=0\n')
PY
ln -s "$HERE/hipcc" "$ROOT/bin/hipcc"
printf '%s' "$ROCM_WHEEL_SHA256" > "$ROOT/.pinned"
[ "$KEEP" = 1 ] && cp "$WHEEL" "$(dirname "$ROOT")/" || true
echo "$ROOT"
