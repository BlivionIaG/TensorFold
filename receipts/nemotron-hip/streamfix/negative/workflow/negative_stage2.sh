#!/bin/bash
set -o pipefail
O="$HOME/hcx/logs/streamfix/negative"
"$HOME/hcx/zb.sh" sfnegative streamfix/negative/limits_qualified_build.log hip-kernel-contract-limits tf-hip-kernel-contract -Dhip-arch=gfx1100 -Doptimize=ReleaseFast --prefix out-gfx1100
x=$?; echo "QUALIFIED_BUILD_STATUS $x"; [ "$x" = 0 ] || exit "$x"
cd "$HOME/nh/sfnegative" || exit 1
bin=out-gfx1100/bin/tf-hip-kernel-contract
[ -x "$bin" ] || { echo MISSING_BINARY; exit 1; }
git -c gc.auto=0 rev-parse HEAD > "$O/source_commit.txt"
sha256sum "$bin" > "$O/binary_sha256.txt"
sha256sum zig/kernels/hip/quant/mlx_decoder.hpp zig/kernels/hip/tiles/stream.hpp > "$O/header_sha256.txt"
echo "GPU_NEGATIVE_START HIP_VISIBLE_DEVICES=1"
env -u ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=1 LD_LIBRARY_PATH="$HOME/.local/rocm-rdna/lib" timeout -s KILL 1800 "$bin" > "$O/strict_gfx1100.log" 2>&1
x=$?; echo "NEGATIVE_RUN_STATUS $x"
[ "$x" = 1 ] || exit 2
grep '^SUMMARY ' "$O/strict_gfx1100.log"
grep -q '0 known findings; 5 controls: 5 red as planted, 0 missed; 0 limit problems' "$O/strict_gfx1100.log" || exit 3
grep -q '^FAIL nonfinite hip affine_dot2_stream' "$O/strict_gfx1100.log" || exit 4
echo NEGATIVE_CONTROL_EXPECTED_FAILURE_PASS
