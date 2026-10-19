#!/bin/bash
set -o pipefail
H="$HOME/hcx/streamfix/finite_o3"
mkdir -p "$H/builds" "$H/dumps" "$H/reports"
exec 9> "$HOME/w1012/build.lock"
flock 9
for arch in gfx1100 gfx1030; do
  for side in old new; do
    tree=sfbase; [ "$side" = new ] && tree=sf46
    echo "BUILD_START $side $arch $(git -C "$HOME/nh/$tree" rev-parse HEAD)"
    python3 "$H/build.py" "$HOME/nh/$tree" "$arch" "$H/builds/${side}_${arch}" --hipcc /opt/rocm/core-7.14/bin/hipcc || exit 1
    echo "BUILD_DONE $side $arch"
  done
done
flock -u 9
# Negative strict GPU control owns HIP1 first; do not overlap it.
while ! grep -q NEGATIVE_CONTROL_EXPECTED_FAILURE_PASS "$HOME/hcx/logs/streamfix/negative/stage2.out"; do sleep 5; done
run_arch() {
  local arch=$1 dev=$2 side
  for side in old new; do
    mkdir -p "$H/dumps/${side}_${arch}"
    echo "GPU_START $side $arch HIP=$dev"
    env -u ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES="$dev" LD_LIBRARY_PATH="$HOME/.local/rocm-rdna/lib" timeout -s KILL 1800 "$H/builds/${side}_${arch}" "$H/dumps/${side}_${arch}" > "$H/reports/${side}_${arch}.out" 2>&1 || return 1
    echo "GPU_DONE $side $arch"
  done
  python3 "$H/compare.py" "$H/dumps/old_${arch}" "$H/dumps/new_${arch}" "$H/builds/old_${arch}.json" "$H/builds/new_${arch}.json" --output "$H/reports/comparison_${arch}.json"
}
run_arch gfx1100 1 & w=$!
run_arch gfx1030 3 & v=$!
wait "$w"; wx=$?; wait "$v"; vx=$?
echo "O3_COMPARE_STATUS gfx1100=$wx gfx1030=$vx"
[ "$wx" = 0 ] && [ "$vx" = 0 ] && echo FINITE_O3_ALL_PASS
