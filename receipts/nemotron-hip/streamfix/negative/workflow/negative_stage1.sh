#!/bin/bash
set -o pipefail
mkdir -p "$HOME/hcx/logs/streamfix/negative"
"$HOME/hcx/zb.sh" sfnegative streamfix/negative/objects.log hip-kernel-contract-objects -Dhip-arch=gfx1100 -Doptimize=ReleaseFast --prefix out-gfx1100
x=$?; echo "OBJECTS_STATUS $x"; [ "$x" = 0 ] || exit "$x"
"$HOME/hcx/zb.sh" sfnegative streamfix/negative/limits_stale.log hip-kernel-contract-limits -Dhip-arch=gfx1100 -Doptimize=ReleaseFast --prefix out-gfx1100
x=$?; echo "LIMITS_BEFORE_HYDRATION_STATUS $x"
