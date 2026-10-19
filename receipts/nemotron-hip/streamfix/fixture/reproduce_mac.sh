#!/usr/bin/env bash
set -euo pipefail
: "${SOURCE_TREE:?Set SOURCE_TREE to the reviewed checkout}"
: "${MAC_WORK:?Set MAC_WORK to a new cache/log directory}"
ZIG=${ZIG:-zig}
cd "$SOURCE_TREE"
[ "$(git rev-parse HEAD)" = 1596156e21769324b675ef5e0caae0452f361660 ]
git diff --quiet
git diff --cached --quiet
[ ! -e "$MAC_WORK" ]
mkdir -p "$MAC_WORK"
export ZIG_LOCAL_CACHE_DIR="$MAC_WORK/local"
export ZIG_GLOBAL_CACHE_DIR="$MAC_WORK/global"
export CLANG_MODULE_CACHE_PATH="$MAC_WORK/clang-modules"
"$ZIG" build test test-kernel-contract test-golden fuzz-cortheon -Dcpu=apple_m1 --summary all > "$MAC_WORK/gate.log" 2>&1
"$ZIG" build --summary all > "$MAC_WORK/plain.log" 2>&1
