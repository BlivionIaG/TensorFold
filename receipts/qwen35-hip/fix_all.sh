#!/bin/bash
# fix_all.sh: the window fix: tp checks on both GPU types, then the three tp receipt configs again, one at a time
cd ~/rcpt655/receipt; F=$HOME/fixtp; M=$HOME/models/Qwen3.6-35B-A3B-MLX-4bit-MTP
tp() { devs=$1; arch=$2; port=$3; n=$(echo $devs | awk -F, "{print NF}"); export LD_LIBRARY_PATH=$F/out-$arch/lib; pids=""
  for r in $(seq 1 $((n-1))); do env -u ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=$devs timeout -s KILL 3000 $F/out-$arch/bin/tf-qwen35-test check $M --tp $n --rank $r --port $port > /dev/null 2>&1 & pids="$pids $!"; done
  env -u ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=$devs timeout -s KILL 3000 $F/out-$arch/bin/tf-qwen35-test check $M --tp $n --rank 0 --port $port 2>&1 | grep -E "^check |^ *FAIL|error" | sed "s/^/tp$n $devs: /"; wait $pids; }
echo "== checks"; tp 0,1 gfx1100 29991; tp 2,3 gfx1030 29992; tp 2,3,4,5 gfx1030 29993
./run_fix.sh w7800-tp2 0,1 gfx1100
./run_fix.sh v620-tp2 2,3 gfx1030
./run_fix.sh v620-tp4 2,3,4,5 gfx1030
echo FIX-ALL-DONE
