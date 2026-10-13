#!/bin/bash
# run.sh NAME DEVS ARCH: one receipt config of Qwen3.6-35B-A3B 4-bit MTP; only our own PIDs are measured or stopped
set -u
name=$1; devs=$2; arch=$3
R=$HOME/rcpt655; O=$R/receipt/$name-fix; mkdir -p $O; cd $R/receipt
M=$HOME/models/Qwen3.6-35B-A3B-MLX-4bit-MTP; B=$HOME/fixtp/out-$arch/native-hip/bin/tensorfold-native
export LD_LIBRARY_PATH=$R/out-$arch/lib; PY=python3; T=$R/tools
n=$(echo $devs | awk -F, '{print NF}'); port=$((19100 + RANDOM % 800)); link=$((port + 1000))
start() { # flags...: rank 0 serves, followers join; PIDS gets every rank's pid
  PIDS=""; fl=""
  if [ $n -gt 1 ]; then for r in $(seq 1 $((n - 1))); do
    env -u ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=$devs $B serve $M --tp $n --rank $r --master 127.0.0.1 --master-port $link > $O/$1.r$r.log 2>&1 & PIDS="$PIDS $!"; done
    fl="--tp $n --rank 0 --master 127.0.0.1 --master-port $link"; fi
  local tag=$1; shift
  env -u ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=$devs $B serve $M --port $port --host 127.0.0.1 --no-update-check $fl "$@" > $O/$tag.log 2>&1 & P0=$!; PIDS="$P0$PIDS"
  for i in $(seq 1 300); do curl -s -m 2 localhost:$port/health >/dev/null && break; kill -0 $P0 2>/dev/null || { echo "$tag: server exited"; tail -3 $O/$tag.log; return 1; }; sleep 2; done
  sh ./memwatch.sh $O/$tag.mem $PIDS & MW=$!
  grep -hE "HIP gfx|serving" $O/$tag.log | cut -c1-260
}
stop() { for p in $PIDS; do kill -INT $p 2>/dev/null; done
  for i in $(seq 1 40); do a=0; for p in $PIDS; do kill -0 $p 2>/dev/null && a=1; done; [ $a = 0 ] && break; sleep 1; done
  for p in $PIDS; do kill -0 $p 2>/dev/null && { echo "pid $p still up: KILL"; kill -KILL $p; }; done
  wait $MW 2>/dev/null; cat $O/$1.mem.json; }
model_id() { curl -s localhost:$port/v1/models | $PY -c "import json,sys;print(json.load(sys.stdin)['data'][0]['id'])"; }
echo "== $name devs=$devs arch=$arch head=$(git -C $HOME/fixtp rev-parse --short HEAD)+window-fix $(date -Is)"
echo "-- cold prefill"
if start prefill --prompt-cache-gib 0 --parallel 1 --context 0; then
  id=$(model_id)
  for N in 2048 8192 32768 65536 131072 261888; do
    [ -f prompts/prefill_$N.json ] || continue
    timeout 1800 $PY $T/bench_cache.py http://127.0.0.1:$port $id --prompt-file prompts/prefill_$N.json --expected-prompt-tokens $N --require-cold --tokens 16 --repeats 1 --streams 1 --output $O/prefill_$N.json > /dev/null 2>$O/prefill_$N.err
    $PY -c "import json;d=json.load(open('$O/prefill_$N.json'));print($N,'qualified',d['qualified'],[(r['phase'],r.get('prompt_tokens'),r.get('cached_tokens'),r.get('ttft_s')) for r in d['rows']])" 2>/dev/null || { echo "$N: no receipt"; tail -2 $O/prefill_$N.err; }
  done
  stop prefill
fi
echo "-- decode and concurrency"
if start decode --prompt-cache-gib 0 --parallel 4; then
  id=$(model_id)
  $PY $T/bench_openai.py http://127.0.0.1:$port $id --tokens 256 --reps 3 --temperatures 0 --label $name --output $O/decode.json > $O/decode.out 2>&1; tail -4 $O/decode.out
  $PY $T/bench_concurrent.py http://127.0.0.1:$port $id --tokens 256 --reps 3 --temperatures 0 --levels 1,4 --alone --serial --label $name --output $O/concurrent.json > $O/concurrent.out 2>&1; tail -6 $O/concurrent.out
  stop decode
fi
echo "== done $name $(date -Is)"
