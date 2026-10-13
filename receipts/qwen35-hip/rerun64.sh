#!/bin/bash
# rerun64.sh BIN TAG: W7800 tp2, 64k then 128k cold prefill only; our PIDs only
cd ~/rcpt655/receipt; B=$1; tag=$2; O=rerun-$tag; mkdir -p $O; M=$HOME/models/Qwen3.6-35B-A3B-MLX-4bit-MTP
export LD_LIBRARY_PATH=$HOME/rcpt655/out-gfx1100/lib; port=19970; link=20970
env -u ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=0,1 $B serve $M --tp 2 --rank 1 --master 127.0.0.1 --master-port $link > $O/r1.log 2>&1 & P1=$!
env -u ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=0,1 $B serve $M --port $port --host 127.0.0.1 --no-update-check --tp 2 --rank 0 --master 127.0.0.1 --master-port $link --prompt-cache-gib 0 --parallel 1 --context 0 > $O/r0.log 2>&1 & P0=$!
for i in $(seq 1 150); do curl -s -m2 localhost:$port/health >/dev/null && break; sleep 2; done
for N in 65536 131072; do
  kill -0 $P1 2>/dev/null || { echo "rank 1 gone before $N"; break; }
  timeout 900 python3 ~/rcpt655/tools/bench_cache.py http://127.0.0.1:$port Qwen3.6-35B-A3B-MLX-4bit-MTP --prompt-file prompts/prefill_$N.json --expected-prompt-tokens $N --require-cold --tokens 16 --repeats 1 --streams 1 --output $O/prefill_$N.json >/dev/null 2>&1
  python3 -c "import json;d=json.load(open(\"$O/prefill_$N.json\"));print($N,\"qualified\",d[\"qualified\"],[(r[\"phase\"],r.get(\"prompt_tokens\"),r.get(\"ttft_s\")) for r in d[\"rows\"]])" 2>/dev/null || echo "$N no receipt"
  kill -0 $P1 2>/dev/null || { echo "rank 1 died at $N:"; grep -v "^\s*$" $O/r1.log | tail -15; break; }
done
for p in $P0 $P1; do kill -INT $p 2>/dev/null; done; sleep 15; for p in $P0 $P1; do kill -0 $p 2>/dev/null && kill -KILL $p; done; echo "stopped $P0 $P1"
