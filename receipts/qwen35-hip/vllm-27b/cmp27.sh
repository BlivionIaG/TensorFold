#!/bin/bash
# cmp27.sh: Qwen3.8-27B 4-bit on 4 x V620 (HIP 2-5): TensorFold tp 4 (--no-drafts) against vLLM 0.31 RDNA (recipe
# 27b-awq, prefix caching off, MTP 0); the same bench tools and token arrays; our server PIDs only are measured
set -u
D=$HOME/cmp27; mkdir -p $D; cd $D; T=$HOME/rcpt655/tools; N="2048 8192 32768 65536 131072 261888"
TFM=$HOME/models/Qwen3.8-27B-MLX-Serve-4bit; TFB=$HOME/lr-final/out-gfx1030/native-hip/bin/tensorfold-native
VM=$HOME/.cache/huggingface/hub/models--cyankiwi--Qwen3.8-27B-AWQ-INT4/snapshots/63768c10df38c0395e12ef49edac1bd539eaeeea
VT=$HOME/vllm-rdna-0.31.0-cpu; VCACHE=$VT/cache/vllm
[ -f prompts/prefill_2048.json ] || python3 $HOME/rcpt655/receipt/mkprompts.py $TFM $HOME/rcpt655 prompts $N
bench() { # tag port: cold prefill at every size, then decode and concurrency
  local tag=$1 port=$2 id; id=$(curl -s localhost:$port/v1/models | python3 -c "import json,sys;print(json.load(sys.stdin)['data'][0]['id'])")
  for n in $N; do
    timeout 2400 python3 $T/bench_cache.py http://127.0.0.1:$port $id --prompt-file prompts/prefill_$n.json --expected-prompt-tokens $n --require-cold --tokens 16 --repeats 1 --streams 1 --output $tag/prefill_$n.json > /dev/null 2> $tag/prefill_$n.err
    python3 -c "import json;d=json.load(open('$tag/prefill_$n.json'));print('$tag',$n,'qualified',d['qualified'],[(r['phase'],r.get('prompt_tokens'),r.get('cached_tokens'),r.get('ttft_s')) for r in d['rows']])" 2>/dev/null || echo "$tag $n: no receipt ($(tail -1 $tag/prefill_$n.err | cut -c1-120))"
  done
  python3 $T/bench_openai.py http://127.0.0.1:$port $id --tokens 256 --reps 3 --temperatures 0 --label $tag --output $tag/decode.json > $tag/decode.out 2>&1; tail -2 $tag/decode.out
  python3 $T/bench_concurrent.py http://127.0.0.1:$port $id --tokens 256 --reps 3 --temperatures 0 --levels 1,4 --alone --serial --label $tag --output $tag/concurrent.json > $tag/concurrent.out 2>&1; tail -4 $tag/concurrent.out
}
stop() { for s in INT KILL; do for p in $*; do kill -$s $p 2>/dev/null; done; sleep 12; done; for p in $*; do kill -0 $p 2>/dev/null && echo "pid $p still up"; done; }
up() { for i in $(seq 1 600); do curl -s -m2 localhost:$1/health >/dev/null 2>&1 && return 0; sleep 2; done; return 1; }

echo "== tensorfold tp4 --no-drafts $(date -Is)"; mkdir -p tf; export LD_LIBRARY_PATH=$HOME/rcpt655/out-gfx1100/lib; P=""
for r in 1 2 3; do env -u ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=2,3,4,5 $TFB serve $TFM --tp 4 --rank $r --master 127.0.0.1 --master-port 20410 > tf/r$r.log 2>&1 & P="$P $!"; done
env -u ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=2,3,4,5 $TFB serve $TFM --port 19410 --host 127.0.0.1 --no-update-check --tp 4 --rank 0 --master 127.0.0.1 --master-port 20410 --no-drafts --prompt-cache-gib 0 --parallel 4 --context 0 > tf/r0.log 2>&1 & P="$! $P"
if up 19410; then grep -h "HIP gfx\|serving" tf/r0.log | cut -c1-230; bash $HOME/rcpt655/receipt/memwatch.sh tf/mem $P & MW=$!; bench tf 19410; fi
stop $P; wait $MW 2>/dev/null; cat tf/mem.json 2>/dev/null

echo "== vllm 0.31 27b-awq prefix off MTP 0 $(date -Is)"; mkdir -p vllm; unset LD_LIBRARY_PATH
(cd $VT && setsid bash tools/rdna/serve_rdna.sh RECIPE=27b-awq MODEL=$VM VENV=$HOME/Apps/vllm/venv-7.14.0_0.31.0 VLLM_TREE=$VT HIP_VISIBLE_DEVICES=2,3,4,5 ROCR_VISIBLE_DEVICES= PORT=18310 ENABLE_PREFIX_CACHING=0 MTP=0 > $D/vllm/serve.log 2>&1 < /dev/null &)
if up 18310; then
  VP=""; for p in $(pgrep -f "entrypoints.cli.main serve|VLLM::Worker|VLLM::EngineCore"); do tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -q "^VLLM_CACHE_ROOT=$VCACHE$" && VP="$VP $p"; done
  echo "vllm pids:$VP"; grep -m3 -E "max_model_len|KV cache|GPU KV|Maximum concurrency" vllm/serve.log | cut -c1-200
  bash $HOME/rcpt655/receipt/memwatch.sh vllm/mem $VP & MW=$!; bench vllm 18310
  stop $VP; wait $MW 2>/dev/null; cat vllm/mem.json
else echo "vllm not up"; tail -20 vllm/serve.log | cut -c1-200; fi
echo CMP-DONE $(date -Is)
