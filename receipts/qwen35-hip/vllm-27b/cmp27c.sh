#!/bin/bash
# cmp27c.sh: the same 27B pair with the prompt caches on: per size, one cold request, two repeats and two concurrent
# copies (bench_cache --require-reuse); TensorFold --no-drafts default prompt cache, vLLM recipe 27b-awq MTP 0
set -u
D=$HOME/cmp27; cd $D; T=$HOME/rcpt655/tools; N="2048 8192 32768 65536 131072 261888"
TFM=$HOME/models/Qwen3.8-27B-MLX-Serve-4bit; TFB=$HOME/lr-final/out-gfx1030/native-hip/bin/tensorfold-native
VM=$HOME/.cache/huggingface/hub/models--cyankiwi--Qwen3.8-27B-AWQ-INT4/snapshots/63768c10df38c0395e12ef49edac1bd539eaeeea; VT=$HOME/vllm-rdna-0.31.0-cpu
kids() { for c in $(pgrep -P $1); do echo $c; kids $c; done; }
up() { for i in $(seq 1 600); do curl -s -m2 localhost:$1/health >/dev/null 2>&1 && return 0; sleep 2; done; return 1; }
stop() { for s in INT KILL; do for p in $*; do kill -$s $p 2>/dev/null; done; sleep 12; done; for p in $*; do kill -0 $p 2>/dev/null && echo "pid $p still up"; done; }
bench() { local tag=$1 port=$2 id; id=$(curl -s localhost:$port/v1/models | python3 -c "import json,sys;print(json.load(sys.stdin)['data'][0]['id'])")
  for n in $N; do
    python3 $T/bench_cache.py http://127.0.0.1:$port $id --prompt-file prompts/prefill_$n.json --expected-prompt-tokens $n --require-reuse --tokens 16 --repeats 2 --streams 2 --output $tag/cache_$n.json > /dev/null 2> $tag/cache_$n.err
    python3 -c "import json;d=json.load(open('$tag/cache_$n.json'));print('$tag',$n,'qualified',d['qualified'],[(r['phase'],r.get('cached_tokens'),round(r['ttft_s'],2) if r.get('ttft_s') else None) for r in d['rows']])" 2>/dev/null || echo "$tag $n: no receipt ($(tail -1 $tag/cache_$n.err | cut -c1-120))"
  done; }
echo "== tensorfold tp4 --no-drafts, prompt cache on $(date -Is)"; mkdir -p tfc; export LD_LIBRARY_PATH=$HOME/rcpt655/out-gfx1100/lib; P=""
for r in 1 2 3; do env -u ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=2,3,4,5 $TFB serve $TFM --tp 4 --rank $r --master 127.0.0.1 --master-port 20420 > tfc/r$r.log 2>&1 & P="$P $!"; done
env -u ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=2,3,4,5 $TFB serve $TFM --port 19420 --host 127.0.0.1 --no-update-check --tp 4 --rank 0 --master 127.0.0.1 --master-port 20420 --no-drafts --parallel 2 --context 0 > tfc/r0.log 2>&1 & P="$! $P"
if up 19420; then grep -h "HIP gfx" tfc/r0.log | cut -c1-260; bash $HOME/rcpt655/receipt/memwatch.sh tfc/mem $P & MW=$!; bench tfc 19420; fi
stop $P; wait $MW 2>/dev/null; cat tfc/mem.json 2>/dev/null
echo "== vllm 0.31 27b-awq, prefix caching on, MTP 0 $(date -Is)"; mkdir -p vllmc; unset LD_LIBRARY_PATH
(cd $VT && setsid bash tools/rdna/serve_rdna.sh RECIPE=27b-awq MODEL=$VM VENV=$HOME/Apps/vllm/venv-7.14.0_0.31.0 VLLM_TREE=$VT HIP_VISIBLE_DEVICES=2,3,4,5 ROCR_VISIBLE_DEVICES= PORT=18320 MTP=0 FEATURES="prefix-caching mamba-align language-model-only vision-cap prompt-tokens-details" > $D/vllmc/serve.log 2>&1 < /dev/null &)
if up 18320; then root=$(pgrep -f "entrypoints.cli.main serve.*--port 18320" | head -1); VP="$root $(kids $root | tr '\n' ' ')"; echo "vllm pids: $VP"
  grep -hoE "GPU KV cache size: [0-9,]+ tokens|enable_prefix_caching=[A-Za-z]+" vllmc/serve.log | sort -u
  bash $HOME/rcpt655/receipt/memwatch.sh vllmc/mem $VP & MW=$!; bench vllmc 18320; stop $VP; wait $MW 2>/dev/null; cat vllmc/mem.json
else echo "vllm not up"; tail -15 vllmc/serve.log | cut -c1-200; fi
echo CMPC-DONE $(date -Is)
