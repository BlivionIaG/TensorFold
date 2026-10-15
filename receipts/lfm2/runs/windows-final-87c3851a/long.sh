set -u
S=SCRATCH; O=$S/prfin; T=tf-lfm2-windows/tools; K=tf-lfm2-receipts/receipts/lfm2/tokens; P=SCRATCH/prefill_tokens.py
M=$HOME/.cache/huggingface/hub/models--LiquidAI--LFM2.5-24B-A2B-MLX-4bit/snapshots/667408d494bde191c2cb9dd4f4549827816b9275
quiet(){ while pgrep -x zig >/dev/null || pgrep -x tensorfold-native >/dev/null || pgrep -f "^[^ ]*mlxenv/bin/" >/dev/null; do sleep 10; done; sleep 10; }
up(){ for i in $(seq 1 180); do sleep 2; curl -s localhost:8091/v1/models >/dev/null 2>&1 && return 0; done; return 1; }
stop(){ kill $1; while kill -0 $1 2>/dev/null; do sleep 1; done; sleep 3; }
bin(){ case $1 in head) echo $S/m5build/native/bin/tensorfold-native;; p4) echo $S/bin-4-core-windows/native/bin/tensorfold-native;; p5) echo $S/bin-5-lfm2-windows/native/bin/tensorfold-native;; esac; }
long(){ n=$1; quiet
  $(bin $n) serve $M --port 8091 --context 131072 --prompt-cache-gib 0 --no-update-check > $O/serve_${n}_r$2.log 2>&1 & PID=$!; up
  python3 $P http://127.0.0.1:8091 m $O/prefill_${n}_r$2.json $K/prompt_131000.json:1 > /dev/null 2>&1
  stop $PID; echo "$n r$2 done $(date +%H:%M)" >> $O/progress.log; }
long p5 3; long p4 3; long head 3; long head 4; long p4 4; long p5 4
echo LONG_DONE >> $O/progress.log
