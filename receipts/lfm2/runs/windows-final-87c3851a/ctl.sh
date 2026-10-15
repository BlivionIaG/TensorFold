set -u
S=SCRATCH; O=$S/prfin; T=tf-lfm2-windows/tools; K=tf-lfm2-receipts/receipts/lfm2/tokens; P=SCRATCH/prefill_tokens.py
M=$HOME/.cache/huggingface/hub/models--LiquidAI--LFM2.5-24B-A2B-MLX-4bit/snapshots/667408d494bde191c2cb9dd4f4549827816b9275
quiet(){ while pgrep -x zig >/dev/null || pgrep -x tensorfold-native >/dev/null || pgrep -f "^[^ ]*mlxenv/bin/" >/dev/null; do sleep 10; done; sleep 10; }
up(){ for i in $(seq 1 180); do sleep 2; curl -s localhost:8091/v1/models >/dev/null 2>&1 && return 0; done; return 1; }
stop(){ kill $1; while kill -0 $1 2>/dev/null; do sleep 1; done; sleep 3; }
bin(){ case $1 in head) echo $S/m5build/native/bin/tensorfold-native;; p4) echo $S/bin-4-core-windows/native/bin/tensorfold-native;; p5) echo $S/bin-5-lfm2-windows/native/bin/tensorfold-native;; esac; }
export HF_HUB_OFFLINE=1
while ! grep -q LONG_DONE $O/progress.log; do sleep 20; done
ses(){ n=$1; r=$2; quiet
  if [ $n = mlx ]; then $S/mlxenv/bin/mlx_lm.server --model $M --port 8091 > $O/serve_${n}_r$r.log 2>&1 & PID=$!; up; MN=$M
  else $(bin $n) serve $M --port 8091 --context 131072 --prompt-cache-gib 0 --no-update-check > $O/serve_${n}_r$r.log 2>&1 & PID=$!; up; MN=m; fi
  python3 $T/bench_openai.py http://127.0.0.1:8091 $MN --tokens 256 --reps 3 --temperatures 0 --label $n --output $O/decode_${n}_r$r.json > /dev/null 2>&1
  python3 $T/bench_concurrent.py http://127.0.0.1:8091 $MN --tokens 256 --reps 3 --temperatures 0 --levels 1,4 --alone --serial --output $O/conc_${n}_r$r.json > $O/conc_${n}_r$r.txt 2>&1
  stop $PID; echo "$n r$r done $(date +%H:%M)" >> $O/progress.log; }
ses mlx 1; ses p5 5; ses head 5; ses mlx 2
echo CTL_DONE >> $O/progress.log
