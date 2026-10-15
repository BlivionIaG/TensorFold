set -u
S=SCRATCH; O=$S/bench-m4-final2; T=tf-lfm2-windows/tools; K=tf-lfm2-receipts/receipts/lfm2/tokens; P=$S/prefill_tokens.py
M=$HOME/.cache/huggingface/hub/models--LiquidAI--LFM2.5-24B-A2B-MLX-4bit/snapshots/667408d494bde191c2cb9dd4f4549827816b9275
R=tf-lfm2-receipts/receipts/lfm2/scripts; MC=$R/mlx_cold.py; DOC=tf-lfm2-windows/CHANGELOG.md
G=$S/egate/progress.log
export HF_HUB_OFFLINE=1
log(){ echo "$* $(date +%H:%M:%S) | load $(sysctl -n vm.loadavg)
quiet(){ while [ -e $S/egate/HOLD ] || pgrep -x tensorfold-native >/dev/null || pgrep -f "tf-lfm2-(bench|exact)" >/dev/null || pgrep -f "mlxenv/bin/" >/dev/null; do echo "waiting $(date +%H:%M:%S)" >> $O/progress.log; sleep 15; done; sleep 10; }
up(){ for i in $(seq 1 300); do sleep 2; curl -s localhost:8091/v1/models >/dev/null 2>&1 && return 0; done; return 1; }
stop(){ kill $1; while kill -0 $1 2>/dev/null; do sleep 1; done; sleep 3; }
srv(){ case $1 in before) echo $S/m5build/native/bin/tensorfold-native;; tip) echo $S/bin-egate/native/bin/tensorfold-native;; esac; }
bch(){ case $1 in before) echo $S/m5build/bin/tf-lfm2-bench;; tip) echo $S/bin-egate/bin/tf-lfm2-bench;; esac; }
until grep -q ALL_DONE $O/progress.log; do sleep 30; done
ses(){ n=$1; r=$2; quiet; log "session $n r$r start"
  if [ $n = mlx ]; then $S/mlxenv/bin/mlx_lm.server --model $M --port 8091 > $O/serve_${n}_r$r.log 2>&1 & PID=$!; up || echo "$n r$r NOT UP" >> $O/progress.log; MN=$M
  else $(srv $n) serve $M --port 8091 --context 131072 --prompt-cache-gib 0 --no-update-check > $O/serve_${n}_r$r.log 2>&1 & PID=$!; up || echo "$n r$r NOT UP" >> $O/progress.log; MN=m
    L="$K/prompt_2048.json:2 $K/prompt_8192.json:2 $K/prompt_32768.json:1 $K/prompt_65536.json:1"; [ $r = 1 ] && L="$L $K/prompt_131000.json:1"
    python3 $P http://127.0.0.1:8091 m $O/prefill_${n}_r$r.json $L > $O/prefill_${n}_r$r.txt 2>&1
    python3 $T/bench_cache.py http://127.0.0.1:8091 m --tokens 64 --repeats 1 --streams 1 --prompt-file $K/prompt_32768.json --expected-prompt-tokens 32768 --require-cold --output $O/cache_${n}_r$r.json > $O/cache_${n}_r$r.txt 2>&1
  fi
  echo "=== MARK decode" >> $O/serve_${n}_r$r.log
  python3 $T/bench_openai.py http://127.0.0.1:8091 $MN --tokens 256 --reps 3 --temperatures 0 --label ${n}_r$r --output $O/decode_${n}_r$r.json > $O/decode_${n}_r$r.txt 2>&1
  echo "=== MARK concurrent" >> $O/serve_${n}_r$r.log
  python3 $T/bench_concurrent.py http://127.0.0.1:8091 $MN --tokens 256 --reps 3 --temperatures 0 --levels 1,4 --alone --serial --output $O/conc_${n}_r$r.json > $O/conc_${n}_r$r.txt 2>&1
  vmmap -summary $PID 2>/dev/null | grep -i "Physical footprint" > $O/footprint_${n}_r$r.txt
  stop $PID; log "session $n r$r done"; }
ses mlx 1rerun
log RERUN_DONE
