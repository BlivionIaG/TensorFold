set -u
T=~/tf-lfm2-test; M=$T/models/LFM2.5-24B-A2B-MLX-4bit; O=$T/bench-final; PY=~/.local/bin/python3.12; MX=$T/mlxenv/bin; D=$T/tools/CHANGELOG.md; K=$T/tokens; mkdir -p $O
BB=$T/bin/tensorfold-native; BT=$T/final8/out/native/bin/tensorfold-native; XB=$T/bin/tf-lfm2-bench; XT=$T/final8/out/bin/tf-lfm2-bench; BC=$T/final8/src/tools/bench_cache.py
log(){ echo "$(date +%H:%M:%S) $*" >> $O/progress.log; }
idle(){ if pgrep -x tensorfold-native >/dev/null || pgrep -f "tf-lfm2-(bench|exact)" >/dev/null || pgrep -f mlx_lm >/dev/null; then log "BUSY before $1: $(pgrep -lf 'tensorfold-native|tf-lfm2-|mlx_lm' | tr '\n' ';')"; while pgrep -x tensorfold-native >/dev/null || pgrep -f "tf-lfm2-(bench|exact)" >/dev/null || pgrep -f mlx_lm >/dev/null; do sleep 5; done; fi; }
up(){ for i in $(seq 1 180); do sleep 2; curl -s localhost:8091/v1/models >/dev/null 2>&1 && return 0; done; return 1; }
stop(){ kill $1; while kill -0 $1 2>/dev/null; do sleep 1; done; sleep 3; }
{ sw_vers; sysctl -n machdep.cpu.brand_string hw.memsize; $MX/python -c "import mlx_lm,mlx.core as m;print(\"mlx\",m.__version__,\"mlx-lm\",mlx_lm.__version__)"; echo "before=$BB (5a6ee034) tip=$BT (74aff635)"; shasum -a 256 $BB $BT $XB $XT; } > $O/versions.txt 2>&1
log START
ses(){ n=$1; r=$2; long=$3
 idle ses_${n}_r$r
 if [ $n = mlx ]; then $MX/mlx_lm.server --model $M --port 8091 > $O/serve_${n}_r$r.log 2>&1 & P=$!; up || log "UP FAIL $n r$r"; MN=$M
 else B=$BB; [ $n = tip ] && B=$BT; $B serve $M --port 8091 --context 131072 --prompt-cache-gib 0 --no-update-check > $O/serve_${n}_r$r.log 2>&1 & P=$!; up || log "UP FAIL $n r$r"; MN=m
  L="$K/prompt_2048.json:2 $K/prompt_8192.json:2 $K/prompt_32768.json:1 $K/prompt_65536.json:1"; [ $long = y ] && L="$L $K/prompt_131000.json:1"
  $PY $T/tools/prefill_tokens.py http://127.0.0.1:8091 m $O/prefill_${n}_r$r.json $L > $O/prefill_${n}_r$r.txt 2>&1; log "$n r$r prefill exit $?"
  $PY $BC http://127.0.0.1:8091 m --tokens 64 --repeats 1 --streams 1 --prompt-file $K/prompt_32768.json --expected-prompt-tokens 32768 --require-cold --output $O/cache_${n}_r$r.json > $O/cache_${n}_r$r.txt 2>&1; log "$n r$r cache exit $?"; fi
 $PY $T/tools/bench_openai.py http://127.0.0.1:8091 $MN --tokens 256 --reps 3 --temperatures 0 --label ${n}_r$r --output $O/decode_${n}_r$r.json > $O/decode_${n}_r$r.txt 2>&1; log "$n r$r decode exit $?"
 $PY $T/tools/bench_concurrent.py http://127.0.0.1:8091 $MN --tokens 256 --reps 3 --temperatures 0 --levels 1,4 --alone --serial --output $O/conc_${n}_r$r.json > $O/conc_${n}_r$r.txt 2>&1; log "$n r$r conc exit $?"
 vmmap -summary $P 2>/dev/null | grep -i "Physical footprint" > $O/footprint_${n}_r$r.txt
 stop $P; log "$n r$r done"; }
ses mlx 1 n; ses before 1 y; ses tip 1 y; ses tip 2 n; ses before 2 n; ses mlx 2 n
log SERVED_DONE
ip(){ w=$1; r=$2
 for L in 2048:2048:128 8192:8192:128 32768:8192:64 65536:8192:64; do IFS=: read n c t <<< "$L"
  idle ip_${w}_${n}_r$r
  if [ $w = mlx ]; then $MX/python $T/tools/mlx_cold.py $M $D $n $t > $O/ip_mlx_${n}_r$r.txt 2>&1
  else X=$XB; [ $w = tip ] && X=$XT; $X $M $n $c $t > $O/ip_${w}_${n}_r$r.txt 2>&1; fi
  log "ip $w $n r$r exit $?"; done; }
ip before 1; ip tip 1; ip mlx 1; ip tip 2; ip before 2; ip mlx 2
idle ip131_before; /usr/bin/time -l $XB $M 131000 8192 64 > $O/ip131_before.txt 2>&1; log "ip131 before exit $?"
idle ip131_tip; /usr/bin/time -l $XT $M 131000 8192 64 > $O/ip131_tip.txt 2>&1; log "ip131 tip exit $?"
idle ip131_mlx; /usr/bin/time -l $MX/python $T/tools/mlx_cold.py $M $D 131000 64 > $O/ip131_mlx.txt 2>&1; log "ip131 mlx exit $?"
log ALL_DONE
