set -u
T=~/tf-lfm2-test; M=$T/models/LFM2.5-24B-A2B-MLX-4bit; O=$T/final7/runs; PY=~/.local/bin/python3.12; MX=$T/mlxenv/bin; D=$T/tools/CHANGELOG.md; K=$T/tokens; mkdir -p $O
while ! grep -q BUILD_DONE $T/build567.log; do sleep 20; done
grep -q "exit [1-9]" $T/build567.log && { echo BUILD_FAILED >> $O/progress.log; exit 1; }
B5=$T/final5/out/native/bin/tensorfold-native; B7=$T/final7/out/native/bin/tensorfold-native; X7=$T/final7/out/bin
up(){ for i in $(seq 1 180); do sleep 2; curl -s localhost:8091/v1/models >/dev/null 2>&1 && return 0; done; return 1; }
stop(){ kill $1; while kill -0 $1 2>/dev/null; do sleep 1; done; sleep 3; }
{ sw_vers; sysctl -n machdep.cpu.brand_string; $MX/python -c "import mlx_lm,mlx.core as m;print(\"mlx\",m.__version__,\"mlx-lm\",mlx_lm.__version__)"; cat $T/final5/src/.commit $T/final6/src/.commit $T/final7/src/.commit; } > $O/versions.txt
$T/final6/out/bin/tf-tile-check > $O/tile_check_6.txt 2>&1; echo "tile6 exit $?" >> $O/progress.log
$X7/tf-tile-check > $O/tile_check_7.txt 2>&1; echo "tile7 exit $?" >> $O/progress.log
$X7/tf-lfm2-exact $M > $O/exact_4bit.txt 2>&1; echo "exact exit $?" >> $O/progress.log
$X7/tf-lfm2-fp64 $M > $O/fp64.txt 2>&1; echo "fp64 exit $?" >> $O/progress.log
echo GATES_DONE >> $O/progress.log
# in process: tf-lfm2-bench for part 7 (tensor) against mlx_cold, alternated
for r in 1 2; do
 for L in 2048:2048:128 8192:8192:128 32768:8192:64 65536:8192:64; do IFS=: read n c t <<< "$L"; $X7/tf-lfm2-bench $M $n $c $t > $O/ip_t7_${n}_r$r.txt 2>&1; done
 $MX/python $T/tools/mlx_cold.py $M $D 2048,8192 128 > $O/ip_mlx_2k8k_r$r.txt 2>&1; $MX/python $T/tools/mlx_cold.py $M $D 32768 64 > $O/ip_mlx_32k_r$r.txt 2>&1; $MX/python $T/tools/mlx_cold.py $M $D 65536 64 > $O/ip_mlx_64k_r$r.txt 2>&1
done
echo "inproc done" >> $O/progress.log
ses(){ n=$1; r=$2; long=$3
 if [ $n = mlx ]; then $MX/mlx_lm.server --model $M --port 8091 > $O/serve_${n}_r$r.log 2>&1 & P=$!; up; MN=$M
 else B=$B5; [ $n = t7 ] && B=$B7; $B serve $M --port 8091 --context 131072 --prompt-cache-gib 0 --no-update-check > $O/serve_${n}_r$r.log 2>&1 & P=$!; up; MN=m
  L="$K/prompt_2048.json:2 $K/prompt_8192.json:2 $K/prompt_32768.json:1 $K/prompt_65536.json:1"; [ $long = y ] && L="$L $K/prompt_131000.json:1"
  $PY $T/tools/prefill_tokens.py http://127.0.0.1:8091 m $O/prefill_${n}_r$r.json $L > /dev/null 2>&1
  $PY $T/final7/src/tools/bench_cache.py http://127.0.0.1:8091 m --tokens 64 --repeats 1 --streams 1 --prompt-file $K/prompt_32768.json --expected-prompt-tokens 32768 --require-cold --output $O/cache_${n}_r$r.json > $O/cache_${n}_r$r.txt 2>&1; fi
 $PY $T/tools/bench_openai.py http://127.0.0.1:8091 $MN --tokens 256 --reps 3 --temperatures 0 --label $n --output $O/decode_${n}_r$r.json > /dev/null 2>&1
 $PY $T/tools/bench_concurrent.py http://127.0.0.1:8091 $MN --tokens 256 --reps 3 --temperatures 0 --levels 1,4 --alone --serial --output $O/conc_${n}_r$r.json > $O/conc_${n}_r$r.txt 2>&1
 vmmap -summary $P 2>/dev/null | grep -i "Physical footprint" > $O/footprint_${n}_r$r.txt
 stop $P; echo "$n r$r done $(date +%H:%M)" >> $O/progress.log; }
ses mlx 1 n; ses p5 1 y; ses t7 1 y; ses t7 2 n; ses p5 2 n; ses mlx 2 n
echo ALL_DONE >> $O/progress.log
