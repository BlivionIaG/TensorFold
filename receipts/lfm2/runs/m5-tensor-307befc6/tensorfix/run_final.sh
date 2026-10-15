set -u
T=~/tf-lfm2-test; X=$T/tensorfix; S=$X/src; O=$X/final; B=$X/bin; M=$T/models/LFM2.5-24B-A2B-MLX-4bit; PY=$T/mlxenv/bin/python3
mkdir -p $O; cd $T
cp $S/zig-out/bin/tf-lfm2-exact $S/zig-out/bin/tf-lfm2-bench $S/zig-out/bin/tf-tile-check $S/zig-out/bin/tf-lfm2-fp64 $S/zig-out/bin/tf-lfm2-forward $B/ && cp $S/zig-out/native/bin/tensorfold-native $B/
$B/tf-tile-check > $O/tile_check.txt 2>&1; echo "tile-check exit $?" >> $O/tile_check.txt
$B/tf-lfm2-exact $M > $O/exact_4bit.txt 2>&1; echo "exact exit $?" >> $O/exact_4bit.txt
$B/tf-lfm2-fp64 $M > $O/fp64.txt 2>&1; echo "fp64 exit $?" >> $O/fp64.txt
up(){ for i in $(seq 1 180); do sleep 2; curl -s localhost:8091/v1/models >/dev/null 2>&1 && return 0; done; return 1; }
stop(){ kill $1; while kill -0 $1 2>/dev/null; do sleep 1; done; sleep 3; }
for mode in on off; do
  F=""; [ $mode = off ] && F="--no-drafts"
  $B/tensorfold-native serve $M --port 8091 --context 131072 --prompt-cache-gib 0 --no-update-check $F > $O/serve_$mode.log 2>&1 & P=$!; up
  $PY tools/bench_openai.py http://127.0.0.1:8091 m --tokens 256 --reps 3 --temperatures 0 --label tf_$mode --output $O/decode_$mode.json > $O/bench_$mode.txt 2>&1
  $PY tools/bench_concurrent.py http://127.0.0.1:8091 m --tokens 256 --reps 3 --temperatures 0 --levels 1,4 --alone --serial --output $O/conc_$mode.json > $O/conc_$mode.txt 2>&1
  stop $P
done
$B/tf-lfm2-bench $M 2048 2048 128 > $O/tf_2k.txt 2>&1
$B/tf-lfm2-bench $M 32768 8192 64 > $O/tf_32k.txt 2>&1
/usr/bin/time -l $B/tf-lfm2-bench $M 131000 8192 64 > $O/tf_131k.txt 2>&1
/usr/bin/time -l $B/tf-lfm2-bench $M 131000 2048 64 > $O/tf_131k_c2048.txt 2>&1
echo DONE > $O/final_done
