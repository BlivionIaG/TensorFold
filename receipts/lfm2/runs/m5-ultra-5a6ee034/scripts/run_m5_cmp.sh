set -u
T=~/tf-lfm2-test; B=$T/bin; M=$T/models/LFM2.5-24B-A2B-MLX-4bit; O=$T/runs/5a6ee034; PY=~/.local/bin/python3.12; MX=$T/mlxenv/bin; D=$T/tools/CHANGELOG.md
up(){ for i in $(seq 1 180); do sleep 2; curl -s localhost:8091/v1/models >/dev/null 2>&1 && return 0; done; return 1; }
stop(){ kill $1; while kill -0 $1 2>/dev/null; do sleep 1; done; sleep 3; }
echo "$($MX/python -c "import mlx_lm,mlx.core as m;print(\"mlx\",m.__version__,\"mlx-lm\",mlx_lm.__version__)")" >> $O/versions.txt
echo "== alternated 2k 8k"
for r in 1 2; do $B/tf-lfm2-bench $M 2048 2048 128 > $O/alt_tf_2k_r$r.txt 2>&1; $B/tf-lfm2-bench $M 8192 2048 128 > $O/alt_tf_8k_r$r.txt 2>&1; $MX/python $T/tools/mlx_cold.py $M $D 2048,8192 128 > $O/alt_mlx_r$r.txt 2>&1; done
echo "== alternated 32k 64k"
for r in 1 2; do $B/tf-lfm2-bench $M 32768 2048 64 > $O/alt_tf_32k_r$r.txt 2>&1; $MX/python $T/tools/mlx_cold.py $M $D 32768 64 > $O/mlx_32k_r$r.txt 2>&1; $B/tf-lfm2-bench $M 65536 2048 64 > $O/alt_tf_64k_r$r.txt 2>&1; $MX/python $T/tools/mlx_cold.py $M $D 65536 64 > $O/mlx_64k_r$r.txt 2>&1; done
echo "== 131k"
/usr/bin/time -l $MX/python $T/tools/mlx_cold.py $M $D 131000 64 > $O/mlx_131k.txt 2>&1
echo "== mlx server"
$MX/mlx_lm.server --model $M --port 8091 > $O/mlx_server.log 2>&1 & P=$!; up
$PY $T/tools/bench_openai.py http://127.0.0.1:8091 $M --tokens 256 --reps 3 --temperatures 0 --label mlx --output $O/mlx_decode.json > $O/mlx_bench.txt 2>&1
$PY $T/tools/bench_concurrent.py http://127.0.0.1:8091 $M --tokens 256 --reps 3 --temperatures 0 --levels 1,4 --alone --serial --output $O/mlx_concurrent.json > $O/mlx_concurrent.txt 2>&1
$PY $T/tools/two_turn.py http://127.0.0.1:8091 $M $O/mlx_two_turn.json $D > /dev/null 2>&1
vmmap -summary $P 2>/dev/null | grep -i "Physical footprint" > $O/mlx_footprint.txt; stop $P
echo "== tf server adjacent"
$B/tensorfold-native serve $M --port 8091 --context 131072 --prompt-cache-gib 0 --no-update-check > $O/served_on_r3.log 2>&1 & P=$!; up
$PY $T/tools/bench_openai.py http://127.0.0.1:8091 m --tokens 256 --reps 3 --temperatures 0 --label tf_on --output $O/tf_decode_on_r3.json > /dev/null 2>&1; stop $P
echo CMP DONE
