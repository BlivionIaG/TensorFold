#!/bin/zsh
# LFM2 receipts, TensorFold vs mlx-lm, one model process at a time. Usage: receipts_lfm2.sh OUTDIR
set -u
S=/private/tmp/claude-501/-Users-kletorch-Projects-Infra/c49e3d99-57d9-496c-b831-a1315198c160/scratchpad
R=/Users/kletorch/Projects/Infra/tf-lfm2
PY=$S/mlxenv/bin/python
snap() { ls -d $HOME/.cache/huggingface/hub/models--LiquidAI--LFM2.5-24B-A2B-MLX-$1bit/snapshots/*; }
M=$(snap 4)
O=$1; mkdir -p $O
up() { for i in $(seq 1 180); do sleep 2; curl -s localhost:8091/v1/models >/dev/null 2>&1 && return 0; done; return 1; }
peak() { vmmap -summary $1 2>/dev/null | grep -i "Physical footprint" > $2; }
stop() { kill $1; while kill -0 $1 2>/dev/null; do sleep 1; done; sleep 3; }
echo "build $(git -C $R rev-parse --short HEAD)" > $O/versions.txt
$PY -c 'import mlx.core as mx, mlx_lm; print("mlx", mx.__version__, "mlx-lm", mlx_lm.__version__)' >> $O/versions.txt
sw_vers >> $O/versions.txt; sysctl -n machdep.cpu.brand_string hw.memsize >> $O/versions.txt
for b in 4 6 8; do echo "$b-bit $(basename $(snap $b))" >> $O/versions.txt; done

echo "== 1 TensorFold server, prompt cache off"
$R/zig-out/native/bin/tensorfold-native serve $M --port 8091 --context 131072 --prompt-cache-gib 0 --no-update-check > $O/tf_cold.log 2>&1 &
P=$!; up || exit 1
python3 $S/prefill_tokens.py http://127.0.0.1:8091 m $O/tf_prefill.json $S/prompt_2048.json:3 $S/prompt_8192.json:3 $S/prompt_32768.json:2 $S/prompt_65536.json:1 $S/prompt_131000.json:1
python3 $R/tools/bench_openai.py http://127.0.0.1:8091 m --tokens 256 --reps 3 --temperatures 0 --label tf --output $O/tf_decode.json
python3 $R/tools/bench_concurrent.py http://127.0.0.1:8091 m --tokens 256 --reps 3 --temperatures 0 --levels 1,4 --alone --serial --output $O/tf_concurrent.json
python3 $S/two_turn.py http://127.0.0.1:8091 m $O/tf_two_turn_cold.json
peak $P $O/tf_cold_footprint.txt; stop $P

echo "== 2 TensorFold server, prompt cache on"
$R/zig-out/native/bin/tensorfold-native serve $M --port 8091 --context 131072 --no-update-check > $O/tf_cache.log 2>&1 &
P=$!; up || exit 1
python3 $S/two_turn.py http://127.0.0.1:8091 m $O/tf_two_turn.json
peak $P $O/tf_cache_footprint.txt; stop $P

echo "== 3 mlx-lm server"
$S/mlxenv/bin/mlx_lm.server --model $M --port 8091 > $O/mlx_server.log 2>&1 &
P=$!; up || exit 1
python3 $R/tools/bench_openai.py http://127.0.0.1:8091 $M --tokens 256 --reps 3 --temperatures 0 --label mlx --output $O/mlx_decode.json
python3 $R/tools/bench_concurrent.py http://127.0.0.1:8091 $M --tokens 256 --reps 3 --temperatures 0 --levels 1,4 --alone --serial --output $O/mlx_concurrent.json
python3 $S/two_turn.py http://127.0.0.1:8091 $M $O/mlx_two_turn.json
peak $P $O/mlx_footprint.txt; stop $P

echo "== 4 alternated, in process, 4-bit (2 rounds)"
for round in 1 2; do
  $R/zig-out/bin/tf-lfm2-bench $M 8192 2048 128 > $O/alt_tf_8k_r$round.txt 2>&1
  $R/zig-out/bin/tf-lfm2-bench $M 2048 2048 128 > $O/alt_tf_2k_r$round.txt 2>&1
  $PY $S/mlx_cold.py $M $R/CHANGELOG.md 2048,8192 128 > $O/alt_mlx_r$round.txt 2>&1
done

echo "== 5 long prompts in process"
$R/zig-out/bin/tf-lfm2-bench $M 32768 2048 64 > $O/tf_32k.txt 2>&1
$R/zig-out/bin/tf-lfm2-bench $M 65536 2048 64 > $O/tf_64k.txt 2>&1
/usr/bin/time -l $PY $S/mlx_cold.py $M $R/CHANGELOG.md 32768,65536,131000 64 > $O/mlx_long.txt 2>&1

echo "== 6 6- and 8-bit in process"
for b in 6 8; do
  $R/zig-out/bin/tf-lfm2-bench $(snap $b) 8192 2048 128 > $O/tf_${b}bit_8k.txt 2>&1
  $R/zig-out/bin/tf-lfm2-bench $(snap $b) 32768 2048 64 > $O/tf_${b}bit_32k.txt 2>&1
  $PY $S/mlx_cold.py $(snap $b) $R/CHANGELOG.md 2048,8192,32768 128 > $O/mlx_${b}bit.txt 2>&1
done

echo "== 7 fidelity, final build"
for b in 4 6 8; do $PY $R/tools/zig/lfm2_fidelity.py $(snap $b) $O/fidelity_${b}bit > $O/fidelity_${b}bit.log 2>&1; done
echo done
