set -u
T=~/tf-lfm2-test; B=$T/bin; M=$T/models/LFM2.5-24B-A2B-MLX-4bit; O=$T/runs/5a6ee034; PY=~/.local/bin/python3.12; mkdir -p $O
up(){ for i in $(seq 1 180); do sleep 2; curl -s localhost:8091/v1/models >/dev/null 2>&1 && return 0; done; return 1; }
stop(){ kill $1; while kill -0 $1 2>/dev/null; do sleep 1; done; sleep 3; }
{ sw_vers; sysctl -n machdep.cpu.brand_string hw.memsize; system_profiler SPDisplaysDataType | grep -i cores; } > $O/versions.txt
echo "== exact"; $B/tf-lfm2-exact $M > $O/exact_4bit.txt 2>&1; tail -1 $O/exact_4bit.txt
echo "== bench in process"
for r in 1 2; do $B/tf-lfm2-bench $M 2048 2048 128 > $O/tf_2k_r$r.txt 2>&1; $B/tf-lfm2-bench $M 8192 2048 128 > $O/tf_8k_r$r.txt 2>&1; done
for r in 1 2; do $B/tf-lfm2-bench $M 32768 2048 64 > $O/tf_32k_r$r.txt 2>&1; $B/tf-lfm2-bench $M 65536 2048 64 > $O/tf_64k_r$r.txt 2>&1; done
/usr/bin/time -l $B/tf-lfm2-bench $M 131000 2048 64 > $O/tf_131k.txt 2>&1
echo "== served cold, drafts on"
$B/tensorfold-native serve $M --port 8091 --context 131072 --prompt-cache-gib 0 --no-update-check > $O/tf_cold.log 2>&1 & P=$!; up
$PY $T/tools/prefill_tokens.py http://127.0.0.1:8091 m $O/tf_prefill.json $T/tokens/prompt_2048.json:3 $T/tokens/prompt_8192.json:3 $T/tokens/prompt_32768.json:2 $T/tokens/prompt_65536.json:1 $T/tokens/prompt_131000.json:1 > /dev/null 2>&1
$PY $T/tools/bench_openai.py http://127.0.0.1:8091 m --tokens 256 --reps 3 --temperatures 0 --label tf --output $O/tf_decode.json > /dev/null 2>&1
$PY $T/tools/bench_concurrent.py http://127.0.0.1:8091 m --tokens 256 --reps 3 --temperatures 0 --levels 1,4 --alone --serial --output $O/tf_concurrent.json > $O/tf_concurrent.txt 2>&1
vmmap -summary $P 2>/dev/null | grep -i "Physical footprint" > $O/tf_cold_footprint.txt; stop $P
echo "== served, drafts off"
$B/tensorfold-native serve $M --port 8091 --context 131072 --prompt-cache-gib 0 --no-update-check --no-drafts > $O/tf_nodraft.log 2>&1 & P=$!; up
$PY $T/tools/bench_openai.py http://127.0.0.1:8091 m --tokens 256 --reps 3 --temperatures 0 --label tf_off --output $O/tf_decode_nodraft.json > /dev/null 2>&1
$PY $T/tools/bench_concurrent.py http://127.0.0.1:8091 m --tokens 256 --reps 3 --temperatures 0 --levels 1,4 --alone --serial --output $O/tf_concurrent_nodraft.json > $O/tf_concurrent_nodraft.txt 2>&1
stop $P
echo "== served, prompt cache"
$B/tensorfold-native serve $M --port 8091 --context 131072 --prompt-cache-gib 0 --no-update-check > $O/tf_two_cold.log 2>&1 & P=$!; up
$PY $T/tools/two_turn.py http://127.0.0.1:8091 m $O/tf_two_turn_cold.json $T/tools/CHANGELOG.md > /dev/null 2>&1; stop $P
$B/tensorfold-native serve $M --port 8091 --context 131072 --no-update-check > $O/tf_cache.log 2>&1 & P=$!; up
$PY $T/tools/two_turn.py http://127.0.0.1:8091 m $O/tf_two_turn.json $T/tools/CHANGELOG.md > /dev/null 2>&1
vmmap -summary $P 2>/dev/null | grep -i "Physical footprint" > $O/tf_cache_footprint.txt; stop $P
echo ALL DONE
