set -u
T=~/tf-lfm2-test; B=$T/bin; M=$T/models/LFM2.5-24B-A2B-MLX-4bit; O=$T/runs/5a6ee034; PY=~/.local/bin/python3.12
up(){ for i in $(seq 1 180); do sleep 2; curl -s localhost:8091/v1/models >/dev/null 2>&1 && return 0; done; return 1; }
stop(){ kill $1; while kill -0 $1 2>/dev/null; do sleep 1; done; sleep 3; }
for r in 1 2; do for mode in on off; do
  F=""; [ $mode = off ] && F="--no-drafts"
  $B/tensorfold-native serve $M --port 8091 --context 131072 --prompt-cache-gib 0 --no-update-check $F > $O/served_${mode}_r$r.log 2>&1 & P=$!; up
  $PY $T/tools/bench_openai.py http://127.0.0.1:8091 m --tokens 256 --reps 3 --temperatures 0 --label tf_$mode --output $O/tf_decode_${mode}_r$r.json > $O/bench_${mode}_r$r.txt 2>&1
  [ $r = 1 ] && $PY $T/tools/bench_concurrent.py http://127.0.0.1:8091 m --tokens 256 --reps 3 --temperatures 0 --levels 1,4 --alone --serial --output $O/tf_concurrent_$mode.json > $O/tf_concurrent_$mode.txt 2>&1
  stop $P
done; done; echo SERVED DONE
