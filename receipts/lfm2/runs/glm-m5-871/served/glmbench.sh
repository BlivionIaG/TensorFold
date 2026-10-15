set -u
T=~/tf-lfm2-test; O=$T/glmbench; mkdir -p $O; PY=~/.local/bin/python3.12; D=$T/tools/CHANGELOG.md
M=$(ls -d ~/.cache/huggingface/hub/models--Vontra--GLM-5.3-Flash-MLX-4bit-MTP/snapshots/*/ | head -1)
up(){ for i in $(seq 1 450); do sleep 2; curl -s localhost:8091/v1/models >/dev/null 2>&1 && return 0; kill -0 $1 2>/dev/null || return 1; done; return 1; }
stop(){ kill $1; while kill -0 $1 2>/dev/null; do sleep 1; done; sleep 5; }
ses(){ g=$1; r=$2
 until mkdir $T/.model_lock 2>/dev/null; do sleep 10; done
 $T/$g/out/native/bin/tensorfold-native serve $M --port 8091 --context 32768 --prompt-cache-gib 0 --no-update-check > $O/serve_${g}_r$r.log 2>&1 & P=$!
 if up $P; then
  $PY $T/tools/bench_openai.py http://127.0.0.1:8091 m --tokens 256 --reps 2 --temperatures 0 --label $g --output $O/decode_${g}_r$r.json > $O/decode_${g}_r$r.txt 2>&1
  $PY $T/tools/bench_concurrent.py http://127.0.0.1:8091 m --tokens 256 --reps 1 --temperatures 0 --levels 1,4 --alone --serial --output $O/conc_${g}_r$r.json > $O/conc_${g}_r$r.txt 2>&1
  $PY $T/tools/two_turn.py http://127.0.0.1:8091 m $O/two_turn_${g}_r$r.json $D > $O/two_turn_${g}_r$r.txt 2>&1
  vmmap -summary $P 2>/dev/null | grep -i "Physical footprint" > $O/footprint_${g}_r$r.txt
  echo "$g r$r done $(date +%H:%M)" >> $O/progress.log
 else echo "$g r$r SERVE FAILED $(date +%H:%M)" >> $O/progress.log; fi
 kill -0 $P 2>/dev/null && stop $P; rmdir $T/.model_lock; }
ses g868 1; ses g871 1; ses g871 2; ses g868 2
echo ALL_DONE >> $O/progress.log
