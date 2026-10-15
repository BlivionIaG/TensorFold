#!/bin/zsh
cd ~/tf-lfm2-test/f4; mkdir -p served
M=~/tf-lfm2-test/models/LFM2.5-24B-A2B-MLX-4bit; PY=~/.local/bin/python3.12
up(){ for i in $(seq 1 180); do sleep 2; curl -s localhost:8091/v1/models >/dev/null 2>&1 && return 0; done; return 1; }
for tr in "$@"; do t=${tr%%:*}; r=${tr##*:}
  until mkdir ~/tf-lfm2-test/.model_lock 2>/dev/null; do sleep 20; done
  ./out_$t/native/bin/tensorfold-native serve $M --port 8091 --context 131072 --prompt-cache-gib 0 --no-update-check > served/serve_${t}_$r.log 2>&1 & P=$!
  up
  $PY bench_openai.py http://127.0.0.1:8091 m --tokens 256 --reps 3 --temperatures 0 --label $t --output served/decode_${t}_$r.json > served/decode_${t}_$r.txt 2>&1
  $PY bench_concurrent.py http://127.0.0.1:8091 m --tokens 256 --reps 3 --temperatures 0 --levels 1,4 --alone --serial --output served/conc_${t}_$r.json > served/conc_${t}_$r.txt 2>&1
  kill $P; while kill -0 $P 2>/dev/null; do sleep 1; done; sleep 2
  rmdir ~/tf-lfm2-test/.model_lock
  echo "$t r$r done $(date +%H:%M)" >> served/progress.log
done
