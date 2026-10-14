#!/bin/sh
S=SCRATCH; G=$S/cwgate
T=tf-lfm2-windows/tools
M=$(ls -d $HOME/.cache/huggingface/hub/models--LiquidAI--LFM2.5-24B-A2B-MLX-4bit/snapshots/*)
i=0
for who in head final final head head final; do
  i=$((i+1))
  case $who in head) B=$S/m5build/native/bin/tensorfold-native;; final) B=$S/cw-out/native/bin/tensorfold-native;; esac
  sleep 20
  $B serve $M --port 8091 --context 131072 --prompt-cache-gib 0 --no-update-check --no-drafts > $G/ab_${i}_$who.log 2>&1 & P=$!
  for j in $(seq 1 300); do curl -s localhost:8091/v1/models >/dev/null 2>&1 && break; sleep 1; done
  sleep 2
  python3 $T/bench_openai.py http://127.0.0.1:8091 m --tokens 256 --reps 3 --temperatures 0 --label $who --output $G/ab_${i}_$who.json > /dev/null 2>&1
  kill $P; wait $P 2>/dev/null
  echo "$i $who $(grep 'window ms' $G/ab_${i}_$who.log)" >> $G/ab.log
done
echo AB_DONE >> $G/ab.log
