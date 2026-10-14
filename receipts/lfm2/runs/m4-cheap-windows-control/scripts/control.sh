set -u
C=SCRATCH/cwcmp; S=SCRATCH; T=HOME/Projects/Infra/tf-lfm2-pr/tools
M=$HOME/.cache/huggingface/hub/models--LiquidAI--LFM2.5-24B-A2B-MLX-4bit/snapshots/667408d494bde191c2cb9dd4f4549827816b9275
DOC=HOME/Projects/Infra/tf-lfm2/CHANGELOG.md; MLXS=HOME/Projects/Infra/tf-lfm2-receipts/receipts/lfm2/scripts/mlx_cold.py
export HF_HUB_OFFLINE=1
while ! grep -q EXACT_DONE $C/exact_fix.log 2>/dev/null; do sleep 30; done
quiet(){ while pgrep -f "zig build|zig-0.17.0/zig build|zig build-exe" >/dev/null; do sleep 15; done; sleep 20; }
busy(){ pgrep -f "tensorfold-native serve|tf-lfm2-bench|tf-lfm2-exact|mlx_lm.server|mlx_cold.py" >/dev/null; }
while busy; do sleep 10; done
up(){ for i in $(seq 1 180); do sleep 2; curl -s localhost:8091/v1/models >/dev/null 2>&1 && return 0; done; return 1; }
stop(){ kill $1; while kill -0 $1 2>/dev/null; do sleep 1; done; sleep 5; }
bin(){ case $1 in head) echo $S/m5build;; fix) echo $S/fixbuild;; esac; }
served(){ n=$1; tag=$2
  quiet
  if [ $n = mlx ]; then $S/mlxenv/bin/mlx_lm.server --model $M --port 8091 > $C/ctl_${n}_$tag.log 2>&1 & P=$!; up; MN=$M
  else $(bin $n)/native/bin/tensorfold-native serve $M --port 8091 --context 131072 --prompt-cache-gib 0 --no-update-check > $C/ctl_${n}_$tag.log 2>&1 & P=$!; up; MN=m; fi
  python3 $T/bench_openai.py http://127.0.0.1:8091 $MN --tokens 256 --reps 3 --temperatures 0 --label $n --output $C/ctl_dec_${n}_$tag.json > /dev/null 2>&1
  python3 $T/bench_concurrent.py http://127.0.0.1:8091 $MN --tokens 256 --reps 3 --temperatures 0 --levels 1,4 --alone --serial --output $C/ctl_conc_${n}_$tag.json > $C/ctl_conc_${n}_$tag.txt 2>&1
  stop $P; echo "served $n $tag done $(date +%H:%M)" >> $C/control.log; }
served mlx a; served head a; served fix a; served mlx b; served fix b; served head b; served mlx c
inproc(){ n=$1; tag=$2
  quiet
  if [ $n = mlx ]; then $S/mlxenv/bin/python $MLXS $M $DOC 2048,32768 64 > $C/ctl_ip_${n}_$tag.txt 2>&1
  else $(bin $n)/bin/tf-lfm2-bench $M 2048 2048 128 > $C/ctl_ip_${n}_${tag}_2k.txt 2>&1; $(bin $n)/bin/tf-lfm2-bench $M 32768 2048 64 > $C/ctl_ip_${n}_${tag}_32k.txt 2>&1; fi
  echo "inproc $n $tag done $(date +%H:%M)" >> $C/control.log; }
inproc mlx a; inproc head a; inproc fix a; inproc mlx b; inproc fix b; inproc head b; inproc mlx c
echo CONTROL_DONE >> $C/control.log
