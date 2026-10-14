set -u
C=SCRATCH/cwcmp; S=SCRATCH
M=$HOME/.cache/huggingface/hub/models--LiquidAI--LFM2.5-24B-A2B-MLX-4bit/snapshots/667408d494bde191c2cb9dd4f4549827816b9275
DOC=HOME/Projects/Infra/tf-lfm2/CHANGELOG.md; MLXS=HOME/Projects/Infra/tf-lfm2-receipts/receipts/lfm2/scripts/mlx_cold.py
export HF_HUB_OFFLINE=1
quiet(){ while pgrep -x zig >/dev/null || pgrep -x tensorfold-native >/dev/null || pgrep -x tf-lfm2-bench >/dev/null || pgrep -x tf-lfm2-exact >/dev/null || pgrep -f "^[^ ]*mlxenv/bin/" >/dev/null; do sleep 15; done; sleep 10; }
bin(){ case $1 in head) echo $S/m5build;; fix) echo $S/fixbuild;; esac; }
inproc(){ n=$1; tag=$2; quiet
  if [ $n = mlx ]; then $S/mlxenv/bin/python $MLXS $M $DOC 2048,32768 64 > $C/ctl_ip_${n}_$tag.txt 2>&1
  else $(bin $n)/bin/tf-lfm2-bench $M 2048 2048 128 > $C/ctl_ip_${n}_${tag}_2k.txt 2>&1; $(bin $n)/bin/tf-lfm2-bench $M 32768 2048 64 > $C/ctl_ip_${n}_${tag}_32k.txt 2>&1; fi
  echo "inproc $n $tag done $(date +%H:%M)" >> $C/control.log; }
inproc mlx a; inproc head a; inproc fix a; inproc mlx b; inproc fix b; inproc head b; inproc mlx c
echo CONTROL_DONE >> $C/control.log
