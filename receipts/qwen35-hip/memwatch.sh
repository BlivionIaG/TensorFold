#!/bin/sh
# memwatch.sh OUT PID...: every second, peak RSS (VmHWM) and VRAM (rocm-smi --showpids) of exactly these PIDs, until they exit
out=$1; shift; pids="$*"; : > $out.samples
while :; do
  alive=0; for p in $pids; do kill -0 $p 2>/dev/null && alive=1; done; [ $alive = 0 ] && break
  v=$(rocm-smi --showpids 2>/dev/null | awk -v want=" $pids " 'index(want, " "$1" ") {s += $4} END {print s+0}')
  r=0; h=0; for p in $pids; do x=$(awk '/VmRSS/{print $2}' /proc/$p/status 2>/dev/null); y=$(awk '/VmHWM/{print $2}' /proc/$p/status 2>/dev/null); r=$((r + ${x:-0})); h=$((h + ${y:-0})); done
  echo "$(date +%s) $v $r $h" >> $out.samples; sleep 1
done
awk '{if ($2>v) v=$2; if ($3>r) r=$3; if ($4>h) h=$4} END {printf "{\"peak_vram_bytes\": %d, \"peak_rss_kib\": %d, \"peak_hwm_kib_sum\": %d, \"samples\": %d}\n", v, r, h, NR}' $out.samples > $out.json
