#!/bin/bash
# memmon.sh OUT FLAG GPUS PGIDS...: every 5 s while FLAG exists: VmHWM/VmRSS of each tensorfold-native in the groups, VRAM used per GPU
out=$1; flag=$2; gpus=$3; shift 3
while [ -e $flag ]; do
  t=$(date +%s.%N | cut -c1-14); line="{\"t\": $t, \"procs\": ["; sep=""
  for g in "$@"; do for p in $(pgrep -g $g); do
    [ "$(cat /proc/$p/comm 2>/dev/null)" = "tensorfold-nati" ] || continue
    h=$(awk '/^VmHWM/{print $2}' /proc/$p/status 2>/dev/null); r=$(awk '/^VmRSS/{print $2}' /proc/$p/status 2>/dev/null)
    [ -n "$h" ] && { line="$line$sep{\"pgid\": $g, \"vmhwm_kib\": $h, \"vmrss_kib\": $r}"; sep=", "; }
  done; done
  v=$(timeout -s KILL 20 rocm-smi --showmeminfo vram -d $gpus --json 2>/dev/null | tr -d '\n')
  echo "$line], \"rocm_smi_vram\": ${v:-null}}" >> $out
  sleep 5
done
