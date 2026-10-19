#!/bin/bash
# drive.sh RIG DEVS PORT TP1 PLAN: run every hold line of PLAN (TOPIC HOLD TREE MODE) not yet done, in order
rig=$1; devs=$2; port=$3; tp1=$4; plan=$5; R=$RCP
while read topic hold tree mode; do
  [ -z "$topic" ] && continue; [ "${topic:0:1}" = "#" ] && continue
  [ -e $R/out/$topic/$rig/${hold}_${tree}_${mode%%:*}/.done ] && continue
  mkdir -p $R/out/$topic/$rig
  bash $R/hold.sh $rig $devs $port $topic $hold $tree "$mode" "$tp1" < /dev/null
  port=$((port + 2))
done < $plan
echo "$(date +%F_%T) PLAN $plan DONE" >> $R/out/drive_$rig.log
