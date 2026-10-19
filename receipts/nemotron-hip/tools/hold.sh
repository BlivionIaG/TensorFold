#!/bin/bash
# hold.sh RIG DEVS PORT TOPIC HOLD TREE MODE [tp1]: one hold, a fresh server; MODE prefill:SIZES | decode | decodeoff | conc | conc8 | long:SIZES (memory sampled)
rig=$1; devs=$2; port=$3; topic=$4; hold=$5; tree=$6; mode=$7; tp1=$8
mtag=${mode%%:*}; label=rcp_${rig}_${topic}_$hold
. $WORK/libn.sh
R=$RCP; T=$TREE/tools; D=$R/ids; H=$R/out/$topic/$rig/${hold}_${tree}_$mtag; L=$R/out/$topic/$rig/driver.log
rm -rf $H; mkdir -p $H
note() { echo "$(date +%F_%T) $*" >> $L; }
bin=$(dir $tree)/out-$arch/native-hip/bin/tensorfold-native
commit=$(git -c gc.auto=0 -c maintenance.auto=false -C $(dir $tree) rev-parse HEAD)
bsha=$(sha256sum $bin | cut -c1-64)
tsha=$(sha256sum $(dir $tree)/out-$arch/bin/tf-nemotron-hip-test | cut -c1-64)
map=(9 8 4 5 6 7); gpus=""; for d in ${devs//,/ }; do gpus="$gpus ${map[$d]}"; done  # HIP index -> rocm-smi GPU index, by PCI bus
A="source_commit=$commit binary_sha256=$bsha rig=$rig hold=$hold tree=$tree"
vram() { timeout -s KILL 20 rocm-smi --showmeminfo vram -d $gpus --json 2>/dev/null | tr -d '\n'; }
case $mode in
  prefill:*|long:*) sargs="--context 262144 --parallel 1 --prompt-cache-gib 0"; LIMIT=14400;;
  decode|conc) sargs="--context 65536 --parallel 4"; LIMIT=3600;;
  decodeoff) sargs="--context 65536 --parallel 4 --no-drafts"; LIMIT=3600;;
  conc8) sargs="--context 32768 --parallel 8"; LIMIT=3600;;
esac
note "hold $hold $tree $mode start: commit $commit binary $bsha, serve $sargs"
v0=$(vram); t0=$(date +%s.%N)
serve $tree $mtag $port $sargs || { note "hold $hold $tree: server did not start"; stop; exit 1; }
t1=$(date +%s.%N)
rccl=$(for p in $pids; do rcclseen $p; done | tr ' ' '\n' | sort -u | xargs -n1 basename 2>/dev/null | tr '\n' ' ')
if [ $mtag = long ]; then touch $H/.mon; bash $R/memmon.sh $H/memory_samples.jsonl $H/.mon "$gpus" $pids & mon=$!; fi
sleep 2; v1=$(vram)
case $mode in
  prefill:*|long:*) for n in ${mode#*:}; do
      timeout -s KILL 5400 python3 $T/bench_cache.py http://127.0.0.1:$port m --prompt-file $D/prompt$n.json --require-cold --tokens 16 --repeats 1 --streams 1 --expected-prompt-tokens $n --output $H/cold_$n.json > $H/cold_$n.out 2>&1; x=$?
      timeout -s KILL 3600 python3 $R/plain.py http://127.0.0.1:$port $D/prompt$n.json 16 $H/plain_$n.json > /dev/null 2>&1; y=$?
      python3 $R/annotate.py $H/cold_$n.json $A ids_sha256=$(sha256sum $D/prompt$n.json | cut -c1-64) 2>/dev/null
      python3 $R/annotate.py $H/plain_$n.json $A 2>/dev/null
      note "  $n cold exit $x plain exit $y: $(python3 -c "
import json; d=json.load(open('$H/cold_$n.json')); p=json.load(open('$H/plain_$n.json'))['row']
print('qualified', d['qualified'], 'fixture==ids', d['fixture_sha256']==d['ids_sha256'], 'ttft', [round(r['ttft_s'],3) for r in d['rows']], 'cached', [r['cached_tokens'] for r in d['rows']], 'sha', d['rows'][0]['token_sha256'][:16], 'all equal', len({r['token_sha256'] for r in d['rows']})==1, '== draft false', d['rows'][0]['token_sha256']==p['token_sha256'], 'plain cached', p['cached_tokens'])" 2>&1 | tr '\n' ' ' | cut -c1-400)"
    done;;
  decode|decodeoff)
    timeout -s KILL 1800 python3 $T/bench_openai.py http://127.0.0.1:$port m --tokens 256 --reps 3 --temperatures 0 --label ${tree}_$mode --output $H/decode.json > $H/decode.out 2>&1; x=$?
    python3 $R/annotate.py $H/decode.json $A server_drafts=$([ $mode = decode ] && echo on || echo off) 2>/dev/null
    note "  bench_openai exit $x: $(grep '^{' $H/decode.out | python3 -c "
import json,sys
for l in sys.stdin: d=json.loads(l); print(d['prompt'], round(d['decode_tps_median'],2), d['decode_tps_all'], end='; ')")";;
  conc|conc8)
    lv=$([ $mode = conc ] && echo 1,4 || echo 1,4,8)
    timeout -s KILL 3600 python3 $T/bench_concurrent.py http://127.0.0.1:$port m --tokens 256 --reps 3 --temperatures 0 --levels $lv --alone --serial --output $H/concurrent.json > $H/concurrent.out 2>&1; x=$?
    python3 $R/annotate.py $H/concurrent.json $A 2>/dev/null
    note "  bench_concurrent levels $lv exit $x: $(grep '^{' $H/concurrent.out | python3 -c "
import json,sys
for l in sys.stdin: d=json.loads(l); print(d['prompt'], 'n', d['streams'], 'agg', round(d['aggregate_tps'],1), 'alone', '%d/%d/%d' % tuple(d['alone'][k] for k in ('equal','unequal','failed')), 'serial', '%d/%d/%d' % tuple(d['serial'][k] for k in ('equal','unequal','failed')), end='; ')" 2>&1 | cut -c1-600)";;
esac
v2=$(vram)
[ -n "$mon" ] && { rm -f $H/.mon; wait $mon; }
t2=$(date +%s.%N)
stop; sleep 2; v3=$(vram); t3=$(date +%s.%N)
for f in $W/serve_${label}_${tree}_$mtag*.log; do cp $f $H/$(basename $f | sed "s/serve_${label}_${tree}_$mtag/server/"); done
python3 - <<PY
import json
def j(s):
    try: return json.loads(s)
    except Exception: return None
json.dump({"source_commit": "$commit", "binary_sha256": "$bsha", "test_tool_sha256": "$tsha", "rig": "$rig", "hold": "$hold",
  "tree": "$tree", "mode": "$mode", "server_args": "serve <checkpoint> --name m --port <port> $sargs" + ("" if "$world" == "1" else " --tp $world --rank <r> --master 127.0.0.1 --master-port <port>"),
  "librccl_mapped": "$rccl".split(), "server_start_epoch": $t0, "server_healthy_epoch": $t1, "load_to_health_s": round($t1 - $t0, 2),
  "work_done_epoch": $t2, "server_stopped_epoch": $t3,
  "vram_before_start": j('''$v0'''), "vram_after_health": j('''$v1'''), "vram_after_work": j('''$v2'''), "vram_after_stop": j('''$v3''')},
  open("$H/hold.json", "w"), indent=1)
PY
note "hold $hold $tree $mode done, load ${t0}..${t1}"
touch $H/.done
