# helpers for any world size: every process under setsid + timeout -s KILL, stopped by killing its whole process group
W=$WORK/logs; M=$CHECKPOINT; I=$REFS/ids
export LD_LIBRARY_PATH=$RCCL_RDNA_LIB PYTHONWARNINGS=ignore
P=python3
arch=gfx1030; case $devs in 0*|1*) arch=gfx1100;; esac
d0=${devs%%,*}; world=$(echo $devs | awk -F, '{print NF}'); [ -n "$tp1" ] && world=1
dir() { case $1 in main1012|nem1012|nem-rounds) echo $HOME/$1;; *) echo $WORK/$1;; esac; }
run() { local d=$1; shift; setsid env -u ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=$d timeout -s KILL ${LIMIT:-2400} "$@"; }
up() { local i; for i in $(seq 1 300); do sleep 2; curl -sf localhost:$1/health > /dev/null 2>&1 && return 0; kill -0 $2 2>/dev/null || return 1; done; return 1; }
rcclseen() { local p; for p in $(pgrep -g $1); do grep -m1 -o '/[^ ]*librccl[^ ]*' /proc/$p/maps 2>/dev/null; done | sort -u | tr '\n' ' '; }
# serve TREE TAG PORT extra...: pids (session leaders) in $pids; rank 0 last
serve() { local tree=$1 tag=$2 p=$3; shift 3
  local bin=$(dir $tree)/out-$arch/native-hip/bin/tensorfold-native r
  pids=""
  if [ $world = 1 ]; then run $d0 $bin serve $M --name m --port $p "$@" > $W/serve_${label}_${tree}_$tag.log 2>&1 < /dev/null & pids=$!; up $p $pids; return; fi
  for r in $(seq 1 $((world - 1))); do
    run $devs $bin serve $M --tp $world --rank $r --master 127.0.0.1 --master-port $((p + 1000)) "$@" > $W/serve_${label}_${tree}_${tag}_r$r.log 2>&1 < /dev/null & pids="$pids $!"
  done
  sleep 1
  run $devs $bin serve $M --tp $world --rank 0 --master 127.0.0.1 --master-port $((p + 1000)) --name m --port $p "$@" > $W/serve_${label}_${tree}_$tag.log 2>&1 < /dev/null & local p0=$!
  pids="$p0 $pids"; up $p $p0; }
stop() { for p in $pids; do kill -KILL -- -$p 2>/dev/null; done; wait $pids 2>/dev/null; sleep 3
  for p in $pids; do pgrep -g $p > /dev/null && echo "WARN group $p survived" >> $L; done; pids=; }
# tool TREE NAME args...: the test tool on every rank, rank 0's output kept
tool() { local tree=$1 name=$2; shift 2; local bin=$(dir $tree)/out-$arch/bin/tf-nemotron-hip-test x r tp=""
  if [ $world = 1 ]; then LIMIT=1800 run $d0 $bin "$@" > $W/tool_${label}_$name.log 2>&1; x=$?
  else
    for r in $(seq 1 $((world - 1))); do LIMIT=1800 run $devs $bin "$@" --tp $world --rank $r --master 127.0.0.1 --port $port > $W/tool_${label}_$name.r$r.log 2>&1 & tp="$tp $!"; done
    LIMIT=1800 run $devs $bin "$@" --tp $world --rank 0 --master 127.0.0.1 --port $port > $W/tool_${label}_$name.log 2>&1; x=$?
    sleep 2; for r in $tp; do kill -KILL -- -$r 2>/dev/null; done; wait $tp 2>/dev/null
  fi
  note "$name exit $x: $(grep -h "^window\|^snapshots\|^rounds\|^replay:\|^teacher\|^decode\|FAIL\|error\|REFUSE" $W/tool_${label}_$name.log | tail -3 | cut -c1-220 | tr '\n' '|')"
}
trap "stop" EXIT
