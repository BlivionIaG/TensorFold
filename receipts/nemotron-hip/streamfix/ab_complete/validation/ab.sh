#!/bin/bash
# ab.sh LABEL DEVS PORT [tp1]; requires ab_validate.py beside this script.
# Each condition runs c46,d46,d46,c46. Stops on failed startup, command, or evidence.
set -o pipefail
[[ $# == 3 || $# == 4 ]] || { echo "usage: $0 LABEL DEVS PORT [tp1]" >&2; exit 2; }
label=$1; devs=$2; port=$3; tp1=${4:-}
[[ $label =~ ^[a-zA-Z0-9_-]+$ && $devs =~ ^[0-9]+(,[0-9]+)*$ && $port =~ ^[0-9]+$ ]] || exit 2
[[ -z $tp1 || $tp1 == tp1 ]] || exit 2
# libn consumes devs/tp1 at source time and chooses out-gfx1100 or out-gfx1030.
. "$HOME/nh/libn.sh"
# The retained baseline is the previously published NaN-fix top.
dir() { case $1 in c46) echo "$HOME/nh/d46";; d46) echo "$HOME/nh/sf46";; *) return 2;; esac; }
L="$HOME/hcx/streamfix/logs/ab_$label.log"
O="$HOME/hcx/streamfix/logs/ab/${label}_$(date +%Y%m%dT%H%M%S)_$$"
V="$(cd -- "$(dirname -- "$0")" && pwd)/ab_validate.py"
mkdir -p "$O" "$W" || exit 1
: > "$L"
note() { echo "$(date +%T) $*" | tee -a "$L"; }
fail() { note "AB_FAILED $*"; exit 1; }
trap 'stop' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
[[ -f $V ]] || fail "validator missing: $V"
for tree in c46 d46; do
  bin="$(dir "$tree")/out-$arch/native-hip/bin/tensorfold-native"
  [[ -x $bin ]] || fail "missing native binary: $bin"
  binary_sha=$(sha256sum "$bin" | cut -d' ' -f1) || fail "hashing $bin"
  note "tree=$tree arch=$arch world=$world devs=$devs commit=$(git -C "$(dir "$tree")" rev-parse HEAD) binary=$bin binary_sha256=$binary_sha"
done
for n in 8192 32768; do
  [[ -r $HOME/nem-work/ids/prompt$n.json ]] || fail "missing prompt$n.json"
  note "prompt$n.json sha256=$(sha256sum "$HOME/nem-work/ids/prompt$n.json" | cut -d' ' -f1)"
done
for pair in "decode/bench_openai.py:$HOME/nem-shared/tools/bench_openai.py" \
            "decode/bench_evidence.py:$HOME/nem-shared/tools/bench_evidence.py" \
            "cold/bench_cache.py:$HOME/nem1012/tools/bench_cache.py" \
            "cold/bench_concurrent.py:$HOME/nem1012/tools/bench_concurrent.py" \
            "cold/bench_evidence.py:$HOME/nem1012/tools/bench_evidence.py" \
            "cold/bench_auth.py:$HOME/nem1012/tools/bench_auth.py"; do
  client=${pair%%:*}; source=${pair#*:}
  client_sha=$(sha256sum "$source" | cut -d' ' -f1) || fail "hashing client $client"
  note "client $client sha256=$client_sha"
done
for harness in ab.sh ab_validate.py capture_identity.py; do
  harness_sha=$(sha256sum "$(dirname -- "$V")/$harness" | cut -d' ' -f1) || fail "hashing $harness"
  note "harness $harness sha256=$harness_sha"
done
[[ $port -ge 1024 && $port -le 64519 ]] || fail "port range must leave room for 16 servers and TP master ports"
note "output=$O"
for condition in decode_on decode_off cold_8192 cold_32768; do
  files=()
  case $condition in decode_*) mode=decode; n=0;; cold_*) mode=cold; n=${condition#cold_};; esac
  for tree in c46 d46 d46 c46; do
    port=$((port + 1))
    file="$O/${condition}_${tree}_${port}.json"
    extra=()
    if [[ $mode == decode ]]; then
      extra=(--context 65536 --parallel 4)
      [[ $condition != decode_off ]] || extra+=(--no-drafts)
    else
      extra=(--context 65536 --parallel 1 --prompt-cache-gib 0)
    fi
    # Refuse to benchmark an unrelated server already listening at the endpoint.
    curl -sf "http://localhost:$port/health" >/dev/null 2>&1 && fail "port $port already serves health"
    serve "$tree" "ab$port" "$port" "${extra[@]}" || fail "$condition $tree startup"
    python3 "$(dirname -- "$V")/capture_identity.py" "$file.identity.json" $pids || fail "$condition $tree runtime identity"
    if [[ $mode == decode ]]; then
      timeout -s KILL 900 "$P" "$HOME/nem-shared/tools/bench_openai.py" "http://localhost:$port" m --tokens 256 --reps 3 --temperatures 0 --output "$file" > "$file.stdout" 2>&1
    else
      timeout -s KILL 1800 "$P" "$HOME/nem1012/tools/bench_cache.py" "http://localhost:$port" m --prompt-file "$HOME/nem-work/ids/prompt$n.json" --expected-prompt-tokens "$n" --require-cold --tokens 16 --repeats 1 --streams 1 --output "$file" > "$file.stdout" 2>&1
    fi
    status=$?
    stop
    [[ $status == 0 ]] || fail "$condition $tree benchmark status=$status (see $file.stdout)"
    note "$condition $tree file=$file"
    "$P" "$V" "$mode" "$n" "$file" >> "$L" 2>&1 || fail "$condition $tree invalid receipt"
    files+=("$file")
  done
  "$P" "$V" "$mode" "$n" "${files[@]}" >> "$L" 2>&1 || fail "$condition token equality validation"
  note "PASS $condition token_sha equal across c46,d46,d46,c46"
done
note "AB_DONE"
