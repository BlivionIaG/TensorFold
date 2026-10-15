set -u
T=~/tf-lfm2-test; M=$T/models/LFM2.5-24B-A2B-MLX-4bit; O=$T/bench-final2; PY=~/.local/bin/python3.12; MX=$T/mlxenv/bin; D=$T/tools/CHANGELOG.md; K=$T/tokens; mkdir -p $O
ZIG=~/.local/zig-0.17.0/zig; ZF="-Dcpu=apple_m1 -Dversion=1.0.0 -Dmetal-std=metal4.0 --cache-dir $T/final7/zc"
NB=$T/bin/tensorfold-native; XB=$T/bin/tf-lfm2-bench
NO=$T/final9/out/native/bin/tensorfold-native; XO=$T/final9/out/bin/tf-lfm2-bench
NF=$T/final9off/out/native/bin/tensorfold-native; XF=$T/final9off/out/bin/tf-lfm2-bench
BC=$T/final9/src/tools/bench_cache.py
log(){ echo "$(date +%H:%M:%S) $*" >> $O/progress.log; }
busy(){ pgrep -x tensorfold-native >/dev/null || pgrep -f "tf-lfm2-(bench|exact)" >/dev/null || pgrep -f mlx_lm >/dev/null; }
lock(){ s=0; until mkdir $T/.model_lock 2>/dev/null; do
  if busy; then s=0; else s=$((s+20)); if [ $s -ge 120 ]; then rmdir $T/.model_lock 2>/dev/null && log "stale lock removed before $1"; s=0; fi; fi
  sleep 20; done
 if busy; then log "BUSY before $1: $(pgrep -lf 'tensorfold-native|tf-lfm2-|mlx_lm' | tr '\n' ';')"; while busy; do sleep 5; done; fi; }
unlock(){ rmdir $T/.model_lock; }
up(){ for i in $(seq 1 180); do sleep 2; curl -s localhost:8091/v1/models >/dev/null 2>&1 && return 0; done; return 1; }
stop(){ kill $1; while kill -0 $1 2>/dev/null; do sleep 1; done; sleep 3; }
log START
# builds
cd $T/final9/src && PATH=$T/tools-bin:$PATH $ZIG build tf-lfm2-bench $ZF --prefix $T/final9/out > $O/build_on.log 2>&1; log "build on bench exit $?"
[ -e $T/final9off ] && { log "final9off exists"; exit 1; }; mkdir -p $T/final9off; cp -R $T/final9/src $T/final9off/src
KF=$T/final9off/src/zig/src/families/lfm2/kernels.zig
sed -i '' 's|    out.nax = if (device.tensorUnits()) try naxLoad(device) else null;|    out.nax = null; // test build: tensor units off|; s|    if (device.tensorUnits()) try attnNaxLoad(device, \&out);|    if (false) try attnNaxLoad(device, \&out);|' $KF
grep -n "test build: tensor units off\|if (false) try attnNaxLoad" $KF > $O/off_patch.txt; log "off patch lines $(wc -l < $O/off_patch.txt)"
cd $T/final9off/src && PATH=$T/tools-bin:$PATH $ZIG build native tf-lfm2-bench $ZF --prefix $T/final9off/out > $O/build_off.log 2>&1; log "build off exit $?"
{ sw_vers; sysctl -n machdep.cpu.brand_string hw.memsize; system_profiler SPDisplaysDataType 2>/dev/null | grep -i "cores"; $MX/python -c "import mlx_lm,mlx.core as m;print(\"mlx\",m.__version__,\"mlx-lm\",mlx_lm.__version__)"; echo "before=$NB (5a6ee034) on=$NO (fd2f8e27) off=$NF (fd2f8e27, tensor units off)"; shasum -a 256 $NB $NO $NF $XB $XO $XF; } > $O/versions.txt 2>&1
for f in $NO $NF $XO $XF; do [ -x $f ] || { log "MISSING $f"; log ABORT; exit 1; }; done
nat(){ case $1 in before) echo $NB;; on) echo $NO;; off) echo $NF;; esac; }
bch(){ case $1 in before) echo $XB;; on) echo $XO;; off) echo $XF;; esac; }
ses(){ n=$1; r=$2; long=$3
 lock ses_${n}_r$r
 if [ $n = mlx ]; then $MX/mlx_lm.server --model $M --port 8091 > $O/serve_${n}_r$r.log 2>&1 & P=$!; up || log "UP FAIL $n r$r"; MN=$M
 else B=$(nat $n); $B serve $M --port 8091 --context 131072 --prompt-cache-gib 0 --no-update-check > $O/serve_${n}_r$r.log 2>&1 & P=$!; up || log "UP FAIL $n r$r"; MN=m
  L="$K/prompt_2048.json:2 $K/prompt_8192.json:2 $K/prompt_32768.json:1 $K/prompt_65536.json:1"; [ $long = y ] && L="$L $K/prompt_131000.json:1"
  $PY $T/tools/prefill_tokens.py http://127.0.0.1:8091 m $O/prefill_${n}_r$r.json $L > $O/prefill_${n}_r$r.txt 2>&1; log "$n r$r prefill exit $?"
  $PY $BC http://127.0.0.1:8091 m --tokens 64 --repeats 1 --streams 1 --prompt-file $K/prompt_32768.json --expected-prompt-tokens 32768 --require-cold --output $O/cache_${n}_r$r.json > $O/cache_${n}_r$r.txt 2>&1; log "$n r$r cache exit $?"; fi
 $PY $T/tools/bench_openai.py http://127.0.0.1:8091 $MN --tokens 256 --reps 3 --temperatures 0 --label ${n}_r$r --output $O/decode_${n}_r$r.json > $O/decode_${n}_r$r.txt 2>&1; log "$n r$r decode exit $?"
 $PY $T/tools/bench_concurrent.py http://127.0.0.1:8091 $MN --tokens 256 --reps 3 --temperatures 0 --levels 1,4 --alone --serial --output $O/conc_${n}_r$r.json > $O/conc_${n}_r$r.txt 2>&1; log "$n r$r conc exit $?"
 vmmap -summary $P 2>/dev/null | grep -i "Physical footprint" > $O/footprint_${n}_r$r.txt
 stop $P; unlock; log "$n r$r done"; }
ses mlx 1 n; ses before 1 y; ses on 1 y; ses off 1 y; ses off 2 n; ses on 2 n; ses before 2 n; ses mlx 2 n
log SERVED_DONE
# two-turn resume: on with cache on, mlx once
lock tt_on; $NO serve $M --port 8091 --context 131072 --no-update-check > $O/serve_tt_on.log 2>&1 & P=$!; up || log "UP FAIL tt on"
$PY $T/tools/two_turn.py http://127.0.0.1:8091 m $O/two_turn_on.json $D > $O/two_turn_on.txt 2>&1; log "two_turn on exit $?"; stop $P; unlock
lock tt_mlx; $MX/mlx_lm.server --model $M --port 8091 > $O/serve_tt_mlx.log 2>&1 & P=$!; up || log "UP FAIL tt mlx"
$PY $T/tools/two_turn.py http://127.0.0.1:8091 $M $O/two_turn_mlx.json $D > $O/two_turn_mlx.txt 2>&1; log "two_turn mlx exit $?"; stop $P; unlock
ip(){ w=$1; r=$2
 for L in 2048:2048:128 8192:8192:128 32768:8192:64 65536:8192:64; do IFS=: read n c t <<< "$L"
  lock ip_${w}_${n}_r$r
  if [ $w = mlx ]; then $MX/python $T/tools/mlx_cold.py $M $D $n $t > $O/ip_mlx_${n}_r$r.txt 2>&1
  else $(bch $w) $M $n $c $t > $O/ip_${w}_${n}_r$r.txt 2>&1; fi
  e=$?; unlock; log "ip $w $n r$r exit $e"; done; }
ip before 1; ip on 1; ip off 1; ip mlx 1; ip mlx 2; ip off 2; ip on 2; ip before 2
for w in before on off; do lock ip131_$w; /usr/bin/time -l $(bch $w) $M 131000 8192 64 > $O/ip131_$w.txt 2>&1; e=$?; unlock; log "ip131 $w exit $e"; done
lock ip131_mlx; /usr/bin/time -l $MX/python $T/tools/mlx_cold.py $M $D 131000 64 > $O/ip131_mlx.txt 2>&1; e=$?; unlock; log "ip131 mlx exit $e"
log ALL_DONE
