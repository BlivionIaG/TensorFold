set -u
T=~/tf-lfm2-test; M=$T/models/LFM2.5-24B-A2B-MLX-4bit; O=$T/final7/runs; PY=~/.local/bin/python3.12; D=$T/tools/CHANGELOG.md; B7=$T/final7/out/native/bin/tensorfold-native
cd $T/final7/src && PATH=$T/tools-bin:$PATH ~/.local/zig-0.17.0/zig build tf-lfm2-exact -Dcpu=apple_m1 -Dversion=1.0.0 -Dmetal-std=metal4.0 --cache-dir $T/final7/zc --prefix $T/final7/out > $T/final7/build_exact.log 2>&1; echo "exact build exit $?" >> $O/progress.log
$T/final7/out/bin/tf-lfm2-exact $M > $O/exact_4bit_v2.txt 2>&1; echo "exact v2 exit $?" >> $O/progress.log
up(){ for i in $(seq 1 180); do sleep 2; curl -s localhost:8091/v1/models >/dev/null 2>&1 && return 0; done; return 1; }
stop(){ kill $1; while kill -0 $1 2>/dev/null; do sleep 1; done; sleep 3; }
$B7 serve $M --port 8091 --context 131072 --no-update-check > $O/serve_t7_cache.log 2>&1 & P=$!; up
$PY $T/tools/two_turn.py http://127.0.0.1:8091 m $O/two_turn_cache.json $D > /dev/null 2>&1; stop $P
$B7 serve $M --port 8091 --context 131072 --prompt-cache-gib 0 --no-update-check > $O/serve_t7_nocache.log 2>&1 & P=$!; up
$PY $T/tools/two_turn.py http://127.0.0.1:8091 m $O/two_turn_nocache.json $D > /dev/null 2>&1; stop $P
echo AFTER_DONE >> $O/progress.log
