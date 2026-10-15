set -u
T=~/tf-lfm2-test; M=$T/models/LFM2.5-24B-A2B-MLX-4bit; O=$T/final8/runs; mkdir -p $O; PY=~/.local/bin/python3.12; D=$T/tools/CHANGELOG.md; B7=$T/final8/out/native/bin/tensorfold-native
up(){ for i in $(seq 1 180); do sleep 2; curl -s localhost:8091/v1/models >/dev/null 2>&1 && return 0; done; return 1; }
stop(){ kill $1; while kill -0 $1 2>/dev/null; do sleep 1; done; sleep 3; }
$B7 serve $M --port 8091 --context 131072 --no-update-check > $O/serve_cache.log 2>&1 & P=$!; up || echo "serve failed" >> $O/progress.log
$PY $T/tools/two_turn.py http://127.0.0.1:8091 m $O/two_turn_cache.json $D > $O/two_turn.out 2>&1; stop $P
$B7 serve $M --port 8091 --context 131072 --no-update-check > $O/serve_fresh.log 2>&1 & P=$!; up
$PY $T/fresh2.py http://127.0.0.1:8091 m $O/two_turn_cache.json $O/fresh_on.json m2,m2b; stop $P
$B7 serve $M --port 8091 --context 131072 --prompt-cache-gib 0 --no-update-check > $O/serve_off.log 2>&1 & P=$!; up
$PY $T/fresh2.py http://127.0.0.1:8091 m $O/two_turn_cache.json $O/fresh_off.json m1,m2; stop $P
