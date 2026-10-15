set -u
T=~/tf-lfm2-test; M=$T/models/LFM2.5-24B-A2B-MLX-4bit; O=$T/final8/runs; PY=~/.local/bin/python3.12; B=$T/final8/out/native/bin/tensorfold-native
while ! grep -q GATE8_DONE $O/progress.log 2>/dev/null; do sleep 10; done
cd $T/final8/src && PATH=$T/tools-bin:$PATH ~/.local/zig-0.17.0/zig build tf-lfm2-fp64 -Dcpu=apple_m1 -Dversion=1.0.0 -Dmetal-std=metal4.0 --cache-dir $T/final7/zc --prefix $T/final8/out > $O/build_fp64.log 2>&1
$T/final8/out/bin/tf-lfm2-fp64 $M > $O/fp64.txt 2>&1; echo "fp64 exit $?" >> $O/progress.log
up(){ for i in $(seq 1 180); do sleep 2; curl -s localhost:8091/v1/models >/dev/null 2>&1 && return 0; done; return 1; }
$B serve $M --port 8091 --context 131072 --prompt-cache-gib 0 --no-update-check > $O/serve_conc.log 2>&1 & P=$!; up
$PY $T/tools/bench_openai.py http://127.0.0.1:8091 m --tokens 256 --reps 1 --temperatures 0 --label p7 --output $O/decode.json > /dev/null 2>&1
$PY $T/tools/bench_openai.py http://127.0.0.1:8091 m --tokens 256 --reps 1 --temperatures 0 --label p7nd --no-drafts --output $O/decode_nodraft.json > $O/nodraft.txt 2>&1
$PY $T/tools/bench_concurrent.py http://127.0.0.1:8091 m --tokens 256 --reps 1 --temperatures 0 --levels 4 --alone --serial --output $O/conc.json > $O/conc.txt 2>&1
kill $P; while kill -0 $P 2>/dev/null; do sleep 1; done
echo AFTER8_DONE >> $O/progress.log
