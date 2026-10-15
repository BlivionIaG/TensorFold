#!/bin/zsh
# usage: win.sh tag... ; prints window ms line per tag
cd ~/tf-lfm2-test/f4
M=~/tf-lfm2-test/models/LFM2.5-24B-A2B-MLX-4bit
for t in "$@"; do
  until mkdir ~/tf-lfm2-test/.model_lock 2>/dev/null; do sleep 20; done
  bin=./out_$t/native/bin/tensorfold-native; [ $t = m5base ] && bin=~/tf-lfm2-test/bin/tensorfold-native
  $bin serve $M --port 8091 --context 131072 --prompt-cache-gib 0 --no-update-check > serve_$t.log 2>&1 &
  pid=$!
  for i in {1..240}; do grep -q "window ms by width" serve_$t.log && break; kill -0 $pid 2>/dev/null || break; sleep 1; done
  kill $pid; wait $pid 2>/dev/null
  rmdir ~/tf-lfm2-test/.model_lock
  echo "$t: $(grep "window ms by width" serve_$t.log)" | tee -a win.log
done
