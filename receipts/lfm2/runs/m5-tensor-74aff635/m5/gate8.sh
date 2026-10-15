set -u
T=~/tf-lfm2-test; M=$T/models/LFM2.5-24B-A2B-MLX-4bit; O=$T/final8/runs; mkdir -p $O
cd $T/final8/src && PATH=$T/tools-bin:$PATH ~/.local/zig-0.17.0/zig build native tf-lfm2-exact tf-tile-check -Dcpu=apple_m1 -Dversion=1.0.0 -Dmetal-std=metal4.0 --cache-dir $T/final7/zc --prefix $T/final8/out > $O/build.log 2>&1; echo "build exit $?" >> $O/progress.log
$T/final8/out/bin/tf-tile-check > $O/tile_check.txt 2>&1; echo "tile exit $?" >> $O/progress.log
$T/final8/out/bin/tf-lfm2-exact $M > $O/exact_4bit.txt 2>&1; echo "exact exit $?" >> $O/progress.log
bash $T/resume8.sh; echo "resume done" >> $O/progress.log
echo GATE8_DONE >> $O/progress.log
