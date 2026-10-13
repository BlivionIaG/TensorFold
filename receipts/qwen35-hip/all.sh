#!/bin/bash
# all.sh: the five configs one after another (one model process set on the machine at a time)
cd ~/rcpt655/receipt
./run.sh w7800-tp1 0 gfx1100
./run.sh v620-tp1 2 gfx1030
./run.sh w7800-tp2 0,1 gfx1100
./run.sh v620-tp2 2,3 gfx1030
./run.sh v620-tp4 2,3,4,5 gfx1030
echo ALL-DONE
