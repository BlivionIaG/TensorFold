#!/bin/bash
# HF fp32 CPU reference logits over the longer teacher sequences, one at a time
for n in chat code; do
  HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= nice -n 10 timeout -s KILL 10800 $P $HOME/nem_ref.py logits $M $REFS/${n}_long.ids.bin $REFS/ref_${n}_long.npy > $REFS/ref_${n}_long.log 2>&1
  echo "$n exit $?" >> $REFS/ref_long.done
done
echo REF_DONE >> $REFS/ref_long.done
