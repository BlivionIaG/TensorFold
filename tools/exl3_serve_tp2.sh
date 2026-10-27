#!/bin/bash
# Launch one TP=2 rank of the EXL3 Qwen3.8-Flash-Next server on two RTX 5060 Ti (16 GB), drafts on at every stream count.
# Usage: tools/exl3_serve_tp2.sh MODEL_DIR RANK PORT MASTER_PORT [extra serve args]; start rank 1 first, then rank 0.
# The MTP head is loaded by default (rank 0 holds it); `--no-drafts` in the extra arguments serves without it.
# Expert lines a layer are the most that fit beside the decode graphs, the device planner and a prompt cache of
# 0.6 GiB or more: 72 for one stream, 52 for two or more (measured on a 16 GB card, 32768-token context).
# One rank (no --tp) on such a card: 64 lines, TF_FLASHNEXT_EXPERT_RESERVE_MB=256 TENSORFOLD_MEMORY_RESERVE_GIB=0.5.
# Environment: LINES (expert lines a layer), RESERVE_MB (planner reserve), MEMORY_RESERVE_GIB, CONTEXT,
# PARALLEL (streams, default 2), TENSORFOLD_BIN.
set -euo pipefail
model=$1 rank=$2 port=$3 master=$4
shift 4
bin=${TENSORFOLD_BIN:-./zig-out/native/bin/tensorfold-native}
export TENSORFOLD_CHECKPOINT_SHA256=${TENSORFOLD_CHECKPOINT_SHA256:-$(cat "$model/model.safetensors.index.json" "$model/config.json" | sha256sum | cut -c1-64)}
export TF_FLASHNEXT_EXL3_GRAPH=1 TF_FLASHNEXT_EXL3_GRAPH_PREFILL=1
export TF_FLASHNEXT_EXL3_GRAPH_ASYNC=${TF_FLASHNEXT_EXL3_GRAPH_ASYNC:-0}
if [ "${PARALLEL:-2}" = 1 ]; then
  # one user: a 0.5 GiB device reserve (the supported minimum) and a 256 MB planner reserve
  export TF_FLASHNEXT_EXPERT_LINES=${LINES:-72} TF_FLASHNEXT_EXPERT_RESERVE_MB=${RESERVE_MB:-256} TENSORFOLD_MEMORY_RESERVE_GIB=${MEMORY_RESERVE_GIB:-0.5}
else
  export TF_FLASHNEXT_EXPERT_LINES=${LINES:-52} TF_FLASHNEXT_EXPERT_RESERVE_MB=${RESERVE_MB:-512}
  [ -n "${MEMORY_RESERVE_GIB:-}" ] && export TENSORFOLD_MEMORY_RESERVE_GIB=$MEMORY_RESERVE_GIB
fi
CUDA_VISIBLE_DEVICES=$rank exec "$bin" serve "$model" --backend cuda --context "${CONTEXT:-32768}" --tp 2 --rank "$rank" \
  --master 127.0.0.1 --master-port "$master" --port "$port" --no-update-check --parallel "${PARALLEL:-2}" --prompt-cache-gib 0 "$@"
