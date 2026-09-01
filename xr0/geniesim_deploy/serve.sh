#!/usr/bin/env bash
# Start the two-tier XR-0 inference stack for GenieSim in a tmux session:
#   window 0  xr0-server   :10086  GPU  — mibot/server/deploy.py (DiT rectified-flow)
#   window 1  xr0-adapter  :8007   CPU  — geniesim_deploy/xr0_corobot_adapter.py (corobot <-> pickle/TCP)
#
# Usage:  bash geniesim_deploy/serve.sh <model_dir> [gpu_id] [model_port] [adapter_port] [ckpt]
#   e.g.  bash geniesim_deploy/serve.sh train/xr0_geniesim/g2op_if 0 10086 8007
#   pick a step: bash geniesim_deploy/serve.sh train/xr0_geniesim/g2op_if 0 10086 8007 'epoch=0-step=20000.ckpt'
# [ckpt] is a checkpoint dir NAME under <model_dir> (default last.ckpt) or an absolute path; config.py
# (norm + active_parts) is always read from <model_dir>, so keep <model_dir> = the training output dir.
#
# <model_dir> is the trained ckpt dir that contains config.py + last.ckpt/ (the SFT output), i.e.
# {trainer.default_root_dir}/{trainer.project}/{trainer.exp_name} from the train command — with the
# README's example (default_root_dir=train/, project=xr0_geniesim, exp_name=g2op_if) that is
# train/xr0_geniesim/g2op_if. Its config.py carries active_parts + mean/std, so deploy.py masks the
# same joint channels the model was trained with. Point GenieSim at <host>:<adapter_port> with
# --model_arc corobot.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${1:?usage: serve.sh <model_dir> [gpu_id] [model_port] [adapter_port] [ckpt]}"
GPU_ID="${2:-0}"
MODEL_PORT="${3:-10086}"
ADAPTER_PORT="${4:-8007}"
CKPT="${5:-last.ckpt}"  # ckpt dir name under <model_dir> (e.g. 'epoch=0-step=20000.ckpt') or abs path; default latest
ATTN_IMPL="${XR0_ATTN_IMPL:-flash_attention_2}"  # sdpa for 4090/sm_89 inference
WAIST="${XR0_WAIST:-}"  # set to 1 for a model trained with the waist joint active (waist in active_parts); off otherwise
# tmux session name. Override via XR0_SESSION to run TWO servers side by side (also give each its own
# model_port + adapter_port, and a GPU with enough memory). Default keeps the historical single-server name.
SESSION="${XR0_SESSION:-xr0_geniesim}"
PY="$REPO/.venv/bin/python"
HF_HOME_DIR="${HF_HOME:-$HOME/.cache/huggingface}"  # local HF cache; override HF_HOME to point at a shared one

[[ -x "$PY" ]] || { echo "venv python not found at $PY"; exit 1; }
[[ -d "$MODEL_DIR" ]] || { echo "model dir not found: $MODEL_DIR"; exit 1; }

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n xr0-server -c "$REPO"

# window 0: GPU model server
tmux send-keys -t "$SESSION:xr0-server" \
  "export TOKENIZERS_PARALLELISM=false PYTHONPATH=$REPO HF_HOME=$HF_HOME_DIR HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 XR0_ATTN_IMPL=$ATTN_IMPL XR0_WAIST=$WAIST; \
   CUDA_VISIBLE_DEVICES=$GPU_ID $PY mibot/server/deploy.py --model '$MODEL_DIR' --ckpt '$CKPT' --port $MODEL_PORT" Enter

# window 1: corobot adapter (no GPU) — wait for the model port to open, then start
tmux new-window -d -t "$SESSION" -n xr0-adapter -c "$REPO"
tmux send-keys -t "$SESSION:xr0-adapter" \
  "export TOKENIZERS_PARALLELISM=false PYTHONPATH=$REPO HF_HOME=$HF_HOME_DIR HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 XR0_ATTN_IMPL=$ATTN_IMPL XR0_WAIST=$WAIST; \
   until (echo > /dev/tcp/127.0.0.1/$MODEL_PORT) 2>/dev/null; do echo 'waiting for xr0 server :$MODEL_PORT...'; sleep 3; done; \
   $PY geniesim_deploy/xr0_corobot_adapter.py --host 0.0.0.0 --port $ADAPTER_PORT --upstream-host 127.0.0.1 --upstream-port $MODEL_PORT" Enter

echo "started tmux session '$SESSION':"
echo "  window xr0-server   -> :$MODEL_PORT (GPU $GPU_ID)"
echo "  window xr0-adapter  -> :$ADAPTER_PORT (corobot; GenieSim --infer_host <host>:$ADAPTER_PORT)"
echo "attach: tmux attach -t $SESSION    stop: tmux kill-session -t $SESSION"
