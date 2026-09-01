# XR-0 GenieSim SFT training environment.
# Usage:  source env.sh   (then: RESOURCE_GPU=8 bash scripts/train.sh data=g2op_if model=XR0 ...)
# Resolves the repo root from this file, activates .venv, and exports what training needs.
_XR0_ENV_SRC="${BASH_SOURCE[0]:-$0}"
export XR0_ROOT="$(cd "$(dirname "$_XR0_ENV_SRC")" && pwd)"
source "$XR0_ROOT/.venv/bin/activate"
export PYTHONPATH="$XR0_ROOT"
export CUDA_HOME=/usr/local/cuda
export TOKENIZERS_PARALLELISM=false
# Training box has no public internet — force transformers/hf_hub to use the local HF cache
# instead of HEAD-requesting huggingface.co (Errno 101 Network is unreachable) on every load.
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}  # local HF cache; override HF_HOME to point at a shared one
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline  # no public net -> log wandb locally (sync later with: wandb sync <dir>)
echo "[xr0] venv: $(python -V 2>&1)  root: $XR0_ROOT"
echo "[xr0] launch: RESOURCE_GPU=8 bash scripts/train.sh data=g2op_if model=XR0 \\"
echo "               model.params.pretrained=pretrained_ckpt/xr0_pretrained.pt \\"
echo "               trainer.project=xr0_geniesim trainer.exp_name=g2op_if"
