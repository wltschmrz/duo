#!/bin/bash
set -euo pipefail

#SBATCH -J lm1b_mdlm                  # Job name
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=32000                   # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=kuleshov          # Request partition
#SBATCH --constraint="b200&(gpu-mid|gpu-high)"
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon pre-emption

# To enable preemption re-loading, set `hydra.run.dir` or
# `checkpointing.save_dir` explicitly.

# gpu related settings
GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0}}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
N_GPU="${N_GPU:-$(awk -F',' '{print NF}' <<< "${GPU_IDS}")}"
NUM_WORKERS="${NUM_WORKERS:-64}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# batch size settings
PER_GPU_BATCH="${PER_GPU_BATCH:-128}"
TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-512}"
RESUME_CKPT_PATH="${RESUME_CKPT_PATH:-}"

# batch size validation
if (( TARGET_GLOBAL_BATCH % (PER_GPU_BATCH * N_GPU) != 0 )); then
  echo "ERROR: TARGET_GLOBAL_BATCH must be divisible by PER_GPU_BATCH*N_GPU."
  exit 1
fi
ACCUM=$(( TARGET_GLOBAL_BATCH / (PER_GPU_BATCH * N_GPU) ))

# wandb settings
WANDB_NAME="${WANDB_NAME:-sfldd-lm1b-wrap-small-100k-H200x4}"
SEED="${SEED:-1}"
WANDB_RUN_ID="${WANDB_RUN_ID:-${WANDB_NAME}_${SEED}}"

# optional args
EXTRA_ARGS=()
if [[ -n "${RESUME_CKPT_PATH}" ]]; then
  RESUME_SAVE_DIR="$(dirname "$(dirname "${RESUME_CKPT_PATH}")")"
  # PyTorch 2.6+: torch.load defaults to weights_only=True.
  # Lightning resume checkpoints may contain OmegaConf objects.
  export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
  EXTRA_ARGS+=("checkpointing.resume_from_ckpt=true")
  EXTRA_ARGS+=("checkpointing.resume_ckpt_path=${RESUME_CKPT_PATH}")
  EXTRA_ARGS+=("checkpointing.save_dir=${RESUME_SAVE_DIR}")
  EXTRA_ARGS+=("wandb.id=${WANDB_RUN_ID}")
  EXTRA_ARGS+=("+wandb.resume=must")
fi


python -u -m main \
  loader.batch_size="${PER_GPU_BATCH}" \
  loader.eval_batch_size="${PER_GPU_BATCH}" \
  loader.global_batch_size="${TARGET_GLOBAL_BATCH}" \
  loader.eval_global_batch_size="${TARGET_GLOBAL_BATCH}" \
  loader.num_workers="${NUM_WORKERS}" \
  trainer.devices="${N_GPU}" \
  trainer.accumulate_grad_batches="${ACCUM}" \
  seed="${SEED}" \
  data=lm1b-wrap \
  wandb.name="${WANDB_NAME}" \
  model=small \
  algo=sfldd \
  model.length=128 \
  trainer.max_steps=100000 \
  trainer.val_check_interval=500 \
  trainer.precision=bf16-mixed \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=2000 \
  "${EXTRA_ARGS[@]}"
