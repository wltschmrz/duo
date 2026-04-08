#!/bin/bash
set -euo pipefail

#SBATCH -J duo-lm1b                   # Job name
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=64000                   # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=kuleshov          # Request partition
#SBATCH --constraint="[a5000|a6000|a100|3090]"
#SBATCH --constraint="gpu-mid|gpu-high"
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon pre-emption

# gpu 관련 설정
GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0}}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
N_GPU="${N_GPU:-$(awk -F',' '{print NF}' <<< "${GPU_IDS}")}"

# batch size 관련 설정
PER_GPU_BATCH="${PER_GPU_BATCH:-512}"
TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-512}"
RESUME_CKPT_PATH="${RESUME_CKPT_PATH:-}"

# batch size 검증
if (( TARGET_GLOBAL_BATCH % (PER_GPU_BATCH * N_GPU) != 0 )); then
  echo "ERROR: TARGET_GLOBAL_BATCH must be divisible by PER_GPU_BATCH*N_GPU."
  exit 1
fi
ACCUM=$(( TARGET_GLOBAL_BATCH / (PER_GPU_BATCH * N_GPU) ))

# 추가 인자 설정
EXTRA_ARGS=()
if [[ -n "${RESUME_CKPT_PATH}" ]]; then
  EXTRA_ARGS+=("checkpointing.resume_from_ckpt=true")
  EXTRA_ARGS+=("checkpointing.resume_ckpt_path=${RESUME_CKPT_PATH}")
fi


# To enable preemption re-loading, set `hydra.run.dir` or 
# `checkpointing.save_dir` explicitly.
# Note: we use algo.curriculum.mode=poly9 in the duo2 paper for speed
python -u -m main \
  loader.batch_size="${PER_GPU_BATCH}" \
  loader.eval_batch_size="${PER_GPU_BATCH}" \
  loader.global_batch_size="${TARGET_GLOBAL_BATCH}" \
  loader.eval_global_batch_size="${TARGET_GLOBAL_BATCH}" \
  trainer.devices="${N_GPU}" \
  trainer.accumulate_grad_batches="${ACCUM}" \
  data=lm1b-wrap \
  wandb.name=duo-lm1b-wrap-small-100k-b200x1 \
  model=small \
  algo=duo \
  model.length=128 \
  algo.curriculum.mode=simple \
  algo.curriculum.gumbel_tau_log10_start=-3.0 \
  algo.curriculum.gumbel_tau_log10_end=-3.0 \
  algo.curriculum.gamma_min=-3.5 \
  algo.curriculum.gamma_max=-1.75 \
  algo.curriculum.start=0 \
  algo.curriculum.end=500000 \
  trainer.max_steps=100000 \
  trainer.val_check_interval=2000 \
  trainer.precision=bf16-mixed \
  "${EXTRA_ARGS[@]}"