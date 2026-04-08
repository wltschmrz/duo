# settings
conda create -n duo python=3.12
conda activate duo
conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt
pip install ninja packaging
wget "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.3cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
pip install ./flash_attn-2.7.4.post1+cu12torch2.3cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

# settings BW
pip install --upgrade pip setuptools wheel
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r req_bw.txt
wget "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
pip install ./flash_attn-2.8.0.post2+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# pretrained ckpts
pip install -U gdown
gdown --folder "https://drive.google.com/drive/folders/16LuuptK7Xfk-vzhQYZBZ0SA-B-BFluau"
gdown --folder "https://drive.google.com/drive/folders/1JpqFM8XRvifwIkjWPfMyuDvu41r1yk0t"

# 캐시 설정.
export HF_HOME=/NHNHOME/WORKSPACE/0226010404_A/BISPL/DATA/geonyounglee/.hf_cache
export HF_DATASETS_CACHE=/NHNHOME/WORKSPACE/0226010404_A/BISPL/DATA/geonyounglee/.hf_cache/datasets
mkdir -p /NHNHOME/WORKSPACE/0226010404_A/BISPL/DATA/geonyounglee/.hf_cache

# BISPL2 환경이라면 conda activate ddlm 명령어 실행 이후 할 것.
alias fixpath='export PATH="/NHNHOME/WORKSPACE/0226010404_A/BISPL/BISPL2/miniconda/envs/ddlm/bin:$PATH"'
fixpath

# git branch 옆 dir에 clone
git clone -b ddlms https://github.com/geonyounglee/duo.git


cd /NHNHOME/WORKSPACE/0226010404_A/BISPL/DATA/geonyounglee/duo
mkdir -p watch_folder

# data 저장되는 cache 경로 설정.

# batch size 설정.

# bispl2 gpu2,3 에서 실행. per 128
CUDA_VISIBLE_DEVICES=2,3 nohup bash "myscripts/train_lm1b_mdlm_sentencepacking.sh" \
  > watch_folder/mdlm_lm1b_wrap_b200x4.log 2>&1 &

# bispl3 gpu2 에서 실행. per 512
CUDA_VISIBLE_DEVICES=2 nohup bash "myscripts/train_lm1b_ar_sentencepacking.sh" \
  > watch_folder/ar_lm1b_wrap_b200x1.log 2>&1 &

# bispl2 gpu1 에서 실행. per 512
CUDA_VISIBLE_DEVICES=1 nohup bash "myscripts/train_lm1b_duo_sentencepacking.sh" \
  > watch_folder/duo_lm1b_wrap_b200x1.log 2>&1 &

# jordy gpu0123 에서 실행. per 128
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup bash "myscripts/train_lm1b_fldd_sentencepacking.sh" \
  > watch_folder/fldd_lm1b_wrap_h200x4.log 2>&1 &


CUDA_VISIBLE_DEVICES=0,1,2,3 RESUME_CKPT_PATH="/home/geonyounglee/workspace/duo/outputs/lm1b/2026.04.08/194939/checkpoints/last.ckpt" \
WANDB_NAME="fldd-lm1b-wrap-small-100k-H200x4" SEED=1 \
nohup bash "myscripts/train_lm1b_fldd_sentencepacking.sh" \
  > watch_folder/fldd_lm1b_wrap_h200x4-2.log 2>&1 &

# FLDD 학습인데 지금 penda에서 하는 중.
CUDA_VISIBLE_DEVICES=1,2,3,4 RESUME_CKPT_PATH="/mnt/data1/geonyounglee/workspace/duo/outputs/lm1b/2026.04.08/194939/checkpoints/last-v1.ckpt" \
WANDB_NAME="fldd-lm1b-wrap-small-100k-H200x4" SEED=1 PER_GPU_BATCH=16 \
nohup bash "myscripts/train_lm1b_fldd_sentencepacking.sh" \
  > watch_folder/fldd_lm1b_wrap_h200x4-3.log 2>&1 &


pgrep -af "python -u -m main|train_lm1b_mdlm_sentencepacking.sh"
kill -15 <PID>
ps -fp <PID>
kill -9 <PID>
