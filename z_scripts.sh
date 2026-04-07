# 캐시 설정.
export HF_HOME=/NHNHOME/WORKSPACE/0226010404_A/BISPL/DATA/geonyounglee/.hf_cache
export HF_DATASETS_CACHE=/NHNHOME/WORKSPACE/0226010404_A/BISPL/DATA/geonyounglee/.hf_cache/datasets
mkdir -p /NHNHOME/WORKSPACE/0226010404_A/BISPL/DATA/geonyounglee/.hf_cache

# BISPL2 환경이라면 conda activate ddlm 명령어 실행 이후 할 것.
alias fixpath='export PATH="/NHNHOME/WORKSPACE/0226010404_A/BISPL/BISPL2/miniconda/envs/ddlm/bin:$PATH"'
fixpath


cd /NHNHOME/WORKSPACE/0226010404_A/BISPL/DATA/geonyounglee/duo
mkdir -p watch_folder

# bispl2 gpu2,3 에서 실행. per 128
CUDA_VISIBLE_DEVICES=2,3 nohup bash "myscripts/train_lm1b_mdlm_sentencepacking.sh" \
  > watch_folder/mdlm_lm1b_wrap_b200x4.log 2>&1 &

# bispl3 gpu2 에서 실행. per 512
CUDA_VISIBLE_DEVICES=2 nohup bash "myscripts/train_lm1b_ar_sentencepacking.sh" \
  > watch_folder/ar_lm1b_wrap_b200x1.log 2>&1 &

# bispl2 gpu1 에서 실행. per 512
CUDA_VISIBLE_DEVICES=1 nohup bash "myscripts/train_lm1b_duo_sentencepacking.sh" \
  > watch_folder/duo_lm1b_wrap_b200x1.log 2>&1 &

pgrep -af "python -u -m main|train_lm1b_mdlm_sentencepacking.sh"
kill -15 <PID>
ps -fp <PID>
kill -9 <PID>
