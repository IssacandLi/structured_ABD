#!/bin/bash
#SBATCH --partition=camas
#SBATCH --job-name=bd3lm_cnn_dm
#SBATCH --time=48:59:59
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --nodelist=sn16
#SBATCH --gres=gpu:tesla:2
#SBATCH --mem=300000

module load anaconda3
source activate /data/lab/yan/peihong_li/condaenvlist/bd3lm
cd /data/lab/yan/peihong_li/ACL/bd3lms3_cond_BD/bd3lms

BLOCK_SIZE=4

python -u main.py \
  model=small \
  mode=sample_eval \
  loader.eval_batch_size=1 \
  model.length=1024 \
  block_size=4 \
  algo=bd3lm \
  algo.T=5000 \
  data.train=cnn_dailymail \
  data.valid=cnn_dailymail \
  data.wrap=false \
  wandb=null \
  +data.conditional_generation=true \
  data.cache_dir=/data/lab/yan/peihong_li/data_cache/cdm_cnn_dm_dat \
  sampling.nucleus_p=0.9 \
  sampling.kv_cache=false \
  +sampling.context_size=512 \
  sampling.logdir=$PWD/sample_logs/cnn_dm_cond_bs4 \
  eval.checkpoint_path=$PWD/outputs/cnn_dailymail/2026.03.12/155540/checkpoints/best.ckpt