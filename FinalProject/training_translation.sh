#! /bin/bash

#SBATCH --partition=IAI_SLURM_3090
#SBATCH --job-name=fine-tune
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --qos=singlegpu
#SBATCH --cpus-per-task=10
#SBATCH --time 48:00:00

for seed in 222;
do
    TRANSFORMERS_OFFLINE=1 \
    CUDA_VISIBLE_DEVICES=0 \
    WANDB_MODE=offline \
    python /ceph/home/wangkeran/NLPProj/lora_translation.py \
    -s $seed \
    -lr 5.6e-5 \
    -e 1 \
    -d '/ceph/home/wangkeran/NLPProj/datasets/englishdata.jsonl' \
    -m '/ceph/home/wangkeran/NLPProj/mt5-small' \
    -n 'mT5-englishdata' \
    -b 2
done