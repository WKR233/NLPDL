#! /bin/bash

#SBATCH --partition=IAI_SLURM_3090
#SBATCH --job-name=fine-tune
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --qos=singlegpu
#SBATCH --cpus-per-task=10
#SBATCH --time 24:00:00

for seed in 111 222 333;
do
    TRANSFORMERS_OFFLINE=1 \
    CUDA_VISIBLE_DEVICES=0 \
    WANDB_MODE=offline \
    python /ceph/home/wangkeran/NLPProj/lora.py \
    -s $seed \
    -lr 5.6e-5 \
    -e 5 \
    -d '/ceph/home/wangkeran/NLPProj/datasets/nlpcc_data.json' \
    -m '/ceph/home/wangkeran/NLPProj/mt5-large' \
    -n 'mT5-nlpcc_data' \
    -b 4
done