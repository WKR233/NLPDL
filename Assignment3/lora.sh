seed=(123 234 345 456 567)

for round in 0 1 2 3 4;
do
    for ft_task in 'restaurant_sup' 'laptop_sup' 'agnews_sup';
    do
        CUDA_VISIBLE_DEVICES=0 python lora.py \
        --dataset_name ${ft_task} \
        --model_name_or_path './roberta-base' \
        --seed ${seed[$round]} \
        --per_device_eval_batch_size 128 \
        --num_train_epochs 20 \
        --per_device_train_batch_size 128 \
        --output_dir ./output/$ft_task'_'$model_name'_seed'$round'LoRA' \
        --report_to 'wandb' \
        --logging_steps 1 \
        --evaluation_strategy 'epoch'
    done
done