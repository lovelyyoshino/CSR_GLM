accelerate launch --config_file accelerate_config.yaml ../src/train_bash.py \
    --stage sft \
    --model_name_or_path /home/amov/LLaMA-Efficient-Tuning/model/bloom \
    --do_train \
    --dataset_dir ../data \
    --dataset amov \
    --finetuning_type lora \
    --output_dir /home/amov/LLaMA-Efficient-Tuning/output/sft \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --resume_lora_training False \
    --plot_loss \
    --lora_rank 2 \
    --lora_target query_key_value \
    --ddp_find_unused_parameters False \
    --quantization_bit 8 \
    --fp16
    #--checkpoint_dir /home/amov/LLaMA-Efficient-Tuning/output/pt \
    #--overwrite_cache \