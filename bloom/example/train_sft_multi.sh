accelerate launch ../src/train_sft.py \
    --model_name_or_path /home/amov/LLaMA-Efficient-Tuning/model/bloom \
    --do_train \
    --dataset_dir ../data \
    --dataset alpaca_gpt4_en \
    --finetuning_type lora \
    --checkpoint_dir /home/amov/LLaMA-Efficient-Tuning/output/pt \
    --output_dir /home/amov/LLaMA-Efficient-Tuning/output/sft \
    --overwrite_cache \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --resume_lora_training False \
    --plot_loss \
    --lora_target query_key_value \
    --quantization_bit 8 \
    --fp16