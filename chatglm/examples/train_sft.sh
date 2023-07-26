#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ../src/train_bash.py \
    --model_name_or_path  /home/amov/LLaMA-Efficient-Tuning/model/chatglm2-6b \
    --stage sft \
    --do_train \
    --dataset alpaca_gpt4_zh \
    --dataset_dir ../data \
    --finetuning_type lora \
    --output_dir /home/amov/ChatGLM-Efficient-Tuning/output/sft \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --quantization_bit 8 \
    --fp16
