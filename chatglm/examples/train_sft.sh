#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python ../src/train_bash.py \
    --model_name_or_path  /home/amov/LLaMA-Efficient-Tuning/model/chatglm2-6b \
    --stage sft \
    --do_train \
    --dataset amov \
    --dataset_dir ../data \
    --finetuning_type lora \
    --output_dir /home/amov/ChatGLM-Efficient-Tuning/output/sft2\
    --overwrite_cache \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 1e-4 \
    --num_train_epochs 40.0 \
    --plot_loss \
    --quantization_bit 8 \
    --fp16
