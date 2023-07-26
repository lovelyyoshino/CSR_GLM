CUDA_VISIBLE_DEVICES=0 python ../src/train_bash.py \
    --stage sft \
    --model_name_or_path /home/amov/LLaMA-Efficient-Tuning/model/baichuan \
    --do_train \
    --dataset_dir ../data \
    --dataset amov \
    --finetuning_type lora \
    --output_dir /home/amov/LLaMA-Efficient-Tuning/output/sft \
    --overwrite_cache \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --prompt_template baichuan \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --resume_lora_training False \
    --plot_loss \
    --lora_target W_pack \
    --quantization_bit 8 \
    --fp16
