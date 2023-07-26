CUDA_VISIBLE_DEVICES=0 python ../src/train_bash.py \
    --stage ppo \
    --model_name_or_path /home/amov/LLaMA-Efficient-Tuning/model/baichuan \
    --do_train \
    --dataset_dir ../data \
    --dataset amov \
    --finetuning_type lora \
    --checkpoint_dir /home/amov/LLaMA-Efficient-Tuning/output/sft \
    --reward_model /home/amov/LLaMA-Efficient-Tuning/output/rm \
    --output_dir /home/amov/LLaMA-Efficient-Tuning/output/ppo \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --prompt_template baichuan \
    --learning_rate 1e-5 \
    --num_train_epochs 3.0 \
    --resume_lora_training False \
    --plot_loss \
    --lora_target W_pack \
    --quantization_bit 4 \
    --fp16
