MASTER_PORT=$(shuf -n 1 -i 10000-65535)

deepspeed --num_gpus=4 --master_port $MASTER_PORT ../src/train_sft.py \
    --model_name_or_path  /home/amov/LLaMA-Efficient-Tuning/model/bloom \
    --do_train \
    --deepspeed deepspeed.json \
    --dataset_dir ../data \
    --dataset comparison_gpt4_en \
    --finetuning_type lora \
    --output_dir /home/amov/LLaMA-Efficient-Tuning/output/sft \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 1e-3 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --lora_target query_key_value \
    --quantization_bit 8 \
    --fp16