CUDA_VISIBLE_DEVICES=0 python ../python src/train_bash.py \
    --stage sft \
    --model_name_or_path  /home/amov/LLaMA-Efficient-Tuning/model/llama \
    --do_train \
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
    --prompt_template llama2 \
    --plot_loss \
    --quantization_bit 8 \
    --fp16