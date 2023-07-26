CUDA_VISIBLE_DEVICES=1 python ../src/cli_demo.py \
    --model_name_or_path  /home/amov/LLaMA-Efficient-Tuning/model/baichuan \
    --quantization_bit 8 \
    --prompt_template baichuan 
    #--checkpoint_dir /home/amov/LLaMA-Efficient-Tuning/output/sft/checkpoint-3000 \
