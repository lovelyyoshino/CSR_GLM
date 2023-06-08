CUDA_VISIBLE_DEVICES=1 python ../src/cli_demo.py \
    --dataset_dir ../data \
    --model_name_or_path  /home/amov/LLaMA-Efficient-Tuning/model/bloom \
    --checkpoint_dir /home/amov/LLaMA-Efficient-Tuning/output/sft/checkpoint-3000 \
    --quantization_bit 8