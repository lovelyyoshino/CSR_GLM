CUDA_VISIBLE_DEVICES=0 python ../src/cli_demo.py \
    --model_name_or_path  /home/amov/LLaMA-Efficient-Tuning/model/chatglm2-6b \
    --checkpoint_dir /home/amov/ChatGLM-Efficient-Tuning/output/sft1/checkpoint-10000 \
    --quantization_bit 8

