PRE_SEQ_LEN=128
CHECKPOINT=adgen-chatglm-6b-pt-128-2e-2
STEP=200

CUDA_VISIBLE_DEVICES=1 python inference.py \
    --test_file AdvertiseGen/dev.json \
    --model_name_or_path /home/amov/ChatGLM-6B/model_int4  \
    --ptuning_checkpoint ./output/$CHECKPOINT/checkpoint-$STEP \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4

