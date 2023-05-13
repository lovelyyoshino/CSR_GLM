# -*- coding: utf-8 -*-
# @Time    : 2023/3/9 15:29
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments
from deep_training.nlp.models.chatglm import setup_model_profile, ChatGLMConfig
from deep_training.nlp.models.lora.v2 import LoraArguments
from transformers import HfArgumentParser

from data_utils import train_info_args, NN_DataHelper
from models import MyTransformer,ChatGLMTokenizer

if __name__ == '__main__':
    train_info_args['seed'] = None
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, LoraArguments))
    model_args, training_args, data_args, _ = parser.parse_dict(train_info_args)

    setup_model_profile()

    dataHelper = NN_DataHelper(model_args, training_args, data_args)
    tokenizer: ChatGLMTokenizer
    tokenizer, config, _,_ = dataHelper.load_tokenizer_and_config(
        tokenizer_class_name=ChatGLMTokenizer, config_class_name=ChatGLMConfig)
    assert tokenizer.eos_token_id == 130005

    #加载int4模型
    if train_info_args['model_name_or_path'].find('int4') != -1:
        # 4 or 8
        config.quantization_bit = 4
    # 官方28层
    config.precision = 16
    config.num_layers = 28
    config.initializer_weight = False
    
    pl_model = MyTransformer(config=config, model_args=model_args, training_args=training_args)


    model = pl_model.get_glm_model()
    if not model.quantized:
        # 按需修改，目前只支持 4/8 bit 量化 ， 可以保存量化模型
        model.half().quantize(4).cuda()
    else:
        # 已经量化
        model.half().cuda()
    model = model.eval()

    # 注意 长度不等于2048 会影响效果
    response, history = model.chat(tokenizer, "你好", history=[],max_length=2048,
                                   eos_token_id=config.eos_token_id,
                                   do_sample=True, top_p=0.7, temperature=0.95,
                                   )
    print('你好',' ',response)

    response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history,max_length=2048,
                                   eos_token_id=config.eos_token_id,
                                   do_sample=True, top_p=0.7, temperature=0.95,
                                   )
    print('晚上睡不着应该怎么办',' ',response)

    # response, history = base_model.chat(tokenizer, "写一个诗歌，关于冬天", history=[],max_length=30)
    # print('写一个诗歌，关于冬天',' ',response)

