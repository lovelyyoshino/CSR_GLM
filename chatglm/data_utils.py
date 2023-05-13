# @Time    : 2023/1/22 16:22
# @Author  : tk
# @FileName: data_utils.py
import copy
import json
import os
import random
import typing
from enum import Enum
import numpy as np
import torch
import sys
from deep_training.data_helper import DataHelper, ModelArguments, TrainingArguments, DataArguments
from deep_training.nlp.models.chatglm import ChatGLMConfig
from deep_training.nlp.models.lora.v2 import LoraArguments
from fastdatasets.record import load_dataset as Loader, RECORD, WriterObject, gfile
from tqdm import tqdm
from transformers import HfArgumentParser

from data_processer import DataStrategy, TokenTruncation, TokenSingleSliding, TokenDoubleSliding
from models import ChatGLMTokenizer

lora_info_args = {
    'with_lora': True,  # 是否启用lora模块
    'r': 16,
    'target_modules': ['dense','dense_h_to_4h','dense_4h_to_h','query_key_value'],
    'target_dtype': 16, # 半精度
    'lora_alpha': 32,
    'lora_dropout': 0.1,
    'bias': 'none',  # Bias type for Lora. Can be 'none', 'all' or 'lora_only'"
    'modules_to_save' : None, # "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "

}

adalora_info_args = {
    'with_lora': False,  # 是否启用adalora模块
    'r': 8,
    'target_modules': ['query_key_value'],
    'target_dtype': 16, # 半精度
    'lora_alpha': 32,
    'lora_dropout': 0.1,
    'bias': 'none',  # Bias type for Lora. Can be 'none', 'all' or 'lora_only'"
    'modules_to_save' : None, # "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "

    'target_r':8, # Target Lora matrix dimension.
    'init_r': 12, #Intial Lora matrix dimension.
    'tinit': 0, #The steps of initial warmup.
    'tfinal': 0, #The steps of final warmup.
    'deltaT': 1, #Step interval of rank allocation.
    'beta1': 0.85, #Hyperparameter of EMA.
    'beta2': 0.85, #Hyperparameter of EMA.
    'orth_reg_weight': 0.5, #The orthogonal regularization coefficient.
    'total_step': None, #The total training steps.
    'rank_pattern': None, #The saved rank pattern.
}
#训练时的参数及其对应的值
train_info_args = {
    # 'devices': [0, 1, 2, 3],   #表示使用的 GPU 设备数量
    'devices': 1,
    'data_backend': 'record',  #one of record lmdb, 超大数据集可以使用 lmdb , 注 lmdb 存储空间比record大,数据的存储
    'model_type': 'chatglm',
    # 预训练模型路径 , 从0训练，则置空
    'model_name_or_path': '../models/ChatGLM_INT4',
    'config_name': './config/config.json',
    'tokenizer_name': '../models/ChatGLM_INT4', # 分词器路径，如果用预训练模型路径与model_name_or_path相同
    'convert_onnx': False, # 转换onnx模型
    'do_train': True,
    'train_file':  [ './data/output.json'],  #训练数据的路径，这里最好是.json文件的格式
    'max_epochs': 38, 
    'max_steps': -1,    # 最大训练步数,-1表示不受限制，训练会一直进行下去，直到达到预定的训练轮数或发生其他终止条件
    'optimizer': 'lion', # one of adamw,adam,lamb,lion 优化器类型

    'scheduler_type': 'CAWR', #学习率调度器 
    'scheduler':{'T_mult': 1, #倍增因子，这里表示不使用
                 'rewarm_epoch_num': 0.5,  # 如果 max_epochs is not None !
                 # 'T_0': 50000,    # 如果 max_epochs is None , 设定步数
                 'verbose': False }, #是否输出调度器的调试信息
    #其他的学习率调度器及其里面的字典参数
    # 'scheduler_type': 'linear',# one of [linear,WarmupCosine,CAWR,CAL,Step,ReduceLROnPlateau
    # 'scheduler': None,

    # 切换scheduler类型
    # 'scheduler_type': 'WarmupCosine',
    # 'scheduler': None,

    # 'scheduler_type': 'ReduceLROnPlateau',
    # 'scheduler': None,

    # 'scheduler_type': 'Step',
    # 'scheduler':{ 'decay_rate': 0.999,'decay_steps': 100,'verbose': True},

    # 'scheduler_type': 'CAWR',
    # 'scheduler':{'T_mult': 1, 'rewarm_epoch_num': 2, 'verbose': True},

    # 'scheduler_type': 'CAL',
    # 'scheduler': {'rewarm_epoch_num': 2,'verbose': True},


    'optimizer_betas': (0.9, 0.999), #优化器中用于计算梯度的系数
    'train_batch_size': 2,    #训练batch
    'eval_batch_size': 2,     #验证batch
    'test_batch_size': 2,     #测试batch
    'learning_rate': 2e-5,    #学习率 
    'adam_epsilon': 1e-8,     #Adam 优化器中的 epsilon 参数
    'gradient_accumulation_steps': 1,  #梯度累积的步数，1表示不累积
    'max_grad_norm': 1.0,              #梯度的最大范数，限制梯度大小，这里表示不对其进行限制，如果出现梯度爆炸可以使用
    'weight_decay': 0,                 #优化器的权重衰减系数，不使用
    'warmup_steps': 0,                 #学习率的预热步数，不使用
    'output_dir': './output',          #模型输出的目录
    'max_seq_length': 2048, # 如果资源充足，推荐长度2048 与官方保持一致
    'max_target_length': 100,  # 预测最大长度, 保留字段
    'use_fast_tokenizer': False,  #快速分词器
    'do_lower_case': False,       #小写化处理

    ##############  lora模块
    #注意lora,adalora 和 ptuning-v2 禁止同时使用
   'lora': {**lora_info_args},   # LORA 模块
   'adalora': {**adalora_info_args},  # ADA-LORA 模块
}

#lora 模式暂时不支持deepspeed
enable_deepspeed = False


data_conf = {
   'strategy': DataStrategy.truncation, # 数据策略选项
    DataStrategy.truncation: {
        'ensure_answer_min_length': 1,
    },
    DataStrategy.singlesliding: {
        'sliding_size': train_info_args['max_seq_length'] // 3 * 2, #prompt滑动窗口大小
        'p':1, # p < 0 , 随机选举prompt
    },
    DataStrategy.doublesliding: {
        'sliding_size': train_info_args['max_seq_length'] // 3 * 2, #双滑滑动窗口大小
        'p':1,# p < 0 , 随机选举prompt
    },
}



def get_deepspeed_config():
    # 是否开启deepspeed
    if not enable_deepspeed:
        return None
    with open('./deepspeed.json', mode='r', encoding='utf-8') as f:
        deepspeed_config = json.loads(f.read())
    return deepspeed_config

def preprocess(text):
  #text = text.replace("\n", "\\n").replace("\t", "\\t")
  return text

def postprocess(text):
  # return text.replace("\\n", "\n").replace("\\t", "\t")
  return text



class NN_DataHelper(DataHelper):
    index = 1
    def on_data_ready(self):
        self.index = -1

    # 切分词
    def on_data_process(self, data: typing.Any, mode: str):
        self.index += 1
        prompt = data[0]
        answer = data[1]

        tokenizer: ChatGLMTokenizer
        config: ChatGLMConfig
        max_seq_length = self.max_seq_length_dict[mode]
        tokenizer = self.tokenizer
        config = self.config

        if not hasattr(self, 'sptoken'):
            self.sptoken = tokenizer.encode(text="")[-2:]

        a_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
        b_ids = tokenizer.encode(text=answer, add_special_tokens=False)


        strategy = data_conf['strategy']
        if strategy == DataStrategy.truncation:
            ds = TokenTruncation.process(tokenizer,config,a_ids, b_ids, max_seq_length, self.sptoken ,**data_conf[strategy])
        elif strategy == DataStrategy.singlesliding:
            ds = TokenSingleSliding.process(tokenizer,config, a_ids, b_ids, max_seq_length, self.sptoken, **data_conf[strategy])
        elif strategy == DataStrategy.doublesliding:
            ds = TokenDoubleSliding.process(tokenizer,config, a_ids, b_ids, max_seq_length, self.sptoken, **data_conf[strategy])
        else:
            raise ValueError('Invlid strategy',strategy)

        if not ds:
            return None

        if self.index < 3:
            print(ds[0])
        return ds

    # {
    #     "id": 0, "paragraph": [
    #     # 一轮会话
    #     {
    #         "q": "从南京到上海的路线",
    #         "a": [
    #             "你好，南京到上海的路线如下：",
    #             "1. 南京到上海，可以乘坐南京地铁1号线，在南京站乘坐轨道交通1号线。",
    #             "2. 南京到浦东机场，可以搭乘上海地铁1号，在陆家嘴站乘坐地铁1线，在浦东国际机场站乘坐机场快线，前往上海浦东国际机场。",
    #             "3. 上海到南京，可以换乘上海地铁2号线，从南京站换乘地铁2线，再从南京南站换乘地铁1路，然后到达上海站"
    #         ]
    #     }
    #     # 二轮....
    # ]
    # }

    # 读取文件
    def on_get_corpus(self, files: typing.List, mode: str):
        current_line_number = 0
        D = []
        for file in files:
            with open(file, mode='r', encoding='utf-8', newline='\n') as f:
                lines = f.readlines()

            for line_id, line in enumerate(lines):
                current_line_number += 1
                print(f"Processing line number: {current_line_number}")
                jd = json.loads(line)
                if not jd:
                    continue
                paragraph = jd['paragraph']
                if line_id < 10:
                    print(paragraph)
                #兼容支持 answer string
                paragraph = [(preprocess(session['q']),
                              preprocess('\n'.join(session['a'])) if isinstance(session['a'],list) else preprocess(session['a']))
                    for session in paragraph]
                for sid,(q,a) in enumerate(paragraph):
                    # assert len(a),ValueError('answer cannot empty')
                    try:
                        assert len(a), ValueError('answer cannot be empty')
                    except AssertionError as e:
                        line_number = sys.exc_info()[-1].tb_lineno
                        print(f"AssertionError occurred at line {line_number}")
                        raise e
                    if sid == 0:
                        D.append((q, a))
                    else:
                        prompt_text = ''
                        for j in range(sid + 1):
                            if j == sid:
                                prompt_text += "[Round {}]\n问：{}\n答：".format(sid, paragraph[j][0])
                            else:
                                prompt_text += "[Round {}]\n问：{}\n答：{}".format(j, paragraph[j][0], paragraph[j][1])
                        D.append((prompt_text,a))
        return D

    def collate_fn(self,batch):
        if not hasattr(self,'sptoken'):
            self.sptoken = self.tokenizer.encode(text="")[-2:]

        o = {}
        for i, b in enumerate(batch):
            if i == 0:
                for k in b:
                    o[k] = [torch.tensor(b[k])]
            else:
                for k in b:
                    o[k].append(torch.tensor(b[k]))
        for k in o:
            o[k] = torch.stack(o[k])


        max_len = torch.max(o.pop('seqlen')).tolist()
        b_input_ids = o['input_ids'][:, :max_len]
        ctxlens = o.pop('ctxlen')  # 兼容旧版本数据
        if ctxlens is None:
            ctxlens = [None] * len(b_input_ids)

        b_position_ids,b_attention_mask = [],[]
        for input_ids,context_length in zip(b_input_ids,ctxlens):
            context_length = context_length.squeeze(dim=-1)
            mask_position = context_length - 1
            position_ids = list(range(context_length)) + [mask_position] * (max_len - context_length)
            block_position_ids = [0] * context_length + list(range(1, max_len - context_length + 1))


            attention_mask = torch.ones((1, max_len, max_len))
            attention_mask = torch.tril(attention_mask)
            attention_mask[..., :context_length] = 1
            attention_mask = (attention_mask < 0.5)

            b_position_ids.append(torch.stack((torch.tensor(position_ids),torch.tensor(block_position_ids))))
            b_attention_mask.append(attention_mask)

        b_attention_mask = torch.stack(b_attention_mask, dim=0)
        b_position_ids = torch.stack(b_position_ids,dim=0)

        o['input_ids'] = b_input_ids.long()
        o['attention_mask'] = b_attention_mask.bool()
        o['position_ids'] = b_position_ids.long()
        o['labels'] = o['labels'][:, :max_len].long()
        return o


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, LoraArguments)) # 命令行参数解析器
    model_args, training_args, data_args, lora_args = parser.parse_dict(train_info_args)         # 解析结果
    lora_args = lora_args.config

    dataHelper = NN_DataHelper(model_args, training_args, data_args)
    tokenizer, config, _,_ = dataHelper.load_tokenizer_and_config(tokenizer_class_name=ChatGLMTokenizer,config_class_name=ChatGLMConfig)
    assert tokenizer.eos_token_id == 130005



    # 缓存数据集
    # 检测是否存在 output/dataset_0-train.record ，不存在则制作数据集
    if data_args.do_train:
        dataHelper.make_dataset_with_args(data_args.train_file,mixed_data=False,shuffle=True,mode='train')
    if data_args.do_eval:
        dataHelper.make_dataset_with_args(data_args.eval_file, shuffle=False,mode='eval')
    if data_args.do_test:
        dataHelper.make_dataset_with_args(data_args.test_file, shuffle=False,mode='test')


    # def shuffle_records(record_filenames, outfile, compression_type='GZIP'):
    #     print('shuffle_records record...')
    #     options = RECORD.TFRecordOptions(compression_type=compression_type)
    #     dataset_reader = Loader.RandomDataset(record_filenames, options=options, with_share_memory=True)
    #     data_size = len(dataset_reader)
    #     all_example = []
    #     for i in tqdm(range(data_size), desc='load records'):
    #         serialized = dataset_reader[i]
    #         all_example.append(serialized)
    #     dataset_reader.close()
    #
    #     shuffle_idx = list(range(data_size))
    #     random.shuffle(shuffle_idx)
    #     writer = WriterObject(outfile, options=options)
    #     for i in tqdm(shuffle_idx, desc='shuffle record'):
    #         example = all_example[i]
    #         writer.write(example)
    #     writer.close()
    #
    #
    # # 对每个record 再次打乱
    # for filename in dataHelper.train_files:
    #     shuffle_records(filename, filename)
