import os, sys
import mdtex2html

import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)
import json

from arguments import ModelArguments, DataTrainingArguments


model = None
tokenizer = None


def parse_text(text):#拆分 md 代码块
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text


def predict(input, models, tokenizers, max_length, top_p, temperature, history):
    ground_truth = history[0]
    history = []
    for response, history in models.stream_chat(tokenizers, input, history, max_length=max_length, top_p=top_p,
                                               temperature=temperature):
        continue
    print("input:", input)
    print("response:", response)
    print("ground_truth:", ground_truth)



def main():
    global model, tokenizer

    parser = HfArgumentParser((
        ModelArguments, DataTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # 如果我们只传递一个参数给脚本，它是json文件的路径
        model_args,data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args,data_args = parser.parse_args_into_dataclasses()
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True)#传递预训练模型的token
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True)#传递预训练模型的配置文件

    config.pre_seq_len = model_args.pre_seq_len#前缀长度

    if model_args.ptuning_checkpoint is not None:#如果有ptuning的话
        print(f"Loading prefix_encoder weight from {model_args.ptuning_checkpoint}")#加载ptuning的权重
        model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)#加载模型
        prefix_state_dict = torch.load(os.path.join(model_args.ptuning_checkpoint, "pytorch_model.bin"))#加载ptuning的权重
        new_prefix_state_dict = {}
        for k, v in prefix_state_dict.items():# 从ptuning的权重中加载前缀编码器的权重
            if k.startswith("transformer.prefix_encoder."):#如果是前缀编码器的权重
                new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v#则将其加载到新的权重中
        model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)#用模型的transformer的前缀编码器加载新的权重
    else:
        model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)#否则直接加载模型

    if model_args.quantization_bit is not None:#如果有量化的话
        print(f"Quantized to {model_args.quantization_bit} bit")
        model = model.quantize(model_args.quantization_bit)#量化模型

    if model_args.pre_seq_len is not None:#如果有前缀长度的话
        # P-tuning v2
        model = model.half().cuda()#将模型转换为半精度浮点数
        model.transformer.prefix_encoder.float().cuda()#将前缀编码器转换为单精度浮点数
    
    model = model.eval()#将模型设置为评估模式
    # 创建一个空的列表来保存所有的 JSON 对象
    datas = []
    with open(data_args.test_file, 'r', encoding='utf-8') as json_file:
        for line in json_file:
            datas.append(json.loads(line))
    for data in datas:
        predict(data["content"], model, tokenizer, 2048, 0.7, 0.95, [data["summary"]])


if __name__ == "__main__":
    main()