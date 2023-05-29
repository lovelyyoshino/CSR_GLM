# 通用函数，里面主要是一些chatglm常用的算法类
import os
import sys
import torch
import hashlib
from typing import List, Literal, Optional, Tuple

import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

import datasets
from datasets import Dataset, concatenate_datasets, load_dataset

from peft import (
    PeftModel,
    TaskType,
    LoraConfig,
    get_peft_model
)

from trl import AutoModelForCausalLMWithValueHead

from .config import (
    ModelArguments,
    DataTrainingArguments,
    FinetuningArguments
)#模型一些参数信息，主要是在config中

from .other import (
    get_logger,
    load_trainable_params,
    load_valuehead_params,
    print_trainable_params,
    prepare_model_for_training,
    IGNORE_INDEX,
    FINETUNING_ARGS_NAME
)#从other中拿到一些参数获取的函数

check_min_version("4.27.4")#检查一些版本号
require_version("datasets>=2.10.0", "To fix: pip install datasets>=2.10.0")
require_version("peft>=0.3.0", "To fix: pip install peft>=0.3.0")
require_version("trl>=0.4.1", "To fix: pip install trl>=0.4.1")


logger = get_logger(__name__)

"""_summary_
model:主要表示预训练的模型传入
model_args:模型待处理的参数
finetuning_args:微调模型的参数
is_trainable:是否是处于训练阶段还是测试阶段
"""
def init_adapter(
        model: PreTrainedModel,
        model_args: ModelArguments,
        finetuning_args: FinetuningArguments,
        is_trainable: bool
) -> PreTrainedModel:
    r"""
    初始化适配器。

    支持全参数、冻结、P-Tuning v2 和 LoRA 训练。

    请注意，训练时候参数必须转换为 float32。
    """

    if finetuning_args.finetuning_type == "none" and is_trainable:#如果不微调，但是需要训练，报错
        raise ValueError("You cannot use finetuning_type=none while training.")

    if finetuning_args.finetuning_type == "full":#如果是全参数微调
        logger.info("Fine-tuning method: Full")
        model = model.float()#转换为float类型

    if finetuning_args.finetuning_type == "freeze":#如果是冻结
        logger.info("Fine-tuning method: Freeze")
        for name, param in model.named_parameters():#遍历模型的参数
            if not any(trainable_layer in name for trainable_layer in finetuning_args.trainable_layers):#如果不在可训练的层中
                param.requires_grad_(False)#不需要梯度
            else:
                param.data = param.data.to(torch.float32)#否则将参数模型转换为float32类型

    if finetuning_args.finetuning_type == "p_tuning":#如果是p_tuning
        logger.info("Fine-tuning method: P-Tuning v2") #则不要改动模型的参数

    if finetuning_args.finetuning_type != "lora" and model_args.checkpoint_dir is not None:#如果不是lora（lora有一套自己的逻辑），且模型的检查点不为空，则导入模型权重文件
        load_trainable_params(model, model_args.checkpoint_dir[0]) # 加载非 PEPF 方法的模型检查点

    if finetuning_args.finetuning_type == "lora":#如果是lora
        logger.info("Fine-tuning method: LoRA")
        lastest_checkpoint = None#最新的检查点为空

        if model_args.checkpoint_dir is not None:#如果模型的检查点不为空
            if is_trainable and finetuning_args.resume_lora_training: #如果训练的阶段且是可以恢复lora训练：继续训练LoRa权重
                checkpoints_to_merge, lastest_checkpoint = model_args.checkpoint_dir[:-1], model_args.checkpoint_dir[-1]#检查点合并，最新的检查点
            else:
                checkpoints_to_merge = model_args.checkpoint_dir#否则检查点合并为模型的检查点

            for checkpoint in checkpoints_to_merge:#拿到所有的检查点
                model = PeftModel.from_pretrained(model, checkpoint)#从预训练模型中加载模型权重
                model = model.merge_and_unload()#合并并卸载模型权重

            if len(checkpoints_to_merge) > 0:#如果检查点合并的长度大于0
                logger.info("Merged {} model checkpoint(s).".format(len(checkpoints_to_merge)))#打印合并的检查点的长度

            if lastest_checkpoint is not None: # 如果最新的检查点不为空
                model = PeftModel.from_pretrained(model, lastest_checkpoint, is_trainable=True)#从预训练模型中加载模型权重

        if is_trainable and lastest_checkpoint is None: #在训练的时候创建一个新的检查点
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, # we should regard ChatGLM as a causal LM
                inference_mode=False,
                r=finetuning_args.lora_rank,
                lora_alpha=finetuning_args.lora_alpha,
                lora_dropout=finetuning_args.lora_dropout,
                target_modules=finetuning_args.lora_target
            )#lora的配置
            model = get_peft_model(model, lora_config)#获取peft模型

    return model



'''_summary_
model_args:模型待处理的参数
training_args:训练的参数
finetuning_args:微调模型的参数
is_trainable:是否是处于训练阶段还是测试阶段
stage:训练的阶段,其实指的是是否用rhlf
'''
def load_pretrained(
        model_args: ModelArguments,
        training_args: Optional[Seq2SeqTrainingArguments] = None,
        finetuning_args: Optional[FinetuningArguments] = None,
        is_trainable: Optional[bool] = False,
        stage: Optional[Literal["sft", "rm", "ppo"]] = "sft"
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    r"""
    加载预训练模型和分词器。

    支持训练和推理。
    """
    if (not is_trainable) and (model_args.checkpoint_dir is None):#如果不是训练阶段且模型的检查点为空
        logger.warning("Checkpoint is not found at evaluation, load the original model.")
        finetuning_args = FinetuningArguments(finetuning_type="none")#微调的参数为none

    if model_args.checkpoint_dir is not None: # 从检查点加载微调模型
        for checkpoint_dir in model_args.checkpoint_dir:#遍历模型的检查点
            if not os.path.isfile(os.path.join(checkpoint_dir, FINETUNING_ARGS_NAME)):#如果不是文件
                raise ValueError("The fine-tuning arguments are not found in the provided dictionary.")
        logger.info("Load fine-tuned model from checkpoint(s): {}".format(",".join(model_args.checkpoint_dir)))#打印加载的模型的检查点
        finetuning_args = FinetuningArguments.load_from_json(os.path.join(model_args.checkpoint_dir[-1], FINETUNING_ARGS_NAME))#从json文件中加载微调的参数
        if finetuning_args.finetuning_type != "lora" and len(model_args.checkpoint_dir) > 1:#如果不是lora且模型的检查点的长度大于1
            logger.warning("Only LoRA tuning accepts multiple checkpoints.")

    assert stage == "sft" or finetuning_args.finetuning_type == "lora", "RM and PPO training can only be performed with LoRA method."#断言训练的阶段为sft或者微调的类型为lora，因为rm和ppo只能用lora方法

    quantization = None
    if model_args.quantization_bit is not None:#如果模型的量化位不为空
        if is_trainable:#如果是训练阶段
            if finetuning_args.finetuning_type == "full":#如果微调的类型为full
                raise ValueError("Full-parameter fine-tuning does not support quantization.")
            elif finetuning_args.finetuning_type == "p_tuning":
                quantization = "cpm" # 使用cpm的量化
            else:
                quantization = "bnb" # 使用bnb的量化
        else:
            quantization = "cpm"#不在训练阶段，则使用cpm的量化

    config_kwargs = {
        "trust_remote_code": True,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }#配置的参数，主要是ChatGLM的模型下载的参数

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        padding_side="left",
        **config_kwargs
    )#从预训练模型中加载分词器

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        **config_kwargs
    )#从预训练模型中加载配置

    # Tuning v2 配置。我们使用 ChatGLM 的内置 p-tuning 方法，因为 ChatGLM 的注意力掩码不同寻常，所以无法使用 PEFT
    if finetuning_args.finetuning_type == "p_tuning":#如果微调的类型为p_tuning
        config.pre_seq_len = finetuning_args.pre_seq_len # 预测序列的长度，启用此选项将自动修复其他参数
        config.prefix_projection = finetuning_args.prefix_projection#前缀投影

    # 使用bitsandbytes库进行Full、Freeze和LoRA的训练量化配置。
    if quantization == "bnb":#如果是bnb的量化
        assert model_args.quantization_bit == 8, "Freeze and LoRA fine-tuning only accept 8-bit quantization."#断言模型的量化位为8

        require_version("bitsandbytes>=0.37.0", "bitsandbytes library is required to use this feature.")
        from bitsandbytes.cuda_setup.main import get_compute_capability, get_cuda_lib_handle, is_cublasLt_compatible#从bitsandbytes.cuda_setup.main中导入get_compute_capability, get_cuda_lib_handle, is_cublasLt_compatible
        cuda = get_cuda_lib_handle()#获取cuda的库句柄
        cc = get_compute_capability(cuda)#获取cuda的计算能力
        assert is_cublasLt_compatible(cc), "The current GPU(s) is incompatible with quantization."

        config_kwargs["load_in_8bit"] = True # 加载8位的模型
        config_kwargs["device_map"] = "auto" # 它不应该在load_in_8bit之外指定

    # 加载并准备预训练模型，不包括valuehead
    model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, **config_kwargs)
    model = prepare_model_for_training(model) if is_trainable else model#如果是训练阶段，则准备模型
    model = init_adapter(model, model_args, finetuning_args, is_trainable)#初始化适配器

    if not is_trainable:#如果不是训练阶段
        model.requires_grad_(False) # 修复所有模型参数
        model = model.half() # 将所有参数转换为float16以进行推断

    # 使用内置的方法对 P-Tuning v2 训练或评估进行量化。在量化的 P-Tuning 环境中，模型参数应该转换为 float16。
    if quantization == "cpm":
        assert model_args.quantization_bit in [4, 8], "P-Tuning v2 and inference mode only accept 4-bit or 8-bit quantization."#断言模型的量化位为4或8
        assert not (is_trainable and training_args.fp16), "FP16 training conflicts with cpm quantization."

        if is_trainable: # 将所有参数转换为半精度，除了在训练中的prefix_encoder
            for name, param in model.named_parameters():#遍历模型的参数
                if "prefix_encoder" not in name:#如果不是前缀编码器
                    param.data = param.data.to(torch.float16)#将参数转换为float16

        model.quantize(model_args.quantization_bit) # ChatGLM-6B中的内置方法，也是一种就地操作

    if quantization is not None:#如果量化不为空
        logger.info("Quantized model to {} bit.".format(model_args.quantization_bit))

    if stage == "rm" or stage == "ppo": # 增加变量头，这个需要基于lora的
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)#从预训练模型中加载模型

        if stage == "ppo": # 加载reward模型
            assert is_trainable, "PPO stage cannot be performed at evaluation."#断言是训练阶段
            assert model_args.reward_model is not None, "Reward model is necessary for PPO training."
            logger.info("Load reward model from {}".format(model_args.reward_model))
            model.pretrained_model.load_adapter(model_args.reward_model, "reward", is_trainable=False)#加载适配器
            load_valuehead_params(model, model_args.reward_model)#加载valuehead的参数

        # 将参数 _is_int8_training_enabled 设置为 AutoModelForCausalLMWithValueHead 模型。以满足 transformers 库的合规要求
        if quantization == "bnb":
            model._is_int8_training_enabled = True

    print_trainable_params(model)#打印可训练的参数

    return model, tokenizer


def prepare_args() -> Tuple[ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, FinetuningArguments]:

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, FinetuningArguments))#创建一个参数解析器，和chatglm类似

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"): # 提供一个JSON文件作为论据
        model_args, data_args, training_args, finetuning_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, finetuning_args = parser.parse_args_into_dataclasses()

    # 启动logging
    if training_args.should_log:
        # training_args.log_level 的默认值是被动的，因此我们在这里将日志级别设置为 info，以使用该默认值。
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # 检查参数（不要检查finetuning_args，因为它可能从检查点加载）
    if training_args.do_train and training_args.predict_with_generate:
        raise ValueError("`predict_with_generate` cannot be set to True while training.")

    if model_args.quantization_bit is not None and training_args.do_train == False:
        logger.warning("Evaluating model in 4/8-bit mode may cause lower results.")

    if training_args.do_train and (not training_args.fp16):
        logger.warning("We recommend enable fp16 mixed precision training for ChatGLM-6B.")

    if training_args.local_rank != -1 and training_args.ddp_find_unused_parameters is None:
        logger.warning("`ddp_find_unused_parameters` needs to be set as False in DDP training.")
        training_args.ddp_find_unused_parameters = False

    training_args.optim = "adamw_torch" if training_args.optim == "adamw_hf" else training_args.optim # suppress warning

    # 为每个进程记录小摘要
    logger.info(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}\n"
        + f"  distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # 在初始化模型之前设置种子
    transformers.set_seed(training_args.seed)

    return model_args, data_args, training_args, finetuning_args

'''
__summery__
model_args:模型的参数文件
data_args:参数文件
'''
def prepare_data(
        model_args: ModelArguments,
        data_args: DataTrainingArguments
) -> Dataset:

    def checksum(file_path, hash):#检查hash的sum值
        with open(file_path, "rb") as datafile:#读取文件
            binary_data = datafile.read()
        sha1 = hashlib.sha1(binary_data).hexdigest()
        if sha1 != hash:
            logger.warning("Checksum failed for {}. It may vary depending on the platform.".format(file_path))

    max_samples = data_args.max_samples#设置最大的测试例子
    all_datasets: List[Dataset] = [] # support multiple datasets

    for dataset_attr in data_args.dataset_list:#遍历数据集

        logger.info("Loading dataset {}...".format(dataset_attr))#打印数据集的名称

        if dataset_attr.load_from == "hf_hub":#从hf_hub加载数据集
            raw_datasets = load_dataset(dataset_attr.dataset_name, cache_dir=model_args.cache_dir)
        elif dataset_attr.load_from == "script":#从脚本加载数据集
            raw_datasets = load_dataset(
                os.path.join(data_args.dataset_dir, dataset_attr.dataset_name),
                cache_dir=model_args.cache_dir
            )
        elif dataset_attr.load_from == "file":#从文件加载数据集
            data_file = os.path.join(data_args.dataset_dir, dataset_attr.file_name) # support json, jsonl and csv
            extension = dataset_attr.file_name.split(".")[-1]

            if dataset_attr.file_sha1 is not None:
                checksum(data_file, dataset_attr.file_sha1)
            else:
                logger.warning("Checksum failed: missing SHA-1 hash value in dataset_info.json.")

            raw_datasets = load_dataset(
                extension,
                data_files=data_file,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None
            )#加载数据集
        else:
            raise NotImplementedError

        dataset = raw_datasets[data_args.split]#获取数据集的划分

        if max_samples is not None:
            max_samples_temp = min(len(dataset), max_samples)#获取最大的样本数
            dataset = dataset.select(range(max_samples_temp))#选择样本

        dummy_data = [None] * len(dataset)
        for column_name, target_name in [
            ("prompt_column", "prompt"),
            ("query_column", "query"),
            ("response_column", "response"),
            ("history_column", "history")
        ]: # 每个数据集将有4列，这4列都是相同的
            if getattr(dataset_attr, column_name) != target_name:#如果不相等
                if getattr(dataset_attr, column_name):
                    dataset = dataset.rename_column(getattr(dataset_attr, column_name), target_name)#重命名列
                else: # None or empty string
                    dataset = dataset.add_column(target_name, dummy_data)#添加列
        all_datasets.append(dataset)

    if len(data_args.dataset_list) == 1:
        all_datasets = all_datasets[0]
    else:
        all_datasets = concatenate_datasets(all_datasets)#连接数据集

    return all_datasets


def preprocess_data(
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        data_args: DataTrainingArguments,
        training_args: Seq2SeqTrainingArguments,
        stage: Optional[Literal["sft", "rm", "ppo"]] = "sft"
) -> Dataset:

    column_names = list(dataset.column_names)#获取列名
    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""#获取前缀

    def format_example(examples): #支持问题有单一答案或多个答案，这部分在data/dataset_info.json已经指定了，可以看data的README
        for i in range(len(examples["prompt"])):#遍历样本
            if examples["prompt"][i] and examples["response"][i]:#如果prompt和response都不为空
                query, answer = examples["prompt"][i], examples["response"][i]#获取问题和答案
                if examples["query"][i]:
                    query += examples["query"][i]
                if examples["history"][i]:
                    prompt = ""
                    history = examples["history"][i]
                    for j, (old_query, response) in enumerate(history):
                        prompt += "[Round {}]\n问：{}\n答：{}\n".format(j, old_query, response)
                    prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
                else:
                    prompt = query
                prompt = prefix + prompt
                yield prompt, answer#返回问题和答案

    def preprocess_supervised_dataset(examples):#预处理有监督的数据集
        # build inputs with format `X [gMASK] [BOS] Y [EOS]` and labels with format `[IGNORE] ... [IGNORE] [BOS] Y [EOS]`
        model_inputs = {"input_ids": [], "labels": []}#模型输入
        for prompt, answer in format_example(examples):
            source_ids = tokenizer.encode(text=prompt, add_special_tokens=False)#编码问题
            target_ids = tokenizer.encode(text=answer, add_special_tokens=False)#编码答案

            if len(source_ids) > data_args.max_source_length - 2: # gmask and bos tokens
                source_ids = source_ids[:data_args.max_source_length - 2]
            if len(target_ids) > data_args.max_target_length - 1: # eos token
                target_ids = target_ids[:data_args.max_target_length - 1]

            input_ids = tokenizer.build_inputs_with_special_tokens(source_ids, target_ids)

            context_length = input_ids.index(tokenizer.bos_token_id)
            labels = [IGNORE_INDEX] * context_length + input_ids[context_length:]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["labels"].append(labels)
        return model_inputs

    def preprocess_evaluation_dataset(examples):#预处理评估数据集
        # build inputs with format `X [gMASK] [BOS]` and labels with format `Y [gMASK] [BOS]`
        model_inputs = {"input_ids": [], "labels": []}
        for prompt, answer in format_example(examples):
            source_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
            target_ids = tokenizer.encode(text=answer, add_special_tokens=False)

            if len(source_ids) > data_args.max_source_length - 2: # gmask and bos tokens
                source_ids = source_ids[:data_args.max_source_length - 2]
            if len(target_ids) > data_args.max_target_length - 2: # gmask and bos tokens
                target_ids = target_ids[:data_args.max_target_length - 2]

            input_ids = tokenizer.build_inputs_with_special_tokens(source_ids)
            labels = tokenizer.build_inputs_with_special_tokens(target_ids)

            model_inputs["input_ids"].append(input_ids)
            model_inputs["labels"].append(labels)
        return model_inputs

    def preprocess_pairwise_dataset(examples):#预处理成对数据集
        # 使用格式 X [gMASK] [BOS] Y1 [EOS] 和 X [gMASK] [BOS] Y2 [EOS] 构建输入对。
        model_inputs = {"accept_ids": [], "reject_ids": []}
        for prompt, answer in format_example(examples):
            source_ids = tokenizer.encode(text=prompt, add_special_tokens=False)#设置问题
            accept_ids = tokenizer.encode(text=answer[0], add_special_tokens=False)#设置接受的回答
            reject_ids = tokenizer.encode(text=answer[1], add_special_tokens=False)#设置拒绝的回答

            if len(source_ids) > data_args.max_source_length - 2: # gmask and bos tokens
                source_ids = source_ids[:data_args.max_source_length - 2]
            if len(accept_ids) > data_args.max_target_length - 1: # eos token
                accept_ids = accept_ids[:data_args.max_target_length - 1]
            if len(reject_ids) > data_args.max_target_length - 1: # eos token
                reject_ids = reject_ids[:data_args.max_target_length - 1]

            accept_ids = tokenizer.build_inputs_with_special_tokens(source_ids[:], accept_ids) # 避免复制错误
            reject_ids = tokenizer.build_inputs_with_special_tokens(source_ids[:], reject_ids)

            model_inputs["accept_ids"].append(accept_ids)
            model_inputs["reject_ids"].append(reject_ids)
        return model_inputs

    def print_sft_dataset_example(example):
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(tokenizer.decode(example["input_ids"])))
        print("label_ids:\n{}".format(example["labels"]))
        print("labels:\n{}".format(tokenizer.decode(example["labels"])))

    def print_pairwise_dataset_example(example):
        print("accept_ids:\n{}".format(example["accept_ids"]))
        print("accepts:\n{}".format(tokenizer.decode(example["accept_ids"])))
        print("reject_ids:\n{}".format(example["reject_ids"]))
        print("rejects:\n{}".format(tokenizer.decode(example["reject_ids"])))

    def print_ppo_dataset_example(example):
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(tokenizer.decode(example["input_ids"])))

    if stage == "sft":
        if (not training_args.do_train) and training_args.predict_with_generate: # with generation
            preprocess_function = preprocess_evaluation_dataset#转化为可以接受的模型输入格式
        else: # without generation
            preprocess_function = preprocess_supervised_dataset
    elif stage == "rm":
        preprocess_function = preprocess_pairwise_dataset
    elif stage == "ppo":
        preprocess_function = preprocess_evaluation_dataset

    with training_args.main_process_first(desc="dataset map pre-processing"):
        dataset = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset"
        )#传入方法以及信息，以供使用对应的解析方法

    if stage == "sft":
        print_sft_dataset_example(dataset[0])
    elif stage == "rm":
        print_pairwise_dataset_example(dataset[0])
    elif stage == "ppo":
        print_ppo_dataset_example(dataset[0])

    return dataset
