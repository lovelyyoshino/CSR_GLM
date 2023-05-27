import os
import json
from typing import List, Literal, Optional
from dataclasses import asdict, dataclass, field


CHATGLM_REPO_NAME = "THUDM/chatglm-6b"
CHATGLM_VERSION = "a8ede826cf1b62bd3c78bdfb3625c7c5d2048fbd"


@dataclass
class DatasetAttr:#数据集的参数信息，其中load_from，dataset_name，file_name，file_sha1是可选的,作为传入参数

    load_from: str#设置数据集载入的路径
    dataset_name: Optional[str] = None#数据集名字
    file_name: Optional[str] = None#数据集文件名
    file_sha1: Optional[str] = None#数据集文件的sha1值

    def __post_init__(self):#检查数据集的参数是否正确
        self.prompt_column = "instruction"
        self.query_column = "input"
        self.response_column = "output"
        self.history_column = None


@dataclass#使用dataclass装饰器，将类变成一个可变的数据类
class ModelArguments:#模型参数
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """
    model_name_or_path: Optional[str] = field(
        default=CHATGLM_REPO_NAME,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."}
    )#模型名字或者路径，其中field是dataclass的一个装饰器，用来设置默认值和帮助信息
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name."}
    )#配置文件名字或者路径
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name."}
    )#分词器名字或者路径
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co."}
    )#缓存路径
    use_fast_tokenizer: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."}
    )#是否使用快速分词器
    model_revision: Optional[str] = field(
        default=CHATGLM_VERSION,
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."}
    )#模型版本
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={"help": "Will use the token generated when running `huggingface-cli login`."}
    )#是否使用token
    quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "The number of bits to quantize the model."}
    )#量化位数
    checkpoint_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory containing the model checkpoints as well as the configurations."}
    )#  模型检查点的路径
    reward_model: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory containing the checkpoints of the reward model."}
    )#奖励模型的路径

    def __post_init__(self):
        if self.checkpoint_dir is not None: # support merging lora weights
            self.checkpoint_dir = [cd.strip() for cd in self.checkpoint_dir.split(",")]


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and evaluation.
    """
    dataset: Optional[str] = field(
        default="alpaca_zh",
        metadata={"help": "The name of provided dataset(s) to use. Use comma to separate multiple datasets."}
    )#数据集名字
    dataset_dir: Optional[str] = field(
        default="data",
        metadata={"help": "The name of the folder containing datasets."}
    )#数据集文件夹名字
    split: Optional[str] = field(
        default="train",
        metadata={"help": "Which dataset split to use for training and evaluation."}
    )#数据集划分
    overwrite_cache: Optional[bool] = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets."}
    )#是否覆盖缓存
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."}
    )#预处理的进程数
    max_source_length: Optional[int] = field(
        default=512,
        metadata={"help": "The maximum total input sequence length after tokenization."}
    )#最大输入长度
    max_target_length: Optional[int] = field(
        default=512,
        metadata={"help": "The maximum total output sequence length after tokenization."}
    )#最大输出长度
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes, truncate the number of examples for each dataset."}
    )#最大样本数
    num_beams: Optional[int] = field(
        default=None,
        metadata={"help": "Number of beams to use for evaluation. This argument will be passed to `model.generate`"}
    )#评估时使用的beam数
    ignore_pad_token_for_loss: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."}
    )#是否忽略填充标签
    source_prefix: Optional[str] = field(
        default=None,
        metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )#每个源文本前面添加的前缀
    dev_ratio: Optional[float] = field(
        default=0,
        metadata={"help": "Proportion of the dataset to include in the development set, should be between 0.0 and 1.0."}
    )#开发集比例

    def __post_init__(self): # support mixing multiple datasets
        dataset_names = [ds.strip() for ds in self.dataset.split(",")]
        dataset_info = json.load(open(os.path.join(self.dataset_dir, "dataset_info.json"), "r"))

        self.dataset_list: List[DatasetAttr] = []#数据集列表
        for name in dataset_names:#遍历数据集名字
            if name not in dataset_info:#如果数据集名字不在数据集信息中
                raise ValueError("Undefined dataset {} in dataset_info.json.".format(name))

            if "hf_hub_url" in dataset_info[name]:#如果数据集信息中有hf_hub_url
                dataset_attr = DatasetAttr("hf_hub", dataset_name=dataset_info[name]["hf_hub_url"])#数据集属性为hf_hub，这些参数需要按照DatasetAttr定义的顺序传入
            elif "script_url" in dataset_info[name]:
                dataset_attr = DatasetAttr("script", dataset_name=dataset_info[name]["script_url"])#数据集属性为script
            else:
                dataset_attr = DatasetAttr(
                    "file",
                    file_name=dataset_info[name]["file_name"],
                    file_sha1=dataset_info[name]["file_sha1"] if "file_sha1" in dataset_info[name] else None
                )#数据集属性为file

            if "columns" in dataset_info[name]:#如果数据集信息中有columns
                dataset_attr.prompt_column = dataset_info[name]["columns"].get("prompt", None)
                dataset_attr.query_column = dataset_info[name]["columns"].get("query", None)
                dataset_attr.response_column = dataset_info[name]["columns"].get("response", None)
                dataset_attr.history_column = dataset_info[name]["columns"].get("history", None)

            self.dataset_list.append(dataset_attr)#将数据集属性添加到数据集列表中


@dataclass
class FinetuningArguments:#微调参数
    """
    Arguments pertaining to which techniques we are going to fine-tuning with.
    """
    finetuning_type: Optional[Literal["none", "freeze", "p_tuning", "lora", "full"]] = field(
        default="lora",
        metadata={"help": "Which fine-tuning method to use."}
    )#微调类型
    num_layer_trainable: Optional[int] = field(
        default=3,
        metadata={"help": "Number of trainable layers for Freeze fine-tuning."}
    )#可训练层数
    name_module_trainable: Optional[Literal["mlp", "qkv"]] = field(
        default="mlp",
        metadata={"help": "Name of trainable modules for Freeze fine-tuning."}
    )#可训练模块名字
    pre_seq_len: Optional[int] = field(
        default=16,
        metadata={"help": "Number of prefix tokens to use for P-tuning V2."}
    )#前缀长度
    prefix_projection: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to add a project layer for the prefix in P-tuning V2 or not."}
    )#是否为前缀添加一个投影层
    lora_rank: Optional[int] = field(
        default=8,
        metadata={"help": "The intrinsic dimension for LoRA fine-tuning."}
    )#LoRA微调的内在维度
    lora_alpha: Optional[float] = field(
        default=32.0,
        metadata={"help": "The scale factor for LoRA fine-tuning. (similar with the learning rate)"}
    )#LoRA微调的缩放因子
    lora_dropout: Optional[float] = field(
        default=0.1,
        metadata={"help": "Dropout rate for the LoRA fine-tuning."}
    )#LoRA微调的dropout率
    lora_target: Optional[str] = field(
        default="query_key_value",
        metadata={"help": "Name(s) of target modules to apply LoRA. Use comma to separate multiple modules."}
    )#LoRA的目标模块
    resume_lora_training: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to resume training from the last LoRA weights or create new weights after merging them."}
    )#是否从最后的LoRA权重恢复训练或在合并后创建新权重
    plot_loss: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to plot the training loss after fine-tuning or not."}
    )#是否绘制微调后的训练损失

    def __post_init__(self):
        if isinstance(self.lora_target, str):#如果LoRA目标是字符串
            self.lora_target = [target.strip() for target in self.lora_target.split(",")] # 支持LoRA的自定义目标模块

        if self.num_layer_trainable > 0: # 微调最后n层如果num_layer_trainable > 0
            trainable_layer_ids = [27-k for k in range(self.num_layer_trainable)]#可训练层的id
        else: # 如果num_layer_trainable < 0，则对前n层进行微调
            trainable_layer_ids = [k for k in range(-self.num_layer_trainable)]

        if self.name_module_trainable == "mlp":#如果可训练模块名字是mlp
            self.trainable_layers = ["layers.{:d}.mlp".format(idx) for idx in trainable_layer_ids]#可训练层
        elif self.name_module_trainable == "qkv":
            self.trainable_layers = ["layers.{:d}.attention.query_key_value".format(idx) for idx in trainable_layer_ids]

        assert self.finetuning_type in ["none", "freeze", "p_tuning", "lora", "full"], "Invalid fine-tuning method."

    def save_to_json(self, json_path: str):
        """Save the content of this instance in JSON format inside `json_path`."""
        json_string = json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"#将实例的内容以json格式保存在json_path中
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)#写入json字符串

    @classmethod
    def load_from_json(cls, json_path: str):#从json_path中加载实例
        """Create an instance from the content of `json_path`."""
        with open(json_path, "r", encoding="utf-8") as f:#打开json_path
            text = f.read()#读取
        return cls(**json.loads(text))#返回实例
