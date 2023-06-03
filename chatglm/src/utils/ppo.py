import os
import math
import torch
from tqdm import tqdm
from typing import Callable, Dict, List, Literal, Optional, Tuple

from transformers import Seq2SeqTrainingArguments
from transformers.trainer import TrainerState
from transformers.modeling_utils import PreTrainedModel

from trl import PPOTrainer, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from trl.trainer.ppo_trainer import PPODecorators, logprobs_from_logits

from .peft_trainer import PeftTrainer

from .config import FinetuningArguments

from .other import (
    AverageMeter,
    get_logger,
    get_logits_processor
)


logger = get_logger(__name__)

'''
替换模型的valuehead
'''
def replace_model(model: AutoModelForCausalLMWithValueHead, target: Literal["default", "reward"]) -> None:
    if target == "reward": # save original head temporarily
        valuehead_state_dict = model.v_head.state_dict()#获取valuehead的权重

        setattr(model, "origin_head_weight", valuehead_state_dict["summary.weight"])#保存权重
        setattr(model, "origin_head_bias", valuehead_state_dict["summary.bias"])#保存偏置

    model.pretrained_model.set_adapter(target) # 将LoRA适配器设置为活动状态
    model.v_head.load_state_dict({
        "summary.weight": getattr(model, "{}_head_weight".format(target)),
        "summary.bias": getattr(model, "{}_head_bias".format(target))
    })

'''
将模型的valuehead恢复为原始的valuehead
'''
def cast_layernorm_dtype(
        model: AutoModelForCausalLMWithValueHead,
        layer_norm_names: List[str] = ["layernorm"], # 用于chatglm设置
        layer_norm_params: Optional[Dict[str, torch.Tensor]] = None
) -> Tuple[AutoModelForCausalLMWithValueHead, Dict[str, torch.Tensor]]:

    layer_norm_state_dict = {}

    for name, param in model.named_parameters():
        if param.ndim == 1 and any(layer_norm_name in name for layer_norm_name in layer_norm_names):
            if layer_norm_params is not None:
                param.data = layer_norm_params[name] # 恢复float32权重
            else:
                layer_norm_state_dict[name] = param.data.detach().clone() # 存储float32类型变量的权重以保持稳定性
                param.data = param.data.to(torch.float16)

    return model, layer_norm_state_dict


class PPOTrainerForChatGLM(PPOTrainer, PeftTrainer):
    r"""
    继承了PPOTrainer。
    """

    def __init__(self, training_args: Seq2SeqTrainingArguments, finetuning_args: FinetuningArguments, **kwargs):
        PPOTrainer.__init__(self, **kwargs)
        self.args = training_args
        self.finetuning_args = finetuning_args
        self.state = TrainerState()
        self.data_collator = self.accelerator.prepare(kwargs["data_collator"])

    def ppo_train(self, max_target_length: int) -> None:
        r"""
        实现PPO阶段的训练循环，如Huggingface的训练器中的_inner_training_loop()。
        """
        total_train_batch_size = self.config.batch_size * self.config.gradient_accumulation_steps * self.args.world_size# 计算总的训练批次大小
        len_dataloader = len(self.dataloader)
        num_steps_per_epoch = max(len_dataloader // self.config.gradient_accumulation_steps, 1)# 计算每个epoch的步数
        num_examples = len(self.dataset)# 计算数据集的大小
        num_train_epochs = self.args.num_train_epochs# 训练的epoch数
        max_steps = math.ceil(num_train_epochs * num_steps_per_epoch)# 计算最大步数

        if self.is_world_process_zero():# 如果是主进程
            logger.info("***** Running training *****")
            logger.info(f"  Num examples = {num_examples}")
            logger.info(f"  Num Epochs = {num_train_epochs}")
            logger.info(f"  Instantaneous batch size per device = {self.config.batch_size}")
            logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
            logger.info(f"  Gradient Accumulation steps = {self.config.gradient_accumulation_steps}")
            logger.info(f"  Total optimization steps = {max_steps}")
            logger.info(f"  Number of trainable parameters = {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")

        # model.generate 的关键字参数
        gen_kwargs = {
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "logits_processor": get_logits_processor()
        }
        output_length_sampler = LengthSampler(max_target_length // 2, max_target_length)# 用于生成长度的采样器
        unwrapped_model: PreTrainedModel = self.accelerator.unwrap_model(self.model)# 获取模型

        dataiter = iter(self.dataloader)# 获取数据迭代器
        steps_trained = 0
        loss_meter = AverageMeter()# 用于计算平均损失
        reward_meter = AverageMeter()# 用于计算平均奖励

        for step in tqdm(range(max_steps), disable=not self.is_world_process_zero()):

            for _ in range(self.config.gradient_accumulation_steps):

                batch = next(dataiter)
                steps_trained += 1

                unwrapped_model.gradient_checkpointing_disable()# 禁用梯度检查点
                unwrapped_model.config.use_cache = True# 使用缓存

                # 从ChatGLM获取回复
                query_tensors: torch.Tensor = batch["input_ids"]
                response_tensors = self.generate(batch, length_sampler=output_length_sampler, return_prompt=False, **gen_kwargs)# 生成回复

                queries: List[torch.Tensor] = []
                responses: List[torch.Tensor] = []
                for i in range(len(query_tensors)):
                    query_length = (query_tensors[i] != self.tokenizer.pad_token_id).nonzero()[0]# 获取query的长度
                    response_length = (response_tensors[i] != self.tokenizer.pad_token_id).nonzero()[-1] + 1# 获取response的长度
                    queries.append(query_tensors[i, query_length:]) # 去除左边的填充
                    if response_length < 2: # 使响应至少具有2个标记
                        responses.append(response_tensors.new_empty(2).fill_(self.tokenizer.eos_token_id))
                    else:
                        responses.append(response_tensors[i, :response_length]) #去除右边的填充

                # 计算奖励
                replace_model(unwrapped_model, target="reward")# 替换模型
                _, _, values = self.model(**self.prepare_model_inputs(queries, responses))# 获取奖励
                rewards = [reward for reward in values[-1]]
                replace_model(unwrapped_model, target="default") # 确保模型在最后处于默认状态
                # 跑PPO模型
                unwrapped_model.gradient_checkpointing_enable()
                unwrapped_model.config.use_cache = False

                stats = self.step(queries, responses, rewards)#根据奖励，模型输出状态

                loss_meter.update(stats["ppo/loss/total"])#将损失添加到损失计量器中
                reward_meter.update(torch.tensor(rewards).sum().item(), n=len(rewards))# 将奖励添加到奖励计量器中

                if steps_trained == len_dataloader:
                    dataiter = iter(self.dataloader)# 重置数据迭代器
                    steps_trained = 0

            if self.is_world_process_zero() and (step+1) % self.args.logging_steps == 0:# 打印日志
                logs = {
                    "loss": round(loss_meter.avg, 4),
                    "reward": round(reward_meter.avg, 4),
                    "learning_rate": stats["ppo/learning_rate"],
                    "epoch": round(step / num_steps_per_epoch, 2)
                }
                print(logs)
                logs["step"] = step
                self.state.log_history.append(logs)
                loss_meter.reset()
                reward_meter.reset()

            if (step+1) % self.args.save_steps == 0: # save checkpoint
                self.save_model(os.path.join(self.args.output_dir, f"checkpoint-{step+1}"))

    @torch.no_grad()
    def generate(
            self,
            inputs: Dict[str, torch.Tensor],
            length_sampler: Callable = None,
            return_prompt: bool = True,
            **generation_kwargs,
    ) -> torch.Tensor:
        r"""
        生成给定查询的模型响应。
        子类化并覆盖以注入自定义行为。
        """
        self.model, layer_norm_params = cast_layernorm_dtype(self.model)# 将模型转换为半精度

        if length_sampler is not None:
            generation_kwargs["max_new_tokens"] = length_sampler()# 生成最大长度

        unwrapped_model = self.accelerator.unwrap_model(self.model)# 获取模型

        response = unwrapped_model.generate(**inputs, **generation_kwargs)# 生成回复

        # 临时hack，以确保生成配置不会在评估循环的每个迭代中初始化
        #灵感来源:https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/trainer_seq2seq.py#L273
        if unwrapped_model.pretrained_model.generation_config._from_model_config:
            unwrapped_model.pretrained_model.generation_config._from_model_config = False# 重置生成配置

        self.model, _ = cast_layernorm_dtype(self.model, layer_norm_params)# 将模型转换为全精度

        if not return_prompt and not self.is_encoder_decoder:# 如果不返回提示并且不是编码器解码器
            return response[:, inputs["input_ids"].size(1):]# 返回回复
        return response

    def prepare_model_inputs(self, queries: List[torch.Tensor], responses: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.cat([q, r]) for q, r in zip(queries, responses)]# 将查询和响应连接起来
        input_data = self.data_collator([{"input_ids": ids} for ids in input_ids])# 将输入数据转换为模型输入
        input_data = {k: v.to(self.current_device) for k, v in input_data.items() if v is not None}# 将输入数据移动到当前设备
        input_data.pop("labels", None)  # 我们不想计算LM损失
        return input_data

    @PPODecorators.empty_cuda_cache()
    def batched_forward_pass(
        self,
        model: AutoModelForCausalLMWithValueHead,
        queries: torch.Tensor,
        responses: torch.Tensor,
        model_inputs: dict,
    ):
        r"""
        计算多个批次的模型输出。

        子类化并覆盖以注入自定义行为。
        """
        bs = len(model_inputs["input_ids"])# 获取批次大小
        fbs = self.config.mini_batch_size# 获取前向批次大小
        all_logprobs = []
        all_logits = []
        all_masks = []
        all_values = []

        for i in range(int(bs / fbs)):
            input_kwargs = {k: v[i * fbs : (i + 1) * fbs] for k, v in model_inputs.items()}# 获取输入参数
            input_ids: torch.Tensor = input_kwargs["input_ids"] # 左填充序列
            if self.is_distributed: # 重新生成它们以适应填充输入
                input_kwargs["attention_mask"] = self.data_collator.get_attention_masks(input_ids, device=self.current_device)# 注意力掩码
                input_kwargs["position_ids"] = self.data_collator.get_position_ids(input_ids, device=self.current_device)# 位置id
            logits, _, values = model(**input_kwargs)# 获取模型输出
            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])# 获取对数概率

            values = values.transpose(0, 1)
            masks = torch.zeros_like(input_ids)# 创建掩码

            for j in range(fbs):
                start = (input_ids[j] == self.tokenizer.bos_token_id).nonzero()[0].item()# 获取开始位置
                masks[j][start:] = 1
                if len(masks[j][start:]) < 2:# 如果掩码长度小于2
                    raise ValueError("Responses are too short. Make sure they are at least 4 tokens long.")

            all_logits.append(logits)# 添加logits
            all_values.append(values)# 添加值
            all_logprobs.append(logprobs)# 添加对数概率
            all_masks.append(masks)# 添加掩码

        return (
            torch.cat(all_logprobs),
            torch.cat(all_logits)[:, :-1],
            torch.cat(all_values)[:, :-1],
            torch.cat(all_masks)[:, :-1],
        )

    def save_model(self, output_dir: Optional[str] = None) -> None:
        r"""
        保存模型检查点。
        子类化并覆盖以注入自定义行为。
        """
        if self.args.should_save:
            self._save(output_dir)
