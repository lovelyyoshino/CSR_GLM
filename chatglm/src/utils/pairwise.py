import torch
from typing import Dict, Sequence, Union

from .data_collator import DataCollatorForChatGLM

from .peft_trainer import PeftTrainer

from .other import get_logger

logger = get_logger(__name__)


class PairwiseDataCollatorForChatGLM(DataCollatorForChatGLM):
    r"""
    成对数据的数据整理器
    """

    def __call__(self, features: Sequence[Dict[str, Union[torch.Tensor, Sequence[int]]]]) -> Dict[str, torch.Tensor]:
        r"""
        将批处理数据填充到批处理中最长的序列。

        我们生成2 * n个示例，其中前n个示例表示选择的示例和
        最后n个例子代表被拒绝的例子。
        """
        features = [{"input_ids": feature[key]} for key in ("accept_ids", "reject_ids") for feature in features]
        return super().__call__(features)


class PairwiseTrainerForChatGLM(PeftTrainer):
    r"""
    继承PeftTrainer来计算成对损失
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.can_return_loss = True # override property to return eval_loss

    def compute_loss(self, model, inputs, return_outputs=False):
        r"""
        计算成对损失。选择前n个样本，拒绝后n个样本。

        我们使用EOS token上的分数来表示整个句子的奖励。

        子类化并覆盖以注入自定义行为。它不应该被外部脚本直接使用。
        """
        batch_size = inputs["input_ids"].size(0) // 2
        _, _, values = model(**inputs)
        r_accept, r_reject = values[-1].split(batch_size, dim=0)
        loss = -torch.log(torch.sigmoid(r_accept - r_reject)).mean()
        outputs = {"r_accept": r_accept, "r_reject": r_reject}
        return (loss, outputs) if return_outputs else loss
