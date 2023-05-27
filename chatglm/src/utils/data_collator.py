import torch

from typing import Dict, Optional, Sequence, Union

from transformers import DataCollatorWithPadding
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

from .other import IGNORE_INDEX


class DataCollatorForChatGLM(DataCollatorWithPadding):
    r"""
    ChatGLM的数据整理器。它能够动态填充批量数据
    """
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            model: PreTrainedModel,
            ignore_pad_token_for_loss: Optional[bool] = False
    ):#这个函数传入了tokenizer这个分词器，model这个模型，ignore_pad_token_for_loss这个参数默认为False
        super().__init__(tokenizer, padding=True)
        self.model = model
        self.label_pad_token_id = IGNORE_INDEX if ignore_pad_token_for_loss else tokenizer.pad_token_id

    def get_attention_masks(self, input_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
        r"""
        为左填充序列生成注意力掩码。
        请注意，ChatGLM在注意掩码中为要参与的令牌分配False。在一般情况下，它应该是正确的。
        根据:https://huggingface.co/THUDM/chatglm-6b/blob/v1.1.0/modeling_chatglm.py#L680
        """
        batch_size, seq_length = input_ids.size()#batch_size是批量大小，seq_length是序列长度
        attention_mask = torch.ones((batch_size, seq_length, seq_length), device=device)#attention_mask是注意力掩码，torch.ones是生成全1的张量
        attention_mask.tril_()#tril_()是下三角矩阵
        for i, seq in enumerate(input_ids):#enumerate是枚举函数，i是序号，seq是序列
            attention_mask[i, :, :(seq == self.tokenizer.bos_token_id).nonzero()[0].item()] = 1 # context
            attention_mask[i, :, :(seq != self.tokenizer.pad_token_id).nonzero()[0].item()] = 0 # padding
        attention_mask.unsqueeze_(1)#unsqueeze_()是增加维度
        attention_mask = (attention_mask < 0.5).bool()#然后将将掩码小于0.5的转换为布尔值
        return attention_mask

    def get_position_ids(self, input_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
        r"""
        为左填充序列生成位置id。

        根据:https://huggingface.co/THUDM/chatglm-6b/blob/v1.1.0/modeling_chatglm.py#L69
        """
        batch_size, seq_length = input_ids.size()#batch_size是批量大小，seq_length是序列长度
        mask: int = self.model.config.mask_token_id#mask是掩码
        gmask: int = self.model.config.gmask_token_id#gmask是全局掩码
        position_ids = torch.zeros((batch_size, seq_length), dtype=torch.long, device=device)#position_ids是位置id
        block_position_ids = torch.zeros((batch_size, seq_length), dtype=torch.long, device=device)#block_position_ids是块位置id
        for i, seq in enumerate(input_ids):
            mask_token = gmask if gmask in seq else mask
            context_length = (seq == self.tokenizer.bos_token_id).nonzero()[0].item()
            padding_length = (seq != self.tokenizer.pad_token_id).nonzero()[0].item()
            position_ids[i, padding_length:] = torch.arange(seq_length - padding_length, dtype=torch.long, device=device)
            if self.model.position_encoding_2d or (mask_token != gmask): # 2d position encoding or not gMASK
                position_ids[i, context_length:] = (seq == mask_token).nonzero()[0].item() - padding_length # mask position
            block_position_ids[i, context_length:] = torch.arange(seq_length - context_length, dtype=torch.long, device=device) + 1
        if self.model.position_encoding_2d:
            position_ids = torch.stack((position_ids, block_position_ids), dim=1)
        return position_ids

    def __call__(self, features: Sequence[Dict[str, Union[torch.Tensor, Sequence[int]]]]) -> Dict[str, torch.Tensor]:
        r"""
        将批处理数据填充到批处理中最长的序列。

        我们在训练和评估中都采用了左填充。
        """
        if isinstance(features[0]["input_ids"], torch.Tensor):
            input_ids = [feature["input_ids"].clone().detach().flip(0) for feature in features]
        else:
            input_ids = [torch.tensor(feature["input_ids"]).flip(0) for feature in features]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id).flip(-1)
        batch = {
            "input_ids": input_ids,
            "attention_mask": self.get_attention_masks(input_ids, device=input_ids.device),
            "position_ids": self.get_position_ids(input_ids, device=input_ids.device)
        }
        if "labels" in features[0]:
            if isinstance(features[0]["labels"], torch.Tensor):
                labels = [feature["labels"].clone().detach().flip(0) for feature in features]
            else:
                labels = [torch.tensor(feature["labels"]).flip(0) for feature in features]
            batch["labels"] = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.label_pad_token_id).flip(-1)
        return batch
