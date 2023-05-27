import os
import torch
from typing import Dict, Optional

from transformers import Seq2SeqTrainer
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.modeling_utils import unwrap_model

from peft.utils.other import WEIGHTS_NAME

from .config import FinetuningArguments

from .other import (
    get_logger,
    get_state_dict,
    load_trainable_params,
    load_valuehead_params,
    FINETUNING_ARGS_NAME,
    VALUE_HEAD_FILE_NAME
)


logger = get_logger(__name__)


class PeftTrainer(Seq2SeqTrainer):
    r"""
    继承Seq2SeqTrainer以支持参数高效的检查点
    """

    def __init__(self, finetuning_args: FinetuningArguments, **kwargs):
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args

    def _save(self, output_dir: Optional[str] = None, state_dict: Optional[Dict[str, torch.Tensor]] = None) -> None:
        r"""
        将可训练参数保存为模型检查点。

        此函数仅在进程零中执行。

        子类和覆盖以注入自定义行为。它不应直接被外部脚本使用。
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        model = unwrap_model(self.model)

        if hasattr(model, "pretrained_model"): # for models with valuehead
            backbone_model = getattr(model, "pretrained_model")
        else:
            backbone_model = model

        if hasattr(backbone_model, "peft_config"): # peft methods
            backbone_model.save_pretrained(output_dir, state_dict=get_state_dict(backbone_model)) # save lora weights
        else:
            torch.save(get_state_dict(backbone_model), os.path.join(output_dir, WEIGHTS_NAME)) # save trainable weights

        if hasattr(model, "v_head"): # save valuehead weights
            torch.save(get_state_dict(getattr(model, "v_head")), os.path.join(output_dir, VALUE_HEAD_FILE_NAME))

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        self.finetuning_args.save_to_json(os.path.join(output_dir, FINETUNING_ARGS_NAME))

    def _load_best_model(self):
        r"""
        从模型检查点加载可训练参数。

        子类化并覆盖以注入自定义行为。它不应该被外部脚本直接使用。
        """
        logger.info(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")
        model = unwrap_model(self.model)
        if hasattr(model, "peft_config"): # peft methods
            model.load_adapter(self.state.best_model_checkpoint, getattr(model, "active_adapter"))
        else:
            load_trainable_params(model, self.state.best_model_checkpoint)

        if hasattr(model, "v_head"):
            load_valuehead_params(model, self.state.best_model_checkpoint)
