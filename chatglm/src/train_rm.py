# coding=utf-8
# Implements parameter-efficient training of a reward model based on ChatGLM.
# This code is inspired by:
# https://github.com/lvwerra/trl/blob/main/examples/summarization/scripts/reward_summarization.py
# https://github.com/CarperAI/trlx/blob/main/examples/summarize_rlhf/reward_model/train_reward_model_gptj.py


from utils import (
    prepare_args,
    prepare_data,
    load_pretrained,
    preprocess_data,
    PairwiseDataCollatorForChatGLM,
    PairwiseTrainerForChatGLM,
    plot_loss
)

def main():

    # 准备预训练模型和数据集
    model_args, data_args, training_args, finetuning_args = prepare_args()
    dataset = prepare_data(model_args, data_args)#预处理图像，在common中
    model, tokenizer = load_pretrained(model_args, training_args, finetuning_args, training_args.do_train, stage="rm")#加载预训练模型
    dataset = preprocess_data(dataset, tokenizer, data_args, training_args, stage="rm")#预处理rm模型的输入
    data_collator = PairwiseDataCollatorForChatGLM(tokenizer, model.pretrained_model)#将输入的数据集生成两种例子，分别是接受的例子和拒绝的例子

    training_args.remove_unused_columns = False # 重要的对于成对数据集

    # 将数据集拆分
    if training_args.do_train:#如果是训练阶段
        if data_args.dev_ratio > 1e-6:#如果有验证集
            dataset = dataset.train_test_split(test_size=data_args.dev_ratio)#将数据集拆分成训练集和验证集
            trainer_kwargs = {"train_dataset": dataset["train"], "eval_dataset": dataset["test"]}#训练集和验证集
        else:
            trainer_kwargs = {"train_dataset": dataset}#只有训练集
    else: # do_eval或do_predict
        trainer_kwargs = {"eval_dataset": dataset}

    # 初始化我们的训练器
    trainer = PairwiseTrainerForChatGLM(
        finetuning_args=finetuning_args,
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        **trainer_kwargs
    )

    # 训练参数
    if training_args.do_train:
        train_result = trainer.train()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        trainer.save_model()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args, keys=["loss", "eval_loss"])

    # 评价参数
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
