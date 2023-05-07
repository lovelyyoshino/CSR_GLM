# 简介

这里是阿木实验室和CSR团队共同组建的基于ChatGLM的大模型软件



主要构成有，暂定框架：

```bash
├─bridge # 连接各个模型和上面调用的插件渠道
├─chatglm # chatglm模型所在位置，这里我们选择：https://github.com/ssbuild/chatglm_finetuning作为微调模型
├─function # 通用功能函数
├─models # 本地模型存放位置
└─tools # 通用工具函数

```





# 参考链接

https://github.com/hikariming/alpaca_chinese_dataset

https://github.com/ssbuild/chatglm_finetuning

https://github.com/binary-husky/gpt_academic