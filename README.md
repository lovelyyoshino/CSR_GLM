# 简介

这里是阿木实验室和CSR团队共同组建的基于ChatGLM的大模型软件



主要构成有，暂定框架：

```bash
│  chatglm-prompt.py # 初版的chatgpt训练prompt
│  README.md
│  requirements.txt
│
├─chatglm # chatglm模型所在位置，这里我们选择：https://github.com/ssbuild/chatglm_finetuning作为微调模型
│      __init__.py
│
├─function# 通用功能函数，与模型输入对接
│      __init__.py
│
├─gui_bridge # 该模型和阿木官网对接的插件渠道，暂时不用写
│      __init__.py
│
├─llm_bridge # 连接各个模型和调用的插件渠道
│      __init__.py
│
├─models# 本地模型存放位置
│      __init__.py
│
└─tools# 通用工具函数，主要用于处理文件信息与可视化
        color.py # 颜色信息
        config.py # 配置文件
        get_conf.py # 配置文件读取信息
        __init__.py

```





# 参考链接

https://github.com/hikariming/alpaca_chinese_dataset

https://github.com/ssbuild/chatglm_finetuning

https://github.com/binary-husky/gpt_academic