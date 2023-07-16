# 简介

这里是阿木实验室和CSR团队共同组建的基于ChatGLM&Bloom的大模型软件



主要构成有，暂定框架：

```bash
│  chatglm-prompt.py # 初版的chatgpt训练prompt
│  pdf_to_json.py # 将pdf文件拆分，并喂给chatglm
│  README.md
│  requirements.txt
│
├─bloom # bloom模型所在位置
│      __init__.py
│
├─langchain # langchain，用于加载数据库
│      __init__.py
│
├─chatglm # chatglm模型所在位置，由于版权问题，不会维护这部分内容
│      __init__.py
│
├─function # 通用功能函数，与模型输入对接
│      fuction_utils.py # 函数调用，其他fuction可以直接调用这个函数
│      web_search.py # 网络检索功能
│      __init__.py
│
├─gui_bridge # 该模型和阿木官网对接的插件渠道，暂时不用写
│      __init__.py
│
├─llm_bridge # 连接各个模型和调用的插件渠道
│      bridge_chatgpt.py # 与chatgpt的桥梁
│      llm_bridge_all.py # 所有大模型与上级的接口
│      token_calculate.py # token一些计算原则
│      __init__.py
│
├─models # 本地模型存放位置
│      __init__.py
│
└─tools # 通用工具函数，主要用于处理文件信息与可视化
        color.py # 颜色信息
        config.py # 配置文件
        check_proxy.py # 检查代理状态
        file_conversion.py # 文件转换
        get_confs.py # 配置文件读取信息
        __init__.py

```





# 参考链接

Chatglm(废弃)：

https://github.com/ssbuild/chatglm_finetuning

https://github.com/binary-husky/gpt_academic

https://github.com/GaiZhenbiao/ChuanhuChatGPT/tree/main

https://github.com/liangwq/Chatglm_lora_multi-gpu

https://github.com/hiyouga/ChatGLM-Efficient-Tuning

LLAMA：

https://github.com/project-baize/baize-chatbot

https://github.com/hiyouga/LLaMA-Efficient-Tuning

Bloom：

https://github.com/LianjiaTech/BELLE

https://github.com/yangjianxin1/Firefly

https://github.com/yuanzhoulvpi2017/zero_nlp/tree/main/chinese_bloom

https://github.com/zejunwang1/bloom_tuning

加速：

https://github.com/ztxz16/fastllm

常用数据项目：

https://github.com/chenking2020/FindTheChatGPTer

https://github.com/shibing624/MedicalGPT

https://github.com/carbonz0/alpaca-chinese-dataset

https://github.com/hikariming/alpaca_chinese_dataset

https://github.com/TigerResearch/TigerBot

https://zhuanlan.zhihu.com/p/609172950

https://www.bilibili.com/video/BV1m8411Z7xm/?buvid=YE4976EFE8199FBB493186957D993D760006&is_story_h5=false&mid=Ff%2B5uAveR4pqTkDl4NM1ig%3D%3D&p=1&plat_id=114&share_from=ugc&share_medium=iphone&share_plat=ios&share_session_id=99E0B5B6-9CF5-4A4B-AD00-459DB9DE65E4&share_source=QQ&share_tag=s_i&timestamp=1686447073&unique_k=23duzv6&up_id=29767536

