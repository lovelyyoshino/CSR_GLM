# CHAT-GPT api。可同时填写多个API-KEY，用英文逗号分割，例如API_KEY = "sk-openaikey1,sk-openaikey2,fkxxxx-api2dkey1,fkxxxx-api2dkey2"
API_KEY = "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,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，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"

# CHAT-GPT需要使用代理，所以将USE_PROXY改为True则是应用代理，如果直接在海外服务器部署，此处不修改
USE_PROXY = True
if USE_PROXY:
    # 填写格式是 [协议]://  [地址] :[端口]，填写之前不要忘记把USE_PROXY改成True，如果直接在海外服务器部署，此处不修改
    # 例如    "socks5h://localhost:11284"
    # [协议] 常见协议无非socks5h/http; 例如 v2**y 和 ss* 的默认本地协议是socks5h; 而cl**h 的默认本地协议是http
    # [地址] 懂的都懂，不懂就填localhost或者127.0.0.1肯定错不了（localhost意思是代理软件安装在本机上）
    # [端口] 在代理软件的设置里找。虽然不同的代理软件界面不一样，但端口号都应该在最显眼的位置上

    # 代理网络的地址，打开你的*学*网软件查看代理的协议(socks5/http)、地址(localhost)和端口(11284)
    proxies = {
        #          [协议]://  [地址]  :[端口]
        "http":  "http://localhost:1080",  # 再例如  "http":  "socks5h://127.0.0.1:7890",
        "https": "http://localhost:1080",  # 再例如  "https": "socks5h://127.0.0.1:7890",
    }
else:
    proxies = None

# CHAT-GPT 使用的多线程，在多线程函数插件中，默认允许多少路线程同时访问OpenAI。Free trial users的限制是每分钟3次，Pay-as-you-go users的限制是每分钟3500次
# 一言以蔽之：免费用户填3，OpenAI绑了信用卡的用户可以填 16 或者更高。提高限制请查询：https://platform.openai.com/docs/guides/rate-limits/overview
DEFAULT_WORKER_NUM = 3

# 发送请求到OpenAI后，等待多久判定为超时
TIMEOUT_SECONDS = 30

# 如果OpenAI不响应（网络卡顿、代理失败、KEY失效），重试的次数限制
MAX_RETRY = 2

# OpenAI模型选择是（gpt4现在只对申请成功的人开放，体验gpt-4可以试试api2d）
LLM_MODEL = "gpt-3.5-turbo" # 可选 "gpt-3.5-turbo&chatglm"
AVAIL_LLM_MODELS = ["gpt-3.5-turbo", "api2d-gpt-3.5-turbo", "gpt-4", "api2d-gpt-4", "chatglm", "newbing"]


# 本地LLM模型如ChatGLM的执行方式 CPU/GPU
LOCAL_MODEL_DEVICE = "cuda" # 可选 "cuda"


# if __name__ == "__main__":
#     print("本文件是CHAT-GPT的配置文件，请勿直接运行。这里仅做测试，这里不建议修改相关的内容")
#     print("本地LLM模型如ChatGLM的执行方式 CPU/GPU:", LOCAL_MODEL_DEVICE)
#     LOCAL_MODEL_DEVICE = "cuda"
#     print("本地LLM模型如ChatGLM的执行方式 CPU/GPU: %s" %LOCAL_MODEL_DEVICE)
    