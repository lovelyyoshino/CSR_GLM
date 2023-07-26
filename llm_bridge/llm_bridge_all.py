import sys
sys.path.append("../tools")
import copy
import time
import threading
from get_confs import get_conf
from token_calculate import tokenizer_gpt3_5, tokenizer_gpt4, trimmed_format_exc
from color import print亮红, print亮绿, print亮蓝, print亮黄
from concurrent.futures import ThreadPoolExecutor
from bridge_chatgpt import predict_long_connection as chatgpt_once
from bridge_chatgpt import predict as chatgpt_stream
from bridge_bloom import predict as bloom_once
from bridge_bloom import predict_long_connection as bloom_stream


def __get_token_num_gpt35(txt): return len(
    tokenizer_gpt3_5.encode(txt, disallowed_special=()))


def __get_token_num_gpt4(txt): return len(
    tokenizer_gpt4.encode(txt, disallowed_special=()))


openai_endpoint = "https://api.openai.com/v1/chat/completions"
api2d_endpoint = "https://openai.api2d.net/v1/chat/completions"
newbing_endpoint = "wss://sydney.bing.com/sydney/ChatHub"

model_type = {
    "gpt-3.5-turbo": {
        "fn_with_ui": chatgpt_stream,
        "fn_without_ui": chatgpt_once,
        "endpoint": openai_endpoint,
        "max_token": 4096,
        "tokenizer": tokenizer_gpt3_5,
        "token_cnt": __get_token_num_gpt35,
    },
    "gpt-4": {
        "fn_with_ui": chatgpt_stream,
        "fn_without_ui": chatgpt_once,
        "endpoint": openai_endpoint,
        "max_token": 8192,
        "tokenizer": tokenizer_gpt4,
        "token_cnt": __get_token_num_gpt4,
    },
    "chatglm": {
        "fn_with_ui": bloom_stream,
        "fn_without_ui": bloom_once,
        "endpoint": None,
        "max_token": 1024,
        "tokenizer": tokenizer_gpt3_5,
        "token_cnt": __get_token_num_gpt35,
    }
}


def __LLM_CATCH_EXCEPTION(f):
    """
    装饰器函数，将错误显示出来
    """
    def decorated(inputs, llm_kwargs, history, sys_prompt, console_slience):
        try:
            return f(inputs, llm_kwargs, history, sys_prompt, console_slience)
        except Exception as e:
            tb_str = '\n```\n' + trimmed_format_exc() + '\n```\n'
            print亮黄(tb_str)
            return tb_str
    return decorated


"""
    发送至LLM，等待回复，一次性完成，不显示中间过程。但内部用stream的方法避免中途网线被掐。
    inputs：
        是本次问询的输入
    sys_prompt:
        系统静默prompt
    llm_kwargs：
        LLM的内部调优参数
    history：
        是之前的对话列表
"""


def chat_multiple_with_pre_chat(inputs, llm_kwargs, history, sys_prompt, console_slience=False,llm_models=None):

    model = llm_kwargs['llm_model']  # 获取待使用模型
    if len(model) == 0:  # 如果没有模型
        print亮黄("LLM模型为基础的ChatGLM模型,必须添加")
        model = "chatglm"  # 那直接使用chatglm模型
    models = model.split('&')
    models = [item for item in models if item != "chatglm"] # 删除元素"orange"
    models.append("chatglm")  # 加入chatglm，这是默认模型
    n_model = len(models)
    futures = []
    return_string_collect = []
    if n_model >1:
        executor = ThreadPoolExecutor(max_workers=n_model-1)
        for i in range(n_model):
            model_t = models[i]
            if "chatglm" in model_t:
                print亮黄("LLM模型含有基础的ChatGLM模型,最后会综述")
                # 载入ChatGLM模型，因为载入比较费时这部分已经在主函数里面做了
                continue
            llm_kwargs_feedin = copy.deepcopy(llm_kwargs)
            llm_kwargs_feedin['llm_model'] = model_t
            print亮绿(f"正在使用{model_t}模型")
            method = model_type[model_t]["fn_without_ui"]
            future = executor.submit(__LLM_CATCH_EXCEPTION(
                method), inputs, llm_kwargs_feedin, history, sys_prompt, console_slience)
            futures.append(future)
        while True:
            worker_done = [h.done() for h in futures]
            if all(worker_done):
                executor.shutdown()
                break
            time.sleep(1)

        for i, future in enumerate(futures):  # wait and get
            return_string_collect.append(future.result())
            # return_string_collect.append(
            #     f"【{str(models[i])} 说】: {future.result()} </font>")
    llm_kwargs_feedin = copy.deepcopy(llm_kwargs)
    llm_kwargs_feedin['llm_model'] = "chatglm"
    # 这个时候会由chatglm总结，这时候需要重新设计sys_prompt
    return_string_collect = predict(inputs, llm_kwargs, history=return_string_collect, sys_prompt="你是一个slam专家，请根据上面的内容组合总结并翻译下面的短语",llm_models = llm_models)
    return return_string_collect

"""
    发送至LLM，流式获取输出。
    用于基础的对话功能。
    inputs 是本次问询的输入
    top_p, temperature是LLM的内部调优参数
    history 是之前的对话列表（注意无论是inputs还是history，内容太长了都会触发token数量溢出的错误）
    chatbot 为WebUI中显示的对话列表，修改它，然后yeild出去，可以直接修改对话界面内容
    additional_fn代表点击的哪个按钮，按钮见functional.py
"""
def predict(inputs, llm_kwargs, *args, **kwargs):
    method = model_type[llm_kwargs['llm_model']]["fn_with_ui"]
    return method(inputs, llm_kwargs, *args, **kwargs)


if __name__ == "__main__":
    sys.path.append("../../LLaMA-Efficient-Tuning/src")
    from llmtuner import ChatModel
    from llmtuner.tuner import get_infer_args
    proxies, LLM_MODEL, API_KEY = get_conf('proxies', 'LLM_MODEL',  'API_KEY')
    llm_kwargs = {
        'api_key': API_KEY,
        'llm_model': LLM_MODEL,
        'top_p': 1.0,
        'max_length': None,
        'temperature': 1.0,
    }
    chat_model = None
    sys.argv=['bridge_bloom.py', '--dataset_dir', '../data', '--model_name_or_path', '/home/amov/LLaMA-Efficient-Tuning/model/bloom', '--quantization_bit', '8']
    print("get_infer_args",*get_infer_args())
    device, = get_conf('LOCAL_MODEL_DEVICE')
    if device=='cuda':
        chat_model =  ChatModel(*get_infer_args())
    print(chat_multiple_with_pre_chat("你好", llm_kwargs, [], "",llm_models =chat_model))
    
    llm_kwargs = {
        'api_key': API_KEY,
        'llm_model': "chatglm",
        'top_p': 1.0,
        'max_length': None,
        'temperature': 1.0,
    }
    print(chat_multiple_with_pre_chat("你好", llm_kwargs, [], "",llm_models =chat_model))
