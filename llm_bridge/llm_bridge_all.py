import threading
import time
import copy
from concurrent.futures import ThreadPoolExecutor
from tools.get_conf import get_conf
from token_calculate import tokenizer_gpt3_5, tokenizer_gpt4, trimmed_format_exc
from tools.color import print亮红, print亮绿, print亮蓝, print亮黄


def __get_token_num_gpt35(txt): return len(
    tokenizer_gpt3_5.encode(txt, disallowed_special=()))


def __get_token_num_gpt4(txt): return len(
    tokenizer_gpt4.encode(txt, disallowed_special=()))


openai_endpoint = "https://api.openai.com/v1/chat/completions"
api2d_endpoint = "https://openai.api2d.net/v1/chat/completions"
newbing_endpoint = "wss://sydney.bing.com/sydney/ChatHub"

model_type = {
    "gpt-3.5-turbo": {
        # "fn_with_ui": chatgpt_ui,
        # "fn_without_ui": chatgpt_noui,
        "endpoint": openai_endpoint,
        "max_token": 4096,
        "tokenizer": tokenizer_gpt3_5,
        "token_cnt": __get_token_num_gpt35,
    },
    "gpt-4": {
        # "fn_with_ui": chatgpt_ui,
        # "fn_without_ui": chatgpt_noui,
        "endpoint": openai_endpoint,
        "max_token": 8192,
        "tokenizer": tokenizer_gpt4,
        "token_cnt": __get_token_num_gpt4,
    }
}


def __LLM_CATCH_EXCEPTION(f):
    """
    装饰器函数，将错误显示出来
    """
    def decorated(inputs, llm_kwargs, history, sys_prompt, observe_window, console_slience):
        try:
            return f(inputs, llm_kwargs, history, sys_prompt, observe_window, console_slience)
        except Exception as e:
            tb_str = '\n```\n' + trimmed_format_exc() + '\n```\n'
            observe_window[0] = tb_str
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


def chat_multiple_with_loog_connection(inputs, llm_kwargs, history, sys_prompt, console_slience=False):

    model = llm_kwargs['llm_model']
    if len(model) == 0:
        print亮黄("LLM模型为基础的ChatGLM模型")
        model = "chatglm"
    models = model.split('&')
    n_model = len(models)
    for i in range(n_model):
        model = models[i]
        if "chatglm" in model:
            print亮黄("LLM模型含有基础的ChatGLM模型")
            # 载入ChatGLM模型
            break

    executor = ThreadPoolExecutor(max_workers=4)
    models = model.split('&')
    n_model = len(models)

    window_mutex = [["", time.time(), ""] for _ in range(n_model)] + [True]
    futures = []
    for i in range(n_model):
        model = models[i]
        if "chatglm" in model:
            print亮黄("LLM模型含有基础的ChatGLM模型")
            # 载入ChatGLM模型
            break
        method = model_type[model]["fn_without_ui"]
        llm_kwargs_feedin = copy.deepcopy(llm_kwargs)
        llm_kwargs_feedin['llm_model'] = model
        future = executor.submit(__LLM_CATCH_EXCEPTION(
            method), inputs, llm_kwargs_feedin, history, sys_prompt, window_mutex[i], console_slience)
        futures.append(future)
    return_string_collect = []
    while True:
        worker_done = [h.done() for h in futures]
        if all(worker_done):
            executor.shutdown()
            break
        time.sleep(1)

    for i, future in enumerate(futures):  # wait and get
        return_string_collect.append(
            f"【{str(models[i])} 说】: {future.result()} </font>")


if __name__ == "__main__":
    proxies, LLM_MODEL,  API_KEY = \
        get_conf('proxies', 'LLM_MODEL',  'API_KEY')
    llm_kwargs = {
        'api_key': API_KEY,
        'llm_model': LLM_MODEL,
        'top_p': 1.0,
        'max_length': None,
        'temperature': 1.0,
    }
