from transformers import AutoModel, AutoTokenizer
import time
import threading
import importlib
import sys
from multiprocessing import Process, Pipe
sys.path.append("../tools")
from token_calculate import trimmed_format_exc, clip_history
from get_confs import get_conf, select_api_key, what_keys
from color import print亮红, print亮绿, print亮蓝, print亮黄
from check_proxy import check_proxy
from core_functional import get_core_functions


class GetBloomhandle(Process):
    def __init__(self):
        super().__init__(daemon=True)
        self.parent, self.child = Pipe()
        self.bloom_model = None
        self.start()
        self.threadLock = threading.Lock()

    def ready(self):
        return self.bloom_model is not None

    def chat_get(self,chat_model):
        self.bloom_model = chat_model

    def run(self):
        while True:
            print亮蓝(type(self.bloom_model))
            if self.ready():
                # 进入任务等待状态
                kwargs = self.child.recv()#使用pipe来处理消息
                print亮蓝('[Local Message] Call Bloom with kwargs: ')
                # 收到消息，开始请求
                try:
                    print亮蓝(kwargs['query'],kwargs['history'],kwargs['input_kwargs'])
                    for new_text in self.bloom_model.stream_chat(**kwargs):
                        self.child.send(new_text)
                        #response += new_text
                        # 中途接收可能的终止指令（如果有的话）
                        if self.child.poll(): 
                            command = self.child.recv()
                            if command == '[Terminate]': break
                except:
                    self.child.send('[Local Message] Call Bloom fail.' + '\n```\n' + trimmed_format_exc() + '\n```\n')
                # 请求处理结束，开始下一个循环
                self.child.send('[Finish]')
            else:
                time.sleep(1)

    def stream_chats(self, **kwargs):
        # 主进程执行
        self.threadLock.acquire()
        self.parent.send(kwargs)
        while True:
            res = self.parent.recv()
            if res != '[Finish]':
                yield res
            else:
                break
        self.threadLock.release()


#################################################################################
def predict_long_connection(inputs, llm_kwargs, history=[], sys_prompt="", console_slience=False,llm_models=None):
    """
        多线程方法
        函数的说明请见 request_llm/bridge_all.py
    """
    # if model is None:
    #     model = GetBloomhandle()#实例化handle

    # bloom 没有 sys_prompt 接口，因此把prompt加入 history
    history_feedin = []
    history_feedin.append(["What can I do?", sys_prompt])
    for i in range(len(history)//2):
        history_feedin.append([history[2*i], history[2*i+1]] )

    # 开始接收bloom的回复
    gen_kwargs = {
            "top_p": llm_kwargs['top_p'],
            "top_k": 0.1,
            "temperature": llm_kwargs['temperature'],
            "num_beams": 1,
            "max_length":llm_kwargs['max_length'],
            "max_new_tokens": 1024,
            "repetition_penalty": 1.0,
    }
    response = ""
    new_text = llm_models.chat(query=inputs, history=history_feedin,input_kwargs=gen_kwargs)
    print(new_text)
    response = new_text[0]
    return response



def predict(inputs, llm_kwargs, history=[], sys_prompt='', stream = True, additional_fn=None,llm_models=None):
    """
        单线程方法
        函数的说明请见 request_llm/bridge_all.py
    """

    # if model is None:
    #     model = GetBloomhandle()

    if additional_fn is not None:
        importlib.reload(core_functional)    # 热更新prompt
        core_functional = core_functional.get_core_functions()
        if "PreProcess" in core_functional[additional_fn]: inputs = core_functional[additional_fn]["PreProcess"](inputs)  # 获取预处理函数（如果有的话）
        inputs = core_functional[additional_fn]["Prefix"] + inputs + core_functional[additional_fn]["Suffix"]

    # 处理历史信息
    history_feedin = []
    history_feedin.append(["What can I do?", sys_prompt] )
    for i in range(len(history)//2):
        history_feedin.append([history[2*i], history[2*i+1]] )

    # 开始接收bloom的回复
    gen_kwargs = {
            "top_p": llm_kwargs['top_p'],
            "top_k": 0.1,
            "temperature": llm_kwargs['temperature'],
            "num_beams": 1,
            "max_length":llm_kwargs['max_length'],
            "max_new_tokens": 1024,
            "repetition_penalty": 1.0,
    }
    response = ""
    for new_text in llm_models.stream_chat(query=inputs, history=history_feedin,input_kwargs=gen_kwargs):
        response += new_text
        print亮蓝(response)

    # 总结输出
    if response == "":
        response = "[Local Message]: Bloom响应异常 ..."
    history.extend([inputs, response])
    return response

if __name__ == "__main__":
    sys.path.append("../../LLaMA-Efficient-Tuning/src")
    from llmtuner import ChatModel
    from llmtuner.tuner import get_infer_args
    # 第一次运行，加载参数
    # bloom_handle = None
    chat_model = None
    sys.argv=['bridge_bloom.py', '--dataset_dir', '../data', '--model_name_or_path', '/home/amov/LLaMA-Efficient-Tuning/model/bloom', '--checkpoint_dir', '/home/amov/LLaMA-Efficient-Tuning/output/sft/checkpoint-3000', '--quantization_bit', '8']
    print("get_infer_args",*get_infer_args())
    device, = get_conf('LOCAL_MODEL_DEVICE')
    if device=='cuda':
        chat_model =  ChatModel(*get_infer_args())
    # bloom_handle = GetBloomhandle()
    # bloom_handle.chat_get(chat_model)
    # print亮蓝(type(bloom_handle.bloom_model))

    proxies, LLM_MODEL, API_KEY = get_conf('proxies', 'LLM_MODEL',  'API_KEY')
    llm_kwargs = {
        'api_key': API_KEY,
        'llm_model': "chatglm",
        'top_p': 1.0,
        'max_length': 512,
        'temperature': 1.0,
    }
    result = predict_long_connection("请解释一下mapping的意思", llm_kwargs, history=[""], sys_prompt="你是一个情感专家", model =chat_model)
    print亮蓝(result)
    time.sleep(20)
    result = predict("robot mapping", llm_kwargs, history=["机器人建图:"], sys_prompt="你是一个slam专家，请翻译下面的短语", model =chat_model)
    print亮蓝(result)