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
sys.path.append("../bloom/src")
from llmtuner import ChatModel

class GetBloomhandle:
    def __init__(self):
        super().__init__(daemon=True)
        self.parent, self.child = Pipe()
        self.bloom_model = None
        self.threadLock = threading.Lock()
    def run(self):
        # 子进程执行
        # 第一次运行，加载参数
        retry = 0
        while True:
            try:
                if self.bloom_model is None:
                    sys.argv=['--dataset_dir', '../data', '--model_name_or_path', '/home/amov/LLaMA-Efficient-Tuning/model/bloom', '--checkpoint_dir', '/home/amov/LLaMA-Efficient-Tuning/output/sft/checkpoint-3000', '--quantization_bit', '8']
                    self.bloom_model = ChatModel(*get_infer_args())
                    break
                else:
                    break
            except:
                retry += 1
                if retry > 3: 
                    self.child.send('[Local Message] Call ChatGLM fail 不能正常加载ChatGLM的参数。')
                    raise RuntimeError("不能正常加载ChatGLM的参数！")
        while True:
            # 进入任务等待状态
            kwargs = self.child.recv()#使用pipe来处理消息
            # 收到消息，开始请求
            try:
                for new_text in self.bloom_model.stream_chat(**kwargs):
                    self.child.send(new_text)
                    #response += new_text
                    # 中途接收可能的终止指令（如果有的话）
                    if self.child.poll(): 
                        command = self.child.recv()
                        if command == '[Terminate]': break
            except:
                self.child.send('[Local Message] Call ChatGLM fail.' + '\n```\n' + trimmed_format_exc() + '\n```\n')
            # 请求处理结束，开始下一个循环
            self.child.send('[Finish]')

    def stream_chat(self, **kwargs):
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


global bloom_handle
bloom_handle = None
#################################################################################
def predict_no_ui_long_connection(inputs, llm_kwargs, history=[], sys_prompt="", console_slience=False):
    """
        多线程方法
        函数的说明请见 request_llm/bridge_all.py
    """
    global bloom_handle
    if bloom_handle is None:
        bloom_handle = GetBloomhandle()#实例化handle

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
    for new_text in bloom_handle.stream_chat(query=inputs, history=history_feedin,input_kwargs=gen_kwargs):
        print亮蓝(response)
        response += new_text
    return response



def predict(inputs, llm_kwargs, history=[], system_prompt='', stream = True, additional_fn=None):
    """
        单线程方法
        函数的说明请见 request_llm/bridge_all.py
    """

    global bloom_handle
    if bloom_handle is None:
        bloom_handle = GetBloomhandle()

    if additional_fn is not None:
        importlib.reload(core_functional)    # 热更新prompt
        core_functional = core_functional.get_core_functions()
        if "PreProcess" in core_functional[additional_fn]: inputs = core_functional[additional_fn]["PreProcess"](inputs)  # 获取预处理函数（如果有的话）
        inputs = core_functional[additional_fn]["Prefix"] + inputs + core_functional[additional_fn]["Suffix"]

    # 处理历史信息
    history_feedin = []
    history_feedin.append(["What can I do?", system_prompt] )
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
    for new_text in bloom_handle.stream_chat(query=inputs, history=history_feedin,input_kwargs=gen_kwargs):
        print亮蓝(response)
        response += new_text

    # 总结输出
    if response == "":
        response = "[Local Message]: ChatGLM响应异常 ..."
    history.extend([inputs, response])
    return response
