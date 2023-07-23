import json
import time
import gradio as gr
import logging
import traceback
import requests
import importlib
import sys
import re
sys.path.append("../tools")
from token_calculate import trimmed_format_exc, clip_history
from get_confs import get_conf, select_api_key, what_keys
from color import print亮红, print亮绿, print亮蓝, print亮黄
from check_proxy import check_proxy
from file_conversion import regular_txt_to_markdown
from core_functional import get_core_functions


class GetChatGPTHandle:
    def __init__(self) -> None:
        #super().__init__(daemon=True)  # 设置为守护进程,daemon设置为True时，主线程结束，子线程也会结束
        self.proxies, self.TIMEOUT_SECONDS, self.MAX_RETRY =  \
            get_conf('proxies', 'TIMEOUT_SECONDS', 'MAX_RETRY')
        self.timeout_bot_msg = '[Local Message] Request timeout. Network error. Please check proxy settings in config.py.' + \
            '网络错误，检查代理服务器是否可用，以及代理设置的格式是否正确，格式须是[协议]://[地址]:[端口]，缺一不可。'

    """
        获取完整的从Openai返回的报错
    """

    def get_full_error(self, chunk, stream_response):
        while True:
            try:
                chunk += next(stream_response)
            except:
                break
        return chunk

    """
    整合所有信息，选择LLM模型，生成http请求，为发送请求做准备
    """

    def generate_payload(self, inputs, llm_kwargs, history, sys_prompt, stream):
        api_key = select_api_key(
            llm_kwargs['api_key'], llm_kwargs['llm_model'])  # 选择api_key

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }  # 设置请求头

        # 除以2是因为history中包含了用户和机器人的对话，每次对话都是两条记录
        conversation_cnt = len(history) // 2

        # openai的第一句话是sys_prompt
        messages = [{"role": "system", "content": sys_prompt}]
        if conversation_cnt:  # 如果有对话记录
            for index in range(0, 2*conversation_cnt, 2):  # 从0开始，每次跳两个
                what_i_have_asked = {}
                what_i_have_asked["role"] = "user"
                what_i_have_asked["content"] = history[index]  # 用户的对话
                what_gpt_answer = {}
                what_gpt_answer["role"] = "assistant"
                what_gpt_answer["content"] = history[index+1]  # 机器人的对话
                if what_i_have_asked["content"] != "":  # 如果用户的对话不为空
                    if what_gpt_answer["content"] == "":  # 如果机器人的对话为空
                        continue
                    if what_gpt_answer["content"] == self.timeout_bot_msg:
                        continue
                    messages.append(what_i_have_asked)  # 将用户的对话加入到messages中
                    messages.append(what_gpt_answer)  # 将机器人的对话加入到messages中
                else:
                    messages[-1]['content'] = what_gpt_answer['content']

        what_i_ask_now = {}
        what_i_ask_now["role"] = "user"
        what_i_ask_now["content"] = inputs  # 当前用户输入的对话
        messages.append(what_i_ask_now)  # 将当前用户输入的对话加入到messages中

        payload = {
            "model": llm_kwargs['llm_model'].strip('api2d-'),
            "messages": messages,
            "temperature": llm_kwargs['temperature'],  # 1.0,
            "top_p": llm_kwargs['top_p'],  # 1.0,
            "n": 1,
            "stream": stream,
            "presence_penalty": 0,
            "frequency_penalty": 0,
        }
        try:
            print(
                f" {llm_kwargs['llm_model']} : {conversation_cnt} : {inputs[:100]} ..........")
        except:
            print('输入中可能存在乱码。')
        return headers, payload  # 返回请求头和请求体

    def get_timeout_second(self):
        return self.TIMEOUT_SECONDS

    def get_max_retry(self):
        return self.MAX_RETRY

    def get_proxies(self):
        return self.proxies


global gpt_handle
gpt_handle = None
#################################################################################
"""
    发送至chatGPT，等待回复，一次性完成，不显示中间过程。但内部用stream的方法避免中途网线被掐。
    函数的说明请见 request_llm/bridge_all.py
    inputs：
        是本次问询的输入
    sys_prompt:
        系统静默prompt
    llm_kwargs：
        chatGPT的内部调优参数
    history：
        是之前的对话列表
"""
#################################################################################


def predict_long_connection(inputs, llm_kwargs, history=[], sys_prompt="", console_slience=False,llm_models=None):
    global gpt_handle
    if gpt_handle is None:
        gpt_handle = GetChatGPTHandle()

    try:
        headers, payload = gpt_handle.generate_payload(
            inputs, llm_kwargs, history, sys_prompt=sys_prompt, stream=True)
    except RuntimeError as e:
        print亮红(inputs,
                f"您提供的api-key不满足要求，不包含任何可用于{llm_kwargs['llm_model']}的api-key。您可能选择了错误的模型或请求源。")
        return ""

    retry = 0  # 重试次数
    while True:
        try:
            from llm_bridge_all import model_type
            # make a POST request to the API endpoint, stream=False
            endpoint = model_type[llm_kwargs['llm_model']
                                  ]['endpoint']  # 获取对应的链接api
            print(gpt_handle.get_proxies())
            response = requests.post(endpoint, headers=headers, proxies=gpt_handle.get_proxies(),
                                     json=payload, stream=True, timeout=gpt_handle.get_timeout_second())  # post发送信息
            break
        except requests.exceptions.ReadTimeout as e:
            retry += 1
            traceback.print_exc()
            if retry > gpt_handle.get_max_retry():
                raise TimeoutError
            if gpt_handle.get_max_retry() != 0:
                print(f'请求超时，正在重试 ({retry}/{gpt_handle.get_max_retry()}) ……')
    stream_response = response.iter_lines()  # 获取返回信息，并按照lines排列
    result = ""
    while True:
        try:
            chunk = next(stream_response).decode()  # 直接对post返回信息解析
        except StopIteration:
            break
        except requests.exceptions.ConnectionError:
            chunk = next(stream_response).decode()  # 失败了，重试一次？再失败就没办法了。
        if len(chunk) == 0:  # 未能解析or返回为空
            continue
        if not chunk.startswith('data:'):  # 如果起始为data开头
            error_msg = gpt_handle.get_full_error(chunk.encode(
                'utf8'), stream_response).decode()  # 获取报错
            if "reduce the length" in error_msg:
                print亮红("OpenAI拒绝了请求:" + error_msg)
                return ""
            else:
                print亮红("OpenAI拒绝了请求：" + error_msg)
                return ""
        if ('data: [DONE]' in chunk):
            break  # api2d 正常完成

        json_data = json.loads(chunk.lstrip('data:'))[
            'choices'][0]  # 解析data数据以json形式
        delta = json_data["delta"]
        if len(delta) == 0:
            break
        if "role" in delta:  # 这个只是代表人员
            continue
        if "content" in delta:  # 这个代表内容
            result += delta["content"]
            if not console_slience:
                print(delta["content"], end='')
        else:
            print亮红("意外Json结构：", delta)
            return result
    if json_data['finish_reason'] == 'length':
        print亮红("正常结束，但显示Token不足，导致输出不完整，请削减单次输入的文本量。")
        return ""
    return result


def predict(inputs, llm_kwargs, history=[], sys_prompt='', stream=True, additional_fn=None,llm_models=None):
    global gpt_handle

    if additional_fn is not None:
        importlib.reload(core_functional)    # 热更新prompt
        core_functional = core_functional.get_core_functions()
        if "PreProcess" in core_functional[additional_fn]: inputs = core_functional[additional_fn]["PreProcess"](inputs)  # 获取预处理函数（如果有的话）
        inputs = core_functional[additional_fn]["Prefix"] + inputs + core_functional[additional_fn]["Suffix"]

    if gpt_handle is None:
        gpt_handle = GetChatGPTHandle()
    try:
        headers, payload = gpt_handle.generate_payload(
            inputs, llm_kwargs, history, sys_prompt, stream)
    except RuntimeError as e:
        print亮红(inputs,
                f"您提供的api-key不满足要求，不包含任何可用于{llm_kwargs['llm_model']}的api-key。您可能选择了错误的模型或请求源。")
        return ""

    retry = 0
    while True:
        try:
            from llm_bridge_all import model_type
            # make a POST request to the API endpoint, stream=True
            endpoint = model_type[llm_kwargs['llm_model']]['endpoint']
            response = requests.post(endpoint, headers=headers, proxies=gpt_handle.get_proxies(),
                                     json=payload, stream=True, timeout=gpt_handle.get_timeout_second())
            break
        except:
            retry += 1
            retry_msg = f"，正在重试 ({retry}/{gpt_handle.get_max_retry()}) ……" if gpt_handle.get_max_retry() > 0 else ""
            if retry > gpt_handle.get_max_retry():
                return ""

    gpt_replying_buffer = ""

    is_head_of_the_stream = True
    if stream:
        stream_response = response.iter_lines()
        while True:
            chunk = next(stream_response)
            # print(chunk.decode()[6:])
            if is_head_of_the_stream and (r'"object":"error"' not in chunk.decode()):
                # 数据流的第一帧不携带content
                is_head_of_the_stream = False
                continue

            if chunk:
                try:
                    chunk_decoded = chunk.decode()
                    if not chunk_decoded.startswith('data:'):  # 如果起始为data开头
                        error_msg = gpt_handle.get_full_error(chunk_decoded.encode(
                            'utf8'), stream_response).decode()  # 获取报错
                        if "reduce the length" in error_msg:
                            print亮红("OpenAI拒绝了请求:" + error_msg)
                            return ""
                        else:
                            print亮红("OpenAI拒绝了请求：" + error_msg)
                            return ""
                        
                    # 前者API2D的
                    if ('data: [DONE]' in chunk_decoded) or (len(json.loads(chunk_decoded[6:])['choices'][0]["delta"]) == 0):
                        # 判定为数据流的结束，gpt_replying_buffer也写完了
                        logging.info(f'[response] {gpt_replying_buffer}')
                        return gpt_replying_buffer
                    # 处理数据流的主体
                    chunkjson = json.loads(chunk_decoded[6:])
                    status_text = f"finish_reason: {chunkjson['choices'][0]['finish_reason']}"
                    # 如果这里抛出异常，一般是文本过长，详情见get_full_error的输出
                    gpt_replying_buffer = gpt_replying_buffer + \
                        json.loads(chunk_decoded[6:])[
                            'choices'][0]["delta"]["content"]
                    history[-1] = gpt_replying_buffer
                except Exception as e:
                    traceback.print_exc()
                    chunk = gpt_handle.get_full_error(chunk, stream_response)
                    chunk_decoded = chunk.decode()
                    error_msg = chunk_decoded
                    if "reduce the length" in error_msg:
                        if len(history) >= 2:
                            history[-1] = ""
                            # 清除当前溢出的输入：history[-2] 是本次输入, history[-1] 是本次输出
                            history[-2] = ""
                        history = clip_history(inputs=inputs, history=history, tokenizer=model_type[llm_kwargs['llm_model']]['tokenizer'],
                                               max_token_limit=(model_type[llm_kwargs['llm_model']]['max_token']))  # history至少释放二分之一
                        print亮黄(
                            "[Local Message] Reduce the length. 本次输入过长, 或历史数据过长. 历史缓存数据已部分释放, 您可以请再次尝试. (若再次失败则更可能是因为输入过长.)")
                        # history = []    # 清除历史
                    elif "does not exist" in error_msg:
                        print亮黄(
                            f"[Local Message] Model {llm_kwargs['llm_model']} does not exist. 模型不存在, 或者您没有获得体验资格.")
                    elif "Incorrect API key" in error_msg:
                        print亮黄(
                            "[Local Message] Incorrect API key. OpenAI以提供了不正确的API_KEY为由, 拒绝服务.")
                    elif "exceeded your current quota" in error_msg:
                        print亮黄(
                            "[Local Message] You exceeded your current quota. OpenAI以账户额度不足为由, 拒绝服务.")
                    elif "bad forward key" in error_msg:
                        print亮黄("[Local Message] Bad forward key. API2D账户额度不足.")
                    elif "Not enough point" in error_msg:
                        print亮黄(
                            "[Local Message] Not enough point. API2D账户点数不足.")
                    else:
                        tb_str = '```\n' + trimmed_format_exc() + '```'
                        return regular_txt_to_markdown(chunk_decoded[4:])

if __name__ == "__main__":
    proxies, LLM_MODEL, API_KEY = get_conf('proxies', 'LLM_MODEL',  'API_KEY')
    llm_kwargs = {
        'api_key': API_KEY,
        'llm_model': LLM_MODEL,
        'top_p': 1.0,
        'max_length': None,
        'temperature': 1.0,
    }
    if check_proxy(proxies) == True:
        result = predict_long_connection("请解释一下mapping的意思", llm_kwargs, history=[""], sys_prompt="你是一个情感专家")
        print亮蓝(result)
        time.sleep(20)
        result = predict("robot mapping", llm_kwargs, history=["机器人建图:"], sys_prompt="你是一个slam专家，请翻译下面的短语")
        print亮蓝(result)
