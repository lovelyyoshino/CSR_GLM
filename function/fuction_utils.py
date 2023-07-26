
import numpy as np
import sys
import time
import re
import random
from concurrent.futures import ThreadPoolExecutor
sys.path.append("../llm_bridge")
from llm_bridge_all import model_type
from token_calculate import trimmed_format_exc
from llm_bridge_all import chat_multiple_with_pre_chat
sys.path.append("../tools")
from get_confs import get_conf
from color import print亮红, print亮绿, print亮蓝, print亮黄
from check_proxy import check_proxy


"""
将输入完成裁剪，使其token数量不超过限制
"""
def input_clipping(inputs, history, max_token_limit):
    enc = model_type["gpt-3.5-turbo"]['tokenizer']#获取对应模型的最大token数量
    def get_token_num(txt): return len(enc.encode(txt, disallowed_special=()))

    mode = 'input-and-history'
    # 当 输入部分的token占比 小于 全文的一半时，只裁剪历史
    input_token_num = get_token_num(inputs)
    if input_token_num < max_token_limit//2: 
        mode = 'only-history'
        max_token_limit = max_token_limit - input_token_num#把input的余量留出来

    everything = [inputs] if mode == 'input-and-history' else ['']#如果是input-and-history模式，把input加入到everything中，认为 input是第一个历史记录，也需要被裁剪
    everything.extend(history)#把历史记录传入
    n_token = get_token_num('\n'.join(everything))#获取合并后的token数量
    everything_token = [get_token_num(e) for e in everything]#获取每个历史记录的token数量
    delta = max(everything_token) // 16 # 截断时的颗粒度
        
    while n_token > max_token_limit:#当token数量大于限制时
        where = np.argmax(everything_token)
        encoded = enc.encode(everything[where], disallowed_special=())
        clipped_encoded = encoded[:len(encoded)-delta]#从前往后截断token
        everything[where] = enc.decode(clipped_encoded)[:-1]    # -1 to remove the may-be illegal char
        everything_token[where] = get_token_num(everything[where])
        n_token = get_token_num('\n'.join(everything))#获取合并后的token数量

    if mode == 'input-and-history':
        inputs = everything[0]#如果是input-and-history模式，把everything[0]赋值给inputs
    else:
        pass
    history = everything[1:]#把everything[1:]赋值给history
    return inputs, history

"""
    获取处理文本的比例与信息
"""
def get_reduce_token_percent(text):
    try:
        # text = "maximum context length is 4097 tokens. However, your messages resulted in 4870 tokens"
        pattern = r"(\d+)\s+tokens\b"
        match = re.findall(pattern, text)
        EXCEED_ALLO = 500  # 稍微留一点余地，否则在回复时会因余量太少出问题
        max_limit = float(match[0]) - EXCEED_ALLO
        current_tokens = float(match[1])
        ratio = max_limit/current_tokens
        assert ratio > 0 and ratio < 1
        return ratio, str(int(current_tokens-max_limit))
    except:
        return 0.5, '不详'

"""
请求GPT模型

输入参数 Args （以_array结尾的输入变量都是列表，列表长度为子任务的数量，执行时，会把列表拆解，放到每个子线程中分别执行）:
    inputs (string): List of inputs （输入）
    top_p (float): Top p value for sampling from model distribution （GPT参数，浮点数）
    temperature (float): Temperature value for sampling from model distribution（GPT参数，浮点数）
    history (list): List of chat history （历史，对话历史列表）
    sys_prompt (string): List of system prompts （系统输入，列表，用于输入给GPT的前提提示，比如你是翻译官怎样怎样）
    handle_token_exceed：是否自动处理token溢出的情况，如果选择自动处理，则会在溢出时暴力截断，默认开启
    retry_times_at_unknown_error：失败时的重试次数

输出 Returns:
    future: 输出，GPT返回的结果
"""
def request_gpt_model_in_new_thread(
        inputs, llm_kwargs, history, sys_prompt,
        handle_token_exceed=True, 
        retry_times_at_unknown_error=2):
    executor = ThreadPoolExecutor(max_workers=16)#创建线程池
    def _req_gpt(inputs, history, sys_prompt):#定义一个函数，用于请求GPT模型
        retry_op = retry_times_at_unknown_error#重试次数
        exceeded_cnt = 0
        while True:
            try:
                # 【第一种情况】：顺利完成
                result = chat_multiple_with_pre_chat(
                    inputs=inputs, llm_kwargs=llm_kwargs,
                    history=history, sys_prompt=sys_prompt)
                return result
            except ConnectionAbortedError as token_exceeded_error:
                # 【第二种情况】：Token溢出
                if handle_token_exceed:
                    exceeded_cnt += 1#溢出次数+1
                    # 【选择处理】 尝试计算比例，尽可能多地保留文本      
                    p_ratio, n_exceed = get_reduce_token_percent(str(token_exceeded_error))
                    MAX_TOKEN = 4096
                    EXCEED_ALLO = 512 + 512 * exceeded_cnt
                    inputs, history = input_clipping(inputs, history, max_token_limit=MAX_TOKEN-EXCEED_ALLO)
                    print亮黄(f'[Local Message] 警告，文本过长将进行截断，Token溢出数：{n_exceed}。\n\n')
                    continue # 返回重试
                else:
                    # 【选择放弃】
                    tb_str = '```\n' + trimmed_format_exc() + '```'
                    print亮黄(f"[Local Message] 警告，在执行过程中遭遇问题, Traceback：\n\n{tb_str}\n\n")
                    return "" # 放弃
            except:
                # 【第三种情况】：其他错误：重试几次
                tb_str = '```\n' + trimmed_format_exc() + '```'
                print亮黄(f"[Local Message] 警告，在执行过程中遭遇问题, Traceback：\n\n{tb_str}\n\n")
                if retry_op > 0:
                    retry_op -= 1
                    print亮黄(f"[Local Message] 重试中，请稍等 {retry_times_at_unknown_error-retry_op}/{retry_times_at_unknown_error}：\n\n")
                    if ("Rate limit reached" in tb_str) or ("Too Many Requests" in tb_str):
                        time.sleep(30)
                    time.sleep(5)
                    continue # 返回重试
                else:
                    return "" # 放弃

    # 提交任务
    future = executor.submit(_req_gpt, inputs, history, sys_prompt)
    while True:
        if future.done():
            break
    final_result = future.result()
    return final_result

"""
Request GPT model using multiple threads with UI and high efficiency
请求GPT模型的[多线程]版。
具备以下功能：
    实时在UI上反馈远程数据流
    使用线程池，可调节线程池的大小避免openai的流量限制错误
    处理中途中止的情况
    网络等出问题时，会把traceback和已经接收的数据转入输出

输入参数 Args （以_array结尾的输入变量都是列表，列表长度为子任务的数量，执行时，会把列表拆解，放到每个子线程中分别执行）:
    inputs_array (list): List of inputs （每个子任务的输入）
    inputs_show_user_array (list): List of inputs to show user（每个子任务展现在报告中的输入，借助此参数，在汇总报告中隐藏啰嗦的真实输入，增强报告的可读性）
    llm_kwargs: llm_kwargs参数
    history_array (list): List of chat history （历史对话输入，双层列表，第一层列表是子任务分解，第二层列表是对话历史）
    sys_prompt_array (list): List of system prompts （系统输入，列表，用于输入给GPT的前提提示，比如你是翻译官怎样怎样）
    max_workers (int, optional): Maximum number of threads (default: see config.py) （最大线程数，如果子任务非常多，需要用此选项防止高频地请求openai导致错误）
    handle_token_exceed (bool, optional): （是否在输入过长时，自动缩减文本）
    handle_token_exceed：是否自动处理token溢出的情况，如果选择自动处理，则会在溢出时暴力截断，默认开启
    retry_times_at_unknown_error：子任务失败时的重试次数

输出 Returns:
    list: List of GPT model responses （每个子任务的输出汇总，如果某个子任务出错，response中会携带traceback报错信息，方便调试和定位问题。）
"""
def request_gpt_model_multi_threads_with_high_efficiency(
        inputs_array,  llm_kwargs, 
        history_array, sys_prompt_array, 
        max_workers=-1,
        handle_token_exceed=True, 
        retry_times_at_unknown_error=2,
        ):
    assert len(inputs_array) == len(history_array)#输入数量和历史数量必须一致
    assert len(inputs_array) == len(sys_prompt_array)#输入数量和系统提示数量必须一致
    if max_workers == -1: # 读取配置文件
        try: max_workers, = get_conf('DEFAULT_WORKER_NUM')#读取配置文件
        except: max_workers = 8
        if max_workers <= 0: max_workers = 3
    # 屏蔽掉 chatglm的多线程，可能会导致严重卡顿
    if not (llm_kwargs['llm_model'].startswith('gpt-') or llm_kwargs['llm_model'].startswith('api2d-')):#如果不是gpt模型，不使用多线程
        max_workers = 1
        
    executor = ThreadPoolExecutor(max_workers=max_workers)#线程池
    n_frag = len(inputs_array)

    # 子线程任务
    def _req_gpt(index, inputs, history, sys_prompt):
        gpt_say = ""
        retry_op = retry_times_at_unknown_error
        exceeded_cnt = 0
        while True:
            # watchdog error
            try:
                # 【第一种情况】：顺利完成
                # time.sleep(10); raise RuntimeError("测试")
                gpt_say = chat_multiple_with_pre_chat(
                    inputs=inputs, llm_kwargs=llm_kwargs, history=history, 
                    sys_prompt=sys_prompt, console_slience=True
                )
                return gpt_say
            except ConnectionAbortedError as token_exceeded_error:
                # 【第二种情况】：Token溢出，
                if handle_token_exceed:
                    exceeded_cnt += 1
                    # 【选择处理】 尝试计算比例，尽可能多地保留文本
                    p_ratio, n_exceed = get_reduce_token_percent(str(token_exceeded_error))
                    MAX_TOKEN = 4096
                    EXCEED_ALLO = 512 + 512 * exceeded_cnt
                    inputs, history = input_clipping(inputs, history, max_token_limit=MAX_TOKEN-EXCEED_ALLO)
                    print亮黄(f'[Local Message] 警告，{index}文本过长将进行截断，Token溢出数：{n_exceed}。\n\n')
                    continue # 返回重试
                else:
                    # 【选择放弃】
                    tb_str = '```\n' + trimmed_format_exc() + '```'
                    print亮黄(f"[Local Message] 警告，线程{index}在执行过程中遭遇问题, Traceback：\n\n{tb_str}\n\n")
                    return gpt_say # 放弃
            except:
                # 【第三种情况】：其他错误
                tb_str = '```\n' + trimmed_format_exc() + '```'
                print亮黄(f"[Local Message] 警告，线程{index}在执行过程中遭遇问题, Traceback：\n\n{tb_str}\n\n")
                if retry_op > 0: 
                    retry_op -= 1
                    wait = random.randint(5, 20)
                    if ("Rate limit reached" in tb_str) or ("Too Many Requests" in tb_str):
                        wait = wait * 3
                        fail_info = "OpenAI绑定信用卡可解除频率限制 "
                    else:
                        fail_info = ""
                    # 也许等待十几秒后，情况会好转
                    for i in range(wait):
                        print亮黄(f"{fail_info}等待重试 {wait-i}")
                        time.sleep(1)
                    # 开始重试
                    print亮黄(f"重试中 {retry_times_at_unknown_error-retry_op}/{retry_times_at_unknown_error}")
                    continue # 返回重试
                else:
                    print亮红("已失败")
                    return gpt_say # 放弃

    # 异步任务开始
    futures = [executor.submit(_req_gpt, index, inputs, history, sys_prompt) for index, inputs, history, sys_prompt in zip(
        range(len(inputs_array)), inputs_array, history_array, sys_prompt_array)]
    cnt = 0
    while True:
        cnt += 1
        worker_done = [h.done() for h in futures]
        if all(worker_done):
            executor.shutdown()
            break
    
    # 异步任务结束
    gpt_response_collection = []
    for f in futures:
        gpt_res = f.result()
        gpt_response_collection.extend([gpt_res])
    return gpt_response_collection




if __name__ == "__main__":
    inputs,history = input_clipping("你好",["你好", "你好", "你好"], 8)
    print(inputs,",",history)
    proxies, LLM_MODEL, API_KEY = get_conf('proxies', 'LLM_MODEL',  'API_KEY')
    llm_kwargs = {
        'api_key': API_KEY,
        'llm_model': LLM_MODEL,
        'top_p': 1.0,
        'max_length': None,
        'temperature': 1.0,
    }
    if check_proxy(proxies) == True:
        gpt_say = request_gpt_model_in_new_thread("请解释一下mapping的意思", llm_kwargs, history=[""], sys_prompt="你是一个情感专家")  # 带超时倒计时
        print亮蓝(gpt_say)

        gpt_say = request_gpt_model_multi_threads_with_high_efficiency(["请解释一下mapping的意思","请解释一下love的意思"], llm_kwargs, history_array=[[""],[""]], sys_prompt_array=["你是一个slam专家","你是一个情感专家"])  # 带超时倒计时
        print亮蓝(gpt_say)
