# -*- coding: utf-8 -*-
import config
import importlib
import os
import re
from color import print亮红, print亮绿, print亮蓝
from functools import lru_cache
import random

"""
检查是否是openai的api key，主要是通过正则表达式匹配
"""


def __is_openai_api_key(key: str):
    API_MATCH_ORIGINAL = re.match(r"sk-[a-zA-Z0-9]{48}$", key)
    API_MATCH_AZURE = re.match(r"[a-zA-Z0-9]{32}$", key)
    return bool(API_MATCH_ORIGINAL) or bool(API_MATCH_AZURE)


"""
检查是否是api2d的api key，主要是通过正则表达式匹配
"""


def __is_api2d_key(key: str):
    if key.startswith('fk') and len(key) == 41:
        return True
    else:
        return False


"""
分割字符。并检查是否存在api key
"""


def __is_any_api_key(key: str):
    if ',' in key:
        keys = key.split(',')
        for k in keys:
            if __is_any_api_key(k):
                return True
        return False
    else:
        return __is_openai_api_key(key) or __is_api2d_key(key)


"""
检查环境变量，并从全局变量中替换对应的值
"""


def __read_env_variable(arg, default_value):
    """
    环境变量可以是 `GPT_ACADEMIC_CONFIG`(优先)，也可以直接是`CONFIG`
    例如在windows cmd中，既可以写：
        set USE_PROXY=True
        set API_KEY=sk-j7caBpkRoxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        set proxies={"http":"http://127.0.0.1:10085", "https":"http://127.0.0.1:10085",}
        set AVAIL_LLM_MODELS=["gpt-3.5-turbo", "chatglm"]
    也可以写：
        set GPT_ACADEMIC_USE_PROXY=True
        set GPT_ACADEMIC_API_KEY=sk-j7caBpkRoxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        set GPT_ACADEMIC_proxies={"http":"http://127.0.0.1:10085", "https":"http://127.0.0.1:10085",}
        set GPT_ACADEMIC_AVAIL_LLM_MODELS=["gpt-3.5-turbo", "chatglm"]
    """
    arg_with_prefix = "GPT_ACADEMIC_" + arg
    if arg_with_prefix in os.environ:  # 优先使用带前缀的环境变量
        env_arg = os.environ[arg_with_prefix]
    elif arg in os.environ:  # 其次使用不带前缀的环境变量
        env_arg = os.environ[arg]
    else:
        raise KeyError
    # 修正值是指环境变量的值,只有存在才会打印，否则会raise KeyError
    print(f"[ENV_VAR] 尝试加载{arg}，默认值：{default_value} --> 修正值：{env_arg}")
    try:
        if isinstance(default_value, bool):  # isinstance() 函数来判断一个对象是否是一个已知的类型，类似 type()。
            r = bool(env_arg)
        elif isinstance(default_value, int):
            r = int(env_arg)
        elif isinstance(default_value, float):
            r = float(env_arg)
        elif isinstance(default_value, str):
            r = env_arg.strip()
        elif isinstance(default_value, dict):
            r = eval(env_arg)
        elif isinstance(default_value, list):
            r = eval(env_arg)
        elif default_value is None:
            assert arg == "proxies"
            r = eval(env_arg)
        else:
            print亮红(f"[ENV_VAR] 环境变量{arg}不支持通过环境变量设置! ")
            raise KeyError
    except:
        print亮红(f"[ENV_VAR] 环境变量{arg}加载失败! ")
        raise KeyError(f"[ENV_VAR] 环境变量{arg}加载失败! ")

    print亮绿(f"[ENV_VAR] 成功读取环境变量{arg}")
    return r


"""
用数组的形式返回所有的配置项，并检查是否存在api key
"""
# 将其执行的结果缓存起来，当下次请求的时候，如果请求该函数的传参未变则直接返回缓存起来的结果而不再执行函数：https://blog.csdn.net/momoda118/article/details/120726050


@lru_cache(maxsize=128)
def __read_single_conf_with_lru_cache(arg):
    # 动态地获取另一个py文件中定义好的变量/方法：https://blog.csdn.net/m0_37738114/article/details/120502088
    try:
        # 优先级1. 获取环境变量作为配置
        # 读取默认值作为数据类型转换的参考，importlib.import_module('config')是动态地获取另一个py文件中定义，然后getattr()获取其中arg的变量
        default_ref = getattr(importlib.import_module('config'), arg)
        r = __read_env_variable(arg, default_ref)
    except:
        try:
            # 优先级2. 获取config_private中的配置
            r = getattr(importlib.import_module('config_private'), arg)
        except:
            # 优先级3. 获取config中的配置
            r = getattr(importlib.import_module('config'), arg)

    # 在读取API_KEY时，检查一下是不是忘了改config
    if arg == 'API_KEY':
        print亮蓝(
            f"[API_KEY] 本项目现已支持OpenAI和API2D的api-key。也支持同时填写多个api-key，如API_KEY=\"openai-key1,openai-key2,api2d-key3\"")
        print亮蓝(
            f"[API_KEY] 您既可以在config.py中修改api-key(s)，也可以在问题输入区输入临时的api-key(s)，然后回车键提交后即可生效。")
        if __is_any_api_key(r):
            print亮绿(f"[API_KEY] 您的 API_KEY 是: {r[:15]}*** API_KEY 导入成功")
        else:
            print亮红(
                "[API_KEY] 正确的 API_KEY 是'sk'开头的51位密钥（OpenAI），或者 'fk'开头的41位密钥，请在config文件中修改API密钥之后再运行。")
    if arg == 'proxies':
        if r is None:
            print亮红(
                '[PROXY] 网络代理状态：未配置。无代理状态下很可能无法访问OpenAI家族的模型。建议：检查USE_PROXY选项是否修改。')
        else:
            print亮绿('[PROXY] 网络代理状态：已配置。配置信息如下：', r)
            assert isinstance(r, dict), 'proxies格式错误，请注意proxies选项的格式，不要遗漏括号。'
    return r


"""
------------------- 开放的接口-----------------
"""


def what_keys(keys: str):
    avail_key_list = {'OpenAI Key': 0, "API2D Key": 0}
    key_list = keys.split(',')

    for k in key_list:
        if __is_openai_api_key(k):
            avail_key_list['OpenAI Key'] += 1

    for k in key_list:
        if __is_api2d_key(k):
            avail_key_list['API2D Key'] += 1

    return f"检测到： OpenAI Key {avail_key_list['OpenAI Key']} 个，API2D Key {avail_key_list['API2D Key']} 个"


def select_api_key(keys: str, llm_model: str):
    avail_key_list = []
    key_list = keys.split(',')

    if llm_model.startswith('gpt-'):
        for k in key_list:
            if __is_openai_api_key(k):
                avail_key_list.append(k)

    if llm_model.startswith('api2d-'):
        for k in key_list:
            if __is_api2d_key(k):
                avail_key_list.append(k)

    if len(avail_key_list) == 0:
        raise RuntimeError(
            f"您提供的api-key不满足要求，不包含任何可用于{llm_model}的api-key。您可能选择了错误的模型或请求源。")

    api_key = random.choice(avail_key_list)  # 随机负载均衡
    return api_key


def get_conf(*args):
    res = []
    for arg in args:
        r = __read_single_conf_with_lru_cache(arg)
        res.append(r)
    return res

if __name__ == "__main__":
    key_config = get_conf('API_KEY')
    print(key_config,"\r\n",get_conf('proxies'))
    print(get_conf('API_KEY', 'proxies'))

    api_key = select_api_key(key_config[0],"gpt-3.5-turbo")
    print(api_key)
    print(what_keys("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"))
