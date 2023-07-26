import os
import traceback
import tiktoken
from functools import lru_cache
import numpy as np

"""
这个函数主要是用来解析token和文本之间转化关系的：https://blog.csdn.net/lovechris00/article/details/129889317
"""


class TiktokenCalculate(object):
    def __init__(self, model):
        self.model = model

    @staticmethod
    @lru_cache(maxsize=128)
    def get_encoder(model):
        # 正在加载tokenizer，如果是第一次运行，可能需要一点时间下载参数
        tmp = tiktoken.encoding_for_model(model)
        return tmp

    '''self 是一个特殊的参数，表示类实例本身，用于访问类的属性和方法；
       *args 是一个包含所有非关键字参数的元组，允许函数接受任意数量的参数；
       **kwargs 是一个包含所有关键字参数的字典，允许函数接受任意数量的关键字参数。
    '''

    def encode(self, *args, **kwargs):
        encoder = self.get_encoder(self.model)
        return encoder.encode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        encoder = self.get_encoder(self.model)
        return encoder.decode(*args, **kwargs)


tokenizer_gpt3_5 = TiktokenCalculate("gpt-3.5-turbo")
tokenizer_gpt4 = TiktokenCalculate("gpt-4")

"""
 主要是返回当前异常的回溯信息字符串
"""


def trimmed_format_exc():
    str = traceback.format_exc()  # 返回当前异常的回溯信息字符串
    current_path = os.getcwd()
    replace_path = "."
    return str.replace(current_path, replace_path)  # 将当前路径替换为.，避免信息泄露


"""
通过裁剪来缩短历史记录的长度。 
此函数逐渐地搜索最长的条目进行剪辑，
直到历史记录的标记数量降低到阈值以下。
"""


def clip_history(inputs, history, tokenizer, max_token_limit):
    def get_token_num(txt):  # 获取文字对应的token数量
        return len(tokenizer.encode(txt, disallowed_special=()))
    input_token_num = get_token_num(inputs)
    if input_token_num < max_token_limit * 3 / 4:
        # 当输入部分的token占比小于限制的3/4时，裁剪时
        # 1. 把input的余量留出来
        max_token_limit = max_token_limit - input_token_num
        # 2. 把输出用的余量留出来
        max_token_limit = max_token_limit - 128
        # 3. 如果余量太小了，直接清除历史
        if max_token_limit < 128:
            history = []
            return history
    else:
        # 当输入部分的token占比 > 限制的3/4时，直接清除历史
        history = []
        return history

    everything = ['']
    everything.extend(history)  # 把历史记录传入
    n_token = get_token_num('\n'.join(everything))  # 获取合并后的token数量
    everything_token = [get_token_num(e)
                        for e in everything]  # 获取每个历史记录的token数量

    # 截断时的颗粒度
    delta = max(everything_token) // 16

    while n_token > max_token_limit:
        where = np.argmax(everything_token)  # 获取token数量最多的历史记录
        encoded = tokenizer.encode(
            everything[where], disallowed_special=())  # 将历史记录转化为token
        clipped_encoded = encoded[:len(encoded)-delta]  # 从前往后截断token
        # -1 to remove the may-be illegal char
        everything[where] = tokenizer.decode(clipped_encoded)[
            :-1]  # 将截断后的token转化为文本
        everything_token[where] = get_token_num(
            everything[where])  # 获取截断后的token数量
        n_token = get_token_num('\n'.join(everything))

    history = everything[1:]
    return history


if __name__ == "__main__":
    token_模块预热 = tokenizer_gpt3_5.encode("模块预热")
    print(token_模块预热)
    print(tokenizer_gpt3_5.decode(token_模块预热))

    token_模块预热 = tokenizer_gpt4.encode("模块预热")
    print(token_模块预热)

    print(trimmed_format_exc())
    print(clip_history("我是输入", ["我是历史记录"], tokenizer_gpt3_5, 256+3))
    print(clip_history("我是输入", ["我是历史记录"], tokenizer_gpt3_5, 256+4))
