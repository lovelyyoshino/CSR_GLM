import os
import traceback
import tiktoken
from functools import lru_cache


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


def trimmed_format_exc():
    str = traceback.format_exc()  # 返回当前异常的回溯信息字符串
    current_path = os.getcwd()
    replace_path = "."
    return str.replace(current_path, replace_path)  # 将当前路径替换为.，避免信息泄露


# if __name__ == "__main__":
#     token_模块预热 = tokenizer_gpt3_5.encode("模块预热")
#     print(token_模块预热)
#     print(tokenizer_gpt3_5.decode(token_模块预热))

#     token_模块预热 = tokenizer_gpt4.encode("模块预热")
#     print(token_模块预热)

#     print(trimmed_format_exc())
