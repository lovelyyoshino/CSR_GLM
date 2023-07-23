# -*- coding: utf-8 -*-
import platform
from sys import stdout

# 当系统为Linux时直接跳过
if platform.system()=="Linux":
    pass
else: 
    # 使用colorama库来实现颜色输出
    from colorama import init
    init()

# 按照颜色去分类输出
def print红(*kw,**kargs):
    print("\033[0;31m",*kw,"\033[0m",**kargs)
def print绿(*kw,**kargs):
    print("\033[0;32m",*kw,"\033[0m",**kargs)
def print黄(*kw,**kargs):
    print("\033[0;33m",*kw,"\033[0m",**kargs)
def print蓝(*kw,**kargs):
    print("\033[0;34m",*kw,"\033[0m",**kargs)
def print紫(*kw,**kargs):
    print("\033[0;35m",*kw,"\033[0m",**kargs)
def print靛(*kw,**kargs):
    print("\033[0;36m",*kw,"\033[0m",**kargs)

def print亮红(*kw,**kargs):
    print("\033[1;31m",*kw,"\033[0m",**kargs)
def print亮绿(*kw,**kargs):
    print("\033[1;32m",*kw,"\033[0m",**kargs)
def print亮黄(*kw,**kargs):
    print("\033[1;33m",*kw,"\033[0m",**kargs)
def print亮蓝(*kw,**kargs):
    print("\033[1;34m",*kw,"\033[0m",**kargs)
def print亮紫(*kw,**kargs):
    print("\033[1;35m",*kw,"\033[0m",**kargs)
def print亮靛(*kw,**kargs):
    print("\033[1;36m",*kw,"\033[0m",**kargs)



def print亮红(*kw,**kargs):
    print("\033[1;31m",*kw,"\033[0m",**kargs)
def print亮绿(*kw,**kargs):
    print("\033[1;32m",*kw,"\033[0m",**kargs)
def print亮黄(*kw,**kargs):
    print("\033[1;33m",*kw,"\033[0m",**kargs)
def print亮蓝(*kw,**kargs):
    print("\033[1;34m",*kw,"\033[0m",**kargs)
def print亮紫(*kw,**kargs):
    print("\033[1;35m",*kw,"\033[0m",**kargs)
def print亮靛(*kw,**kargs):
    print("\033[1;36m",*kw,"\033[0m",**kargs)

# 为了方便使用，再定义一些别名  
print_red = print红
print_green = print绿
print_yellow = print黄
print_blue = print蓝
print_purple = print紫
print_indigo = print靛

print_bold_red = print亮红
print_bold_green = print亮绿
print_bold_yellow = print亮黄
print_bold_blue = print亮蓝
print_bold_purple = print亮紫
print_bold_indigo = print亮靛
    
if not stdout.isatty():
    # 重定向，不是终端，不输出颜色
    print红 = print
    print绿 = print
    print黄 = print
    print蓝 = print
    print紫 = print
    print靛 = print
    print亮红 = print
    print亮绿 = print
    print亮黄 = print
    print亮蓝 = print
    print亮紫 = print
    print亮靛 = print
    print_red = print
    print_green = print
    print_yellow = print
    print_blue = print
    print_purple = print
    print_indigo = print
    print_bold_red = print
    print_bold_green = print
    print_bold_yellow = print
    print_bold_blue = print
    print_bold_purple = print
    print_bold_indigo = print

if __name__ == "__main__":
    print_blue("蓝色")
    print_bold_blue("亮蓝色")
    print_red("红色")
    print_bold_red("亮红色")