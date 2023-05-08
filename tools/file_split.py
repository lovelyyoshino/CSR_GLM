# -*- coding:utf-8 -*-
from color import print亮红, print亮绿, print亮蓝

"""
将文本按照段落分隔符分割开，生成带有段落标签的HTML代码。
"""
def text_divide_as_html_paragraph(text:str):
    if '```' in text:
        # careful input
        return text
    else:
        # wtf input
        lines = text.split("\n")
        for i, line in enumerate(lines):
            lines[i] = lines[i].replace(" ", "&nbsp;")
        text = "</br>".join(lines)
        return text

"""
将文本文档按照token的limit数量进行切割，返回切割后的文本列表。
"""
def breakdown_txt_to_satisfy_token_limit(txt:str, get_token_fn, limit:int, must_break_at_empty_line):
    if get_token_fn(txt) <= limit: # 如果长度小于限制，则直接返回
        return [txt]
    else:
        cnt = 0
        lines = txt.split('\n') # 按照换行符分割,lines是一个列表,用于存储每一行的内容
        estimated_line_cut = limit / get_token_fn(txt) * len(lines)# 估计需要切割的行数
        estimated_line_cut = int(estimated_line_cut)# 取整
        prev = ""
        post = ""
        for cnt in reversed(range(estimated_line_cut)):# 从后往前遍历
            if must_break_at_empty_line:# 如果必须在空行处切割
                if lines[cnt] != "":# 如果不是空行，则继续
                    continue
            prev = "\n".join(lines[:cnt])# 将前cnt行拼接起来
            post = "\n".join(lines[cnt:])# 将后面的拼接起来
            print(get_token_fn(prev))
            if get_token_fn(prev) < limit:# 如果前面的长度小于限制，则跳过，并取前面的信息
                break
        if cnt == 0:# 如果切割到第一行，则报错，并直接返回
            if must_break_at_empty_line == False:
                print亮蓝("无法再切割")
                prev = "\n".join('')# 将前cnt行拼接起来
                return prev
            return breakdown_txt_to_satisfy_token_limit(txt,get_token_fn, limit, must_break_at_empty_line=False)#直到没办法再切割为止，然后就返回

        # 列表递归接龙
        result = [prev]# 获取前面的部分
        result.extend(breakdown_txt_to_satisfy_token_limit(post, get_token_fn, limit-get_token_fn(prev),must_break_at_empty_line))# 一次分割一段，然后获取后面的部分，并递归
        return result

if __name__ == "__main__":
    txt = "三维点云配准是标定、定位、建图和环境重建的等任务中的关键任务。有两种主流的点云配准方法: 广义迭代最近邻方法GICP和正态分布变换NDT方法。GICP算法扩展了经典的ICP算法，通过计算分布到分布的形式提高了配准精度。NDT利用体素化方法避免高昂的最近邻搜索，提高处理速度。由于GICP和其他ICP算法的变种均依赖于最近邻搜索，这使得很难在计算资源受限的计算机中实时的处理大量点云数据。而NDT通常对体素的分辨率大小非常敏感。最佳的体素分辨率取决于环境和传感器属性，如果选择不当的分辨率，则NDT的精度将大幅降低。本文的通过聚合每个体素内所有点的分布，使得体素化的过程更为鲁棒。相比于NDT从点的空间位置估计体素的分布，本文的体素化方法即使体素中有很少的点，也能够产生有效的体素分布。这也使得算法对体素分辨率的改变更加鲁棒。VGICP论文内容写得还是比较详细充实的，从作者的归纳来看论文的贡献有三个方面：\r\n \
        \r\n\
        1. 首先，提出了一种多点分布聚合方法来从较少的点稳健估计体素的分布。\r\n\
        2. 其次，提出了VGICP算法，它与GICP一样精确，但比现有方法快得多。\r\n\
        3. 第三，代码开源，并且代码实现了包含了所提出的VGICP以及GICP。"
    def get_token_fn(txt:str): return len(txt)

    result = breakdown_txt_to_satisfy_token_limit(txt,get_token_fn, 200,False)
    print(result)

    result = breakdown_txt_to_satisfy_token_limit(txt,get_token_fn, 500,False)
    print(result)

    result = breakdown_txt_to_satisfy_token_limit(txt,get_token_fn, 500,True)
    print(result)

    # result = text_divide_as_html_paragraph(txt)
    # print(result)