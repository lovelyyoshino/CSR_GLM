# -*- coding:utf-8 -*-
from color import print亮红, print亮绿, print亮蓝

"""
将文本按照段落分隔符分割开，生成带有段落标签的HTML代码。
"""
def text_divide_as_html_paragraph(text:str):
    if '```' in text:# 如果文本中拥有代码，则直接返回文本
        return text
    else:
        # wtf input
        lines = text.split("\n")#否则对每一行拆分
        for i, line in enumerate(lines):#将所有的line完成替换，变为html的段落标签
            lines[i] = lines[i].replace(" ", "&nbsp;")# 将空格替换为&nbsp;
        text = "</br>".join(lines)#在最后加上</br>标签
        return text

"""
当无法用标点、空行分割时，我们用最暴力的方法切割
"""
def txt_force_breakdown(txt:str, get_token_fn,limit:int):
    for i in reversed(range(len(txt))):# 从文本最后往前遍历
        if get_token_fn(txt[:i]) < limit:# 如果长度小于限制，则直接返回
            return txt[:i], True# 返回前面的部分和后面的部分
    return txt, False

"""
将文本文档按照token的limit数量进行切割，返回切割后的文本内容。主要针对文本文档
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

"""
将pdf的文件按照token的limit数量进行切割，返回切割后的文本列表。主要针对pdf文件
"""
def breakdown_txt_to_satisfy_token_limit_for_pdf(txt:str, get_token_fn, limit:int):
    """
    cut函数，四个函数分别为：要切割的文本，是否必须在空行处切割，是否暴力切割
    """
    def cut(txt_tocut:str, limit:int, must_break_at_empty_line:bool, break_anyway=False):  
        if get_token_fn(txt_tocut) <= limit:# 如果长度小于限制，则直接返回
            return [txt_tocut]
        else:
            lines = txt_tocut.split('\n')
            estimated_line_cut = limit / get_token_fn(txt_tocut) * len(lines)
            estimated_line_cut = int(estimated_line_cut)
            cnt = 0
            for cnt in reversed(range(estimated_line_cut)):
                if must_break_at_empty_line:
                    if lines[cnt] != "":
                        continue
                prev = "\n".join(lines[:cnt])
                post = "\n".join(lines[cnt:])
                if get_token_fn(prev) < limit:
                    limit = limit - get_token_fn(prev)
                    break
            if cnt == 0:
                if break_anyway:
                    prev, _ = txt_force_breakdown(txt_tocut, get_token_fn, limit)
                    post = "\n".join(txt_tocut[len(prev):])
                    return [prev]
                else:
                    return ""
            # 列表递归接龙
            result = [prev]
            result.extend(cut(post, limit, must_break_at_empty_line, break_anyway=break_anyway))
            return result
    try:
        # 第1次尝试，将双空行（\n\n）作为切分点
        res =cut(txt,limit, must_break_at_empty_line=True)
        if len(res) != 0:
            return res
        raise RuntimeError(f"存在一行极长的文本！")
    except RuntimeError:
        try:
            # 第2次尝试，将单空行（\n）作为切分点
            res = cut(txt,limit, must_break_at_empty_line=False)
            if len(res) != 0:
                return res
            raise RuntimeError(f"存在一行极长的文本！")
        except RuntimeError:
            try:
                # 第3次尝试，将英文句号（.）作为切分点
                res = cut(txt.replace('.', '。\n'),limit, must_break_at_empty_line=False) # 这个中文的句号是故意的，作为一个标识而存在
                if len(res) != 0:
                    return [r.replace('。\n', '.') for r in res]
                raise RuntimeError(f"存在一行极长的文本！")
            except RuntimeError as e:
                try:
                    # 第4次尝试，将中文句号（。）作为切分点
                    res = cut(txt.replace('。', '。。\n'),limit, must_break_at_empty_line=False)
                    if len(res) != 0:
                        return [r.replace('。。\n', '。') for r in res]
                    raise RuntimeError(f"存在一行极长的文本！")
                except RuntimeError as e:
                    # 第5次尝试，没办法了，随便切一下敷衍吧
                    return cut(txt,limit, must_break_at_empty_line=False, break_anyway=True)



if __name__ == "__main__":
    txt = "三维点云配准是标定、定位、建图和环境重建的等任务中的关键任务。有两种主流的点云配准方法: 广义迭代最近邻方法GICP和正态分布变换NDT方法。\r\n \
        GICP算法扩展了经典的ICP算法，通过计算分布到分布的形式提高了配准精度。NDT利用体素化方法避免高昂的最近邻搜索，提高处理速度。由于GICP和其他ICP算法的变种均依赖于最近邻搜索，这使得很难在计算资源受限的计算机中实时的处理大量点云数据。而NDT通常对体素的分辨率大小非常敏感。最佳的体素分辨率取决于环境和传感器属性，如果选择不当的分辨率，则NDT的精度将大幅降低。本文的通过聚合每个体素内所有点的分布，使得体素化的过程更为鲁棒。相比于NDT从点的空间位置估计体素的分布，本文的体素化方法即使体素中有很少的点，也能够产生有效的体素分布。这也使得算法对体素分辨率的改变更加鲁棒。VGICP论文内容写得还是比较详细充实的，从作者的归纳来看论文的贡献有三个方面：\r\n \
        \r\n\
        1. 首先，提出了一种多点分布聚合方法来从较少的点稳健估计体素的分布。\r\n\
        2. 其次，提出了VGICP算法，它与GICP一样精确，但比现有方法快得多。\r\n\
        3. 第三，代码开源，并且代码实现了包含了所提出的VGICP以及GICP。"
    def get_token_fn(txt:str): return len(txt)

    result,status = txt_force_breakdown(txt,get_token_fn, 50)
    if status:
        print("1:",result)

    result = breakdown_txt_to_satisfy_token_limit(txt,get_token_fn, 50,False)
    print("2:",result)

    result = breakdown_txt_to_satisfy_token_limit(txt,get_token_fn, 200,False)
    print("3:",result)

    result = breakdown_txt_to_satisfy_token_limit(txt,get_token_fn, 500,False)
    print("4:",result)

    result = breakdown_txt_to_satisfy_token_limit(txt,get_token_fn, 500,True)
    print("5:",result)

    result = breakdown_txt_to_satisfy_token_limit_for_pdf(txt,get_token_fn, 500)
    print("6 段落:",result)

    result = breakdown_txt_to_satisfy_token_limit_for_pdf(txt,get_token_fn, 100)
    print("7 句号:",result)

    result = breakdown_txt_to_satisfy_token_limit_for_pdf(txt,get_token_fn, 50)
    print("8 force:",result)
    # result = text_divide_as_html_paragraph(txt)
    # print(result)