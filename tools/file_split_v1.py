# -*- coding:utf-8 -*-
import fitz
import copy
import re
import numpy as np
from color import print亮红, print亮绿, print亮蓝, print亮黄
from file_conversion import write_file, read_file
import os

"""
将文本按照段落分隔符分割开，生成带有段落标签的HTML代码。
"""

def get_token_fn(txt: str): return len(txt)

def text_divide_as_html_paragraph(text: str):
    if '```' in text:  # 如果文本中拥有代码，则直接返回文本
        return text
    else:
        # wtf input
        lines = text.split("\n")  # 否则对每一行拆分
        for i, line in enumerate(lines):  # 将所有的line完成替换，变为html的段落标签
            lines[i] = lines[i].replace(" ", "&nbsp;")  # 将空格替换为&nbsp;
        text = "</br>".join(lines)  # 在最后加上</br>标签
        return text


"""
当无法用标点、空行分割时，我们用最暴力的方法切割
"""

def txt_force_breakdown(txt: str, get_token_fn, limit: int):
    for i in reversed(range(len(txt))):  # 从文本最后往前遍历
        if get_token_fn(str(txt[:i])) < limit:  # 如果长度小于限制，则直接返回
            return txt[:i], True  # 返回前面的部分和后面的部分
    return txt, False


"""
将文本文档按照token的limit数量进行切割，返回切割后的文本内容。主要针对文本文档
"""


def breakdown_txt_to_satisfy_token_limit(txt: str, get_token_fn, limit: int, must_break_at_empty_line):
    if get_token_fn(txt) <= limit:  # 如果长度小于限制，则直接返回
        return [txt]
    else:
        cnt = 0
        lines = txt.split('\n')  # 按照换行符分割,lines是一个列表,用于存储每一行的内容
        estimated_line_cut = limit / \
            get_token_fn(txt) * len(lines)  # 估计需要切割的行数
        estimated_line_cut = int(estimated_line_cut)  # 取整
        prev = ""
        post = ""
        for cnt in reversed(range(estimated_line_cut)):  # 从后往前遍历
            if must_break_at_empty_line:  # 如果必须在空行处切割
                if lines[cnt] != "":  # 如果不是空行，则继续
                    continue
            prev = "\n".join(lines[:cnt])  # 将前cnt行拼接起来
            post = "\n".join(lines[cnt:])  # 将后面的拼接起来
            print(get_token_fn(prev))
            if get_token_fn(prev) < limit:  # 如果前面的长度小于限制，则跳过，并取前面的信息
                break
        if cnt == 0:  # 如果切割到第一行，则报错，并直接返回
            if must_break_at_empty_line == False:
                print亮蓝("无法再切割")
                prev = "\n".join('')  # 将前cnt行拼接起来
                return prev
            # 直到没办法再切割为止，然后就返回
            return breakdown_txt_to_satisfy_token_limit(txt, get_token_fn, limit, must_break_at_empty_line=False)

        # 列表递归接龙
        result = [prev]  # 获取前面的部分
        result.extend(breakdown_txt_to_satisfy_token_limit(post, get_token_fn, limit -
                      get_token_fn(prev), must_break_at_empty_line))  # 一次分割一段，然后获取后面的部分，并递归
        return result



"""
将txt的文件按照token的limit数量进行切割，返回切割后的文本。
"""

def breakdown_txt_to_satisfy_token_limit_using_advance_method(txt: str, get_token_fn, limit: int):
    """
    cut函数，四个函数分别为：要切割的文本，是否必须在空行处切割，是否暴力切割
    """
    def cut(txt_tocut: str, limit: int, must_break_at_empty_line: bool, break_anyway=False):
        if get_token_fn(txt_tocut) <= limit:  # 如果长度小于限制，则直接返回
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
                    prev, _ = txt_force_breakdown(
                        txt_tocut, get_token_fn, limit)
                    post = "\n".join(txt_tocut[len(prev):])
                    return [prev]
                else:
                    return ""
            # 列表递归接龙
            result = [prev]
            result.extend(
                cut(post, limit, must_break_at_empty_line, break_anyway=break_anyway))
            return result
    try:
        # 第1次尝试，将双空行（\n\n）作为切分点
        res = cut(txt, limit, must_break_at_empty_line=True)
        if len(res) != 0:
            return res
        raise RuntimeError(f"存在一行极长的文本！")
    except RuntimeError:
        try:
            # 第2次尝试，将单空行（\n）作为切分点
            res = cut(txt, limit, must_break_at_empty_line=False)
            if len(res) != 0:
                return res
            raise RuntimeError(f"存在一行极长的文本！")
        except RuntimeError:
            try:
                # 第3次尝试，将英文句号（.）作为切分点
                # 这个中文的句号是故意的，作为一个标识而存在
                res = cut(txt.replace('.', '。\n'), limit,
                          must_break_at_empty_line=False)
                if len(res) != 0:
                    return [r.replace('。\n', '.') for r in res]
                raise RuntimeError(f"存在一行极长的文本！")
            except RuntimeError as e:
                try:
                    # 第4次尝试，将中文句号（。）作为切分点
                    res = cut(txt.replace('。', '。。\n'), limit,
                              must_break_at_empty_line=False)
                    if len(res) != 0:
                        return [r.replace('。。\n', '。') for r in res]
                    raise RuntimeError(f"存在一行极长的文本！")
                except RuntimeError as e:
                    # 第5次尝试，没办法了，随便切一下敷衍吧
                    return cut(txt, limit, must_break_at_empty_line=False, break_anyway=True)

"""
将txt的文件按照token的limit数量进行切割，返回切割后的文本。
"""

def breakdown_txt_to_satisfy_token_limit_using_advance_method_list(txt, get_token_fn, limit):
    # 递归
    def cut(txt_tocut, must_break_at_empty_line, break_anyway=False):  
        if get_token_fn(txt_tocut) <= limit:
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
                    break
            status_force = True
            if cnt == 0:
                if break_anyway:
                    prev, status_force = txt_force_breakdown(txt_tocut, get_token_fn, limit)
                    post = "\n".join(txt_tocut[len(prev):])
                else:
                    raise RuntimeError(f"存在一行极长的文本！{txt_tocut}")
            # print(len(post))
            # 列表递归接龙
            result = [prev]
            if status_force == True:
                result.extend(cut(post, must_break_at_empty_line, break_anyway=break_anyway))
            return result
    try:
        # 第1次尝试，将双空行（\n\n）作为切分点
        return cut(txt, must_break_at_empty_line=True)
    except RuntimeError:
        try:
            # 第2次尝试，将单空行（\n）作为切分点
            return cut(txt, must_break_at_empty_line=False)
        except RuntimeError:
            try:
                # 第3次尝试，将英文句号（.）作为切分点
                res = cut(txt.replace('.', '。\n'), must_break_at_empty_line=False) # 这个中文的句号是故意的，作为一个标识而存在
                return [r.replace('。\n', '.') for r in res]
            except RuntimeError as e:
                try:
                    # 第4次尝试，将中文句号（。）作为切分点
                    res = cut(txt.replace('。', '。。\n'), must_break_at_empty_line=False)
                    return [r.replace('。。\n', '。') for r in res]
                except RuntimeError as e:
                    # 第5次尝试，没办法了，随便切一下敷衍吧
                    return cut(txt, must_break_at_empty_line=False, break_anyway=True)

"""
这个函数用于分割pdf，用了很多trick，逻辑较乱，效果奇好

**输入参数说明**
- `fp`：需要读取和清理文本的pdf文件路径

**输出参数说明**
- `meta_txt`：清理后的文本内容字符串
- `page_one_meta`：第一页清理后的文本内容列表

**函数功能**
读取pdf文件并清理其中的文本内容，清理规则包括：
- 提取所有块元的文本信息，并合并为一个字符串
- 去除短块（字符数小于100）并替换为回车符
- 清理多余的空行
- 合并小写字母开头的段落块并替换为空格
- 清除重复的换行
- 将每个换行符替换为两个换行符，使每个段落之间有两个换行符分隔
"""


def read_and_clean_pdf_text(fp):

    fc = 0  # Index 0 文本
    fs = 1  # Index 1 字体
    fb = 2  # Index 2 框框
    REMOVE_FOOT_NOTE = True  # 是否丢弃掉 不是正文的内容 （比正文字体小，如参考文献、脚注、图注等）
    # 小于正文的0.95时，判定为不是正文（有些文章的正文部分字体大小不是100%统一的，有肉眼不可见的小变化）
    REMOVE_FOOT_FFSIZE_PERCENT = 0.95

    def __primary_ffsize(l):
        """
        提取文本块主字体
        """
        fsize_statiscs = {}
        for wtf in l['spans']:
            if wtf['size'] not in fsize_statiscs:
                fsize_statiscs[wtf['size']] = 0
            fsize_statiscs[wtf['size']] += len(wtf['text'])
        return max(fsize_statiscs, key=fsize_statiscs.get)

    def __ffsize_same(a, b):
        """
        提取字体大小是否近似相等
        """
        return abs((a-b)/max(a, b)) < 0.02

    def __remove_useless_space(meta_txt):
        """
        删除多余的空格
        """
        for index in reversed(range(1, len(meta_txt))):
            if meta_txt[index] == '\n' and meta_txt[index-1] == '\n':
                meta_txt.pop(index)
        return meta_txt

    def __remove_less_paragraph_than_100(meta_txt):
        """
        当段落长度小于100时，认为是无效段落，替换为空行
        """
        for index, block_txt in enumerate(meta_txt):
            if len(block_txt) < 100:  # 一般认为一行字数不会少于100
                meta_txt[index] = '\n'
        return meta_txt

    def __merge_paragraph_start_lowercase(meta_txt):
        """
        合并小写开头的段落块
        """
        def __starts_with_lowercase_word(s):
            pattern = r"^[a-z]+"
            match = re.match(pattern, s)
            if match:
                return True
            else:
                return False
        for _ in range(100):
            for index, block_txt in enumerate(meta_txt):
                if __starts_with_lowercase_word(block_txt):
                    if meta_txt[index-1] != '\n':
                        meta_txt[index-1] += ' '
                    else:
                        meta_txt[index-1] = ''
                    meta_txt[index-1] += meta_txt[index]
                    meta_txt[index] = '\n'
        return meta_txt

    with fitz.open(fp) as doc:  # 读取pdf内容
        meta_txt = []
        meta_font = []

        meta_line = []
        meta_span = []
        ############################## <第 1 步，搜集初始信息> ##################################
        for index, page in enumerate(doc):
            # file_content += page.get_text()
            text_areas = page.get_text("dict")  # 获取页面上的文本信息
            for t in text_areas['blocks']:
                if 'lines' in t:
                    pf = 998
                    for l in t['lines']:
                        txt_line = "".join([wtf['text'] for wtf in l['spans']])
                        if len(txt_line) == 0:
                            continue
                        pf = __primary_ffsize(l)
                        meta_line.append([txt_line, pf, l['bbox'], l])
                        for wtf in l['spans']:  # for l in t['lines']:
                            meta_span.append(
                                [wtf['text'], wtf['size'], len(wtf['text'])])
                    # meta_line.append(["NEW_BLOCK", pf])
            # 块元提取                           for each word segment with in line                       for each line         cross-line words                          for each block
            meta_txt.extend([" ".join(["".join([wtf['text'] for wtf in l['spans']]) for l in t['lines']]).replace(
                '- ', '') for t in text_areas['blocks'] if 'lines' in t])
            meta_font.extend([np.mean([np.mean([wtf['size'] for wtf in l['spans']])
                             for l in t['lines']]) for t in text_areas['blocks'] if 'lines' in t])
            if index == 0:
                page_one_meta = [" ".join(["".join([wtf['text'] for wtf in l['spans']]) for l in t['lines']]).replace(
                    '- ', '') for t in text_areas['blocks'] if 'lines' in t]

        ############################## <第 2 步，获取正文主字体> ##################################
        fsize_statiscs = {}
        for span in meta_span:
            if span[1] not in fsize_statiscs:
                fsize_statiscs[span[1]] = 0
            fsize_statiscs[span[1]] += span[2]
        main_fsize = max(fsize_statiscs, key=fsize_statiscs.get)
        if REMOVE_FOOT_NOTE:
            give_up_fize_threshold = main_fsize * REMOVE_FOOT_FFSIZE_PERCENT

        ############################## <第 3 步，切分和重新整合> ##################################
        mega_sec = []
        sec = []
        for index, line in enumerate(meta_line):
            if index == 0:
                sec.append(line[fc])
                continue
            if REMOVE_FOOT_NOTE:
                if meta_line[index][fs] <= give_up_fize_threshold:
                    continue
            if __ffsize_same(meta_line[index][fs], meta_line[index-1][fs]):
                # 尝试识别段落
                if meta_line[index][fc].endswith('.') and\
                    (meta_line[index-1][fc] != 'NEW_BLOCK') and \
                        (meta_line[index][fb][2] - meta_line[index][fb][0]) < (meta_line[index-1][fb][2] - meta_line[index-1][fb][0]) * 0.7:
                    sec[-1] += line[fc]
                    sec[-1] += "\n\n"
                else:
                    sec[-1] += " "
                    sec[-1] += line[fc]
            else:
                if (index+1 < len(meta_line)) and \
                        meta_line[index][fs] > main_fsize:
                    # 单行 + 字体大
                    mega_sec.append(copy.deepcopy(sec))
                    sec = []
                    sec.append("# " + line[fc])
                else:
                    # 尝试识别section
                    if meta_line[index-1][fs] > meta_line[index][fs]:
                        sec.append("\n" + line[fc])
                    else:
                        sec.append(line[fc])
        mega_sec.append(copy.deepcopy(sec))

        finals = []
        for ms in mega_sec:
            final = " ".join(ms)
            final = final.replace('- ', ' ')
            finals.append(final)
        meta_txt = finals

        ############################## <第 4 步，乱七八糟的后处理> ##################################
        meta_txt = __remove_less_paragraph_than_100(meta_txt)

        meta_txt = __remove_useless_space(meta_txt)

        meta_txt = __merge_paragraph_start_lowercase(meta_txt)
        meta_txt = __remove_useless_space(meta_txt)

        meta_txt = '\n'.join(meta_txt)
        # 清除重复的换行
        for _ in range(5):
            meta_txt = meta_txt.replace('\n\n', '\n')

        # 换行 -> 双换行
        meta_txt = meta_txt.replace('\n', '\n\n')

        ############################## <第 5 步，展示分割效果> ##################################
        # for f in finals:
        #     print亮黄(f)
        #     print亮绿('***************************')

    return meta_txt, page_one_meta


if __name__ == "__main__":
    txt = "三维点云配准是标定、定位、建图和环境重建的等任务中的关键任务。有两种主流的点云配准方法: 广义迭代最近邻方法GICP和正态分布变换NDT方法。\r\n \
        GICP算法扩展了经典的ICP算法，通过计算分布到分布的形式提高了配准精度。NDT利用体素化方法避免高昂的最近邻搜索，提高处理速度。由于GICP和其他ICP算法的变种均依赖于最近邻搜索，这使得很难在计算资源受限的计算机中实时的处理大量点云数据。而NDT通常对体素的分辨率大小非常敏感。最佳的体素分辨率取决于环境和传感器属性，如果选择不当的分辨率，则NDT的精度将大幅降低。本文的通过聚合每个体素内所有点的分布，使得体素化的过程更为鲁棒。相比于NDT从点的空间位置估计体素的分布，本文的体素化方法即使体素中有很少的点，也能够产生有效的体素分布。这也使得算法对体素分辨率的改变更加鲁棒。VGICP论文内容写得还是比较详细充实的，从作者的归纳来看论文的贡献有三个方面：\r\n \
        \r\n\
        1. 首先，提出了一种多点分布聚合方法来从较少的点稳健估计体素的分布。\r\n\
        2. 其次，提出了VGICP算法，它与GICP一样精确，但比现有方法快得多。\r\n\
        3. 第三，代码开源，并且代码实现了包含了所提出的VGICP以及GICP。"
    # txt = '1.1市场需求分析从深蓝学院再到各大知识星球的学习和宣传，我们发现确实这些知识付费的形式也越来越受到广大学习者的认可。但是我们发现这类学习除了服务以外，其他都可以被非常便捷的抄袭，这会大大损害作者的收益。而且这类课程基本讲解的还是比较具体的，很多都是带着学院研读等操作，这类方式其实我们完全可以通过大模型的形式来提供非常便捷且合适的学习途径。现在的机器人从业者没有一个非常便捷的可以快速获取并学习的大模型网站，而这个就是我们需要努力的方向。1.2．竞品市场分析（己有竞品的数据分析)：1. Chatgpt:可以根据用户输入的上下文对话生成逻辑连贯和深度的回复。chatgpt能生成更具逻辑性和连贯性的长篇回复。但是其训练模型更多的是英语，所以对于中文还是显得比较生硬和机械，不够贴近人的说话方式。同时无法提供非常专业的问题答案----比如说机器人行业的知识。2.文心一言:文心一言是百度开发的一款AI 聊天机器人。它通过深度学习和大数据技术，可以根据用的文本输入自动生成一句符合语境的回复。它的优点是回复流畅和贴近人话，但生成的回复比较肤浅和欠缺深度，不太能进行深入的对话。同时无法提供非常专业的问题答案---比如说机器人行业的知识。\n'

    def get_token_fn(txt: str): return len(txt)

    result, status = txt_force_breakdown(txt, get_token_fn, 100)
    if status:
        print("1:", result)

    result = breakdown_txt_to_satisfy_token_limit(txt, get_token_fn, 100, False)
    print("2:", result)

    result = breakdown_txt_to_satisfy_token_limit(
        txt, get_token_fn, 100, False)
    print("3:", result)

    result = breakdown_txt_to_satisfy_token_limit(
        txt, get_token_fn, 100, False)
    print("4:", result)

    result = breakdown_txt_to_satisfy_token_limit(txt, get_token_fn, 100, True)
    print("5:", result)

    result = breakdown_txt_to_satisfy_token_limit_using_advance_method(
        txt, get_token_fn, 100)
    print("6 段落:", result)
    start_time = os.times()
    print(start_time)
    result = breakdown_txt_to_satisfy_token_limit_using_advance_method_list(
        txt, get_token_fn, 2)
    print(os.times()-start_time)
    print("7 句号:", result)
   
    result = breakdown_txt_to_satisfy_token_limit_using_advance_method(
        txt, get_token_fn, 100)   
    print("8 force:", result)

    # result, one_paragraph = read_and_clean_pdf_text(
    #     "C://Users//pony//Desktop//bash.pdf")
    # print("9: pdf", one_paragraph)
    # write_file("./11.md", result)

    # result = text_divide_as_html_paragraph(txt)
    # print("10: html", result)
