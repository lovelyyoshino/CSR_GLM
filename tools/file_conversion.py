# -*- coding: utf-8 -*-
import markdown
from latex2mathml.converter import convert as tex2mathml
from functools import lru_cache
import re
import requests
import get_confs

############## 读取输出文件信息  ###############
"""
读取文件内容，txt、md和html格式的文件都可以读取
"""


def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    return content


"""
写入文件内容，txt、md和html格式的文件都可以写入
"""


def write_file(file_path, content):
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)


"""
这个函数是用来获取指定目录下所有指定类型（如.md）的文件，并且对于网络上的文件，也可以获取它。
下面是对每个参数和返回值的说明：
参数 
- txt: 路径或网址，表示要搜索的文件或者文件夹路径或网络上的文件。 
- type: 字符串，表示要搜索的文件类型。默认是.md。
返回值 
- success: 布尔值，表示函数是否成功执行。 
- file_manifest: 文件路径列表，里面包含以指定类型为后缀名的所有文件的绝对路径。 
- project_folder: 字符串，表示文件所在的文件夹路径。如果是网络上的文件，就是临时文件夹的路径。
该函数详细注释已添加，请确认是否满足您的需要。
"""


def get_files_from_everything(txt: str, type):  # type='.md'
    import glob
    import os

    success = True
    if txt.startswith('http'):
        # 网络的远程文件
        proxies, = get_conf('proxies')
        r = requests.get(txt, proxies=proxies)
        with open('./gpt_log/temp'+type, 'wb+') as f:
            f.write(r.content)
        project_folder = './gpt_log/'
        file_manifest = ['./gpt_log/temp'+type]
    elif txt.endswith(type):
        # 直接给定文件
        file_manifest = [txt]
        project_folder = os.path.dirname(txt)
    elif os.path.exists(txt):
        # 本地路径，递归搜索
        project_folder = txt
        file_manifest = [f for f in glob.glob(
            f'{project_folder}/**/*'+type, recursive=True)]  # 读取所有文件
        if len(file_manifest) == 0:
            success = False
    else:
        project_folder = None
        file_manifest = []
        success = False

    return success, file_manifest, project_folder


############## 处理文件信息  ###############

"""
将txt文件转换为markdown格式的文件
"""


def regular_txt_to_markdown(text: str):
    """
    将普通文本转换为Markdown格式的文本。
    """
    text = text.replace('\n', '\n\n')  # 段落分隔符替换
    text = text.replace('\n\n\n', '\n\n')
    text = text.replace('\n\n\n', '\n\n')
    return text


"""
清除txt文档中的换行符
"""


def clear_line_break(txt: str):
    txt = txt.replace('\n', ' ')
    txt = txt.replace('  ', ' ')
    txt = txt.replace('  ', ' ')
    return txt


"""
将Markdown格式的文本转换为HTML格式。如果包含数学公式，则先将公式转换为HTML格式。
"""


@lru_cache(maxsize=128)  # 使用 lru缓存 加快转换速度
def markdown_convertion(txt: str):
    pre = '<div class="markdown-body">'
    suf = '</div>'
    if txt.startswith(pre) and txt.endswith(suf):  # 发现已经转化过HTML格式，则不需要再次转化
        return txt

    markdown_extension_configs = {
        'mdx_math': {
            'enable_dollar_delimiter': True,
            'use_gitlab_delimiters': False,
        },
    }  # 配置markdown扩展
    # 正则表达式，用于匹配数学公式
    find_equation_pattern = r'<script type="math/tex(?:.*?)>(.*?)</script>'

    # 将markdown中的tex格式转化为mathml，并进行异常捕获，避免程序崩溃
    def tex2mathml_catch_exception(content, *args, **kwargs):
        try:
            # 将tex转换为mathml内容的输出
            content = tex2mathml(content, *args, **kwargs)
        except:
            content = content
        return content

    # 先处理math公式，不需要调用tex2mathml进行渲染
    def replace_math_no_render(match: re.Match):
        content = match.group(1)  # 匹配到的内容
        if 'mode=display' in match.group(0):  # 如果是行间公式
            content = content.replace('\n', '</br>')  # 将换行符替换为</br>
            # 将公式用$$包裹起来
            return f"<font color=\"#00FF00\">$$</font><font color=\"#FF00FF\">{content}</font><font color=\"#00FF00\">$$</font>"
        else:
            # 将公式用$包裹起来
            return f"<font color=\"#00FF00\">$</font><font color=\"#FF00FF\">{content}</font><font color=\"#00FF00\">$</font>"

    def replace_math_render(match: re.Match):  # 然后调用tex2mathml来渲染这个公式
        content = match.group(1)  # 匹配到的内容
        if 'mode=display' in match.group(0):  # 如果是行间公式
            if '\\begin{aligned}' in content:  # 如果是多行公式
                content = content.replace(
                    '\\begin{aligned}', '\\begin{array}')  # 将aligned环境替换为array环境
                content = content.replace('\\end{aligned}', '\\end{array}')
                content = content.replace('&', ' ')  # 将&替换为空格
            content = tex2mathml_catch_exception(
                content, display="block")  # 调用tex2mathml将tex转换为mathml
            return content
        else:
            return tex2mathml_catch_exception(content)

    """
    解决一个mdx_math的bug（单$包裹begin命令时多余<script>）
    """
    def markdown_bug_hunt(content):
        content = content.replace('<script type="math/tex">\n<script type="math/tex; mode=display">',
                                  '<script type="math/tex; mode=display">')  # 使用正则表达式替换，避免多出现一个<script type="math/tex">
        # 使用正则表达式替换，避免多出现一个</script>
        content = content.replace('</script>\n</script>', '</script>')
        return content

    def no_code(txt):  # 判断是否包含代码段```，如果包含则返回False，否则返回True
        if '```' not in txt:  # 没有代码段```的标识
            return True
        else:
            if '```reference' in txt:
                return True    # newbing的策略，reference不算代码段
            else:
                return False

    if ('$' in txt) and no_code(txt):  # 有$标识的公式符号，且没有代码段```的标识
        # 将所有内容转换为HTML格式
        split = markdown.markdown(text='---')
        convert_stage_1 = markdown.markdown(text=txt, extensions=[
                                            'mdx_math', 'fenced_code', 'tables', 'sane_lists'], extension_configs=markdown_extension_configs)  # 将markdown转换为HTML格式
        convert_stage_1 = markdown_bug_hunt(
            convert_stage_1)  # 先解决一下markdown中的bug
        # re.DOTALL: 创建'.'这个特殊的字符，它可以匹配任何字符，包括换行符; 没有这个标志, '.' 将匹配除换行符之外的任何内容。换句话说，原本符号“.”遇到换行符就停了，现在打上re.DOTALL标识后，它遇到换行符也不停止了，从而实现了跨行匹配：https://zhuanlan.zhihu.com/p/447968943
        # 1. 转换为易于复制的Tex(不渲染这个数学公式)
        # 这里主要是使用re.subn来执行搜索替换，但是不是返回替换后的字符串，而是返回一个元组，元组的第一个元素是替换后的字符串，第二个元素是执行了多少次替换
        # re.subn中第一个参数是正则表达式，第二个参数是替换函数，第三个参数是要替换的字符串，第四个参数是标志
        convert_stage_2_1, n = re.subn(
            find_equation_pattern, replace_math_no_render, convert_stage_1, flags=re.DOTALL)
        # 2. 转换为渲染后的方程
        convert_stage_2_2, n = re.subn(
            find_equation_pattern, replace_math_render, convert_stage_1, flags=re.DOTALL)
        return pre + f'{split}' + convert_stage_2_2 + suf  # 将转换后的内容放在pre和suf中间
    else:
        return pre + markdown.markdown(txt, extensions=['fenced_code', 'codehilite', 'tables', 'sane_lists']) + suf


if __name__ == "__main__":
    success, file_manifest, project_folder = get_files_from_everything(
        "/home/yc/Desktop/", ".md")
    if success:
        print("project_folder:", project_folder,
              ",file_manifest:", file_manifest)
    txt_file = read_file("C://Users//pony//Desktop//bash.txt")
    md_file = regular_txt_to_markdown(txt_file)
    write_file("C://Users//pony//Desktop//CSR_GLM//bash.md", md_file)
    html_file = markdown_convertion(md_file)
    write_file("C://Users//pony//Desktop//CSR_GLM//bash.html", html_file)
