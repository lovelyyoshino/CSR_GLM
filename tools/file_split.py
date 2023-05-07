

"""
将文本按照段落分隔符分割开，生成带有段落标签的HTML代码。
"""
def text_divide_paragraph(text:str):
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