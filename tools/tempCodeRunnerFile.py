    #     try:
    #     # 第2次尝试，将单空行（\n）作为切分点
    #     return cut(txt,limit, must_break_at_empty_line=False)
    # except RuntimeError:
    #     try:
    #         print("3")
    #         # 第3次尝试，将英文句号（.）作为切分点
    #         res = cut(txt.replace('.', '。\n'),limit, must_break_at_empty_line=False) # 这个中文的句号是故意的，作为一个标识而存在
    #         return [r.replace('。\n', '.') for r in res]
    #     except RuntimeError as e:
    #         try:
    #             print("4")
    #             # 第4次尝试，将中文句号（。）作为切分点
    #             res = cut(txt.replace('。', '。。\n'),limit, must_break_at_empty_line=False)
    #             return [r.replace('。。\n', '。') for r in res]
    #         except RuntimeError as e:
    #             # 第5次尝试，没办法了，随便切一下敷衍吧
    #             return cut(txt,limit, must_break_at_empty_line=False, break_anyway=True)