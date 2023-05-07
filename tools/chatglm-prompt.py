#WORD Version
import openai
import pandas as pd
import json
import re
import tempfile
import PyPDF2
import pdf2image
import os
import time

"""
pdf内容读取，目前是按照@作为分割符的，后续可以考虑按照。号或者其他符号作为分割符
"""
def pdf_reader(pdf_file):
    #images = pdf2image.convert_from_path(pdf_file)
    with open(pdf_file, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file) 
        doc_text = ""
        for page_num in range(len(pdf_reader.pages)):
            doc_text += pdf_reader.pages[page_num].extract_text()
    prompts = doc_text.split("@")
    return prompts

"""
markdown内容读取，目前是按照@作为分割符的，后续可以考虑按照。号或者其他符号作为分割符
"""
def mk_reader(mk_file):
    with open(mk_file, "r", encoding="utf-8") as file:
        content = file.read()
        prompts = content.split("@")
    return content

"""
目前使用的是gpt-3.5-turbo模型，这部分需要试用情况调整提问prompt的长度，以及temperature的值
"""
def openai_auto_prompts(prompts):
    # 使用gpt-3.5-turbo模型生成问题和答案
    prompt_list = []
    completion_list = []
    for prompt in prompts:
        question_prompt = "你现在是一个提问专家，请根据下面的一段文字用中文提炼出一个问题："+prompt
        if len(question_prompt) > 0:
            print("Starting........")
            response_question = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": question_prompt}
            ],
            max_tokens=300,
            deployment_id = None,
            temperature= 0.7
            )
            print("Waiting............")
            choices_question = response_question["choices"]  # type: ignore
            completion_question = choices_question[0].message.content.strip()
            print(f"Question: {completion_question}")
            time.sleep(20)#需要根据实际情况调整等待时间

            answer_prompt = "你现在是一个专业的机器人专家。请结合下面的文字"+prompt+"。并最终回答下面的问题："+completion_question
            response_answer = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": answer_prompt}
            ],
            max_tokens=200,
            deployment_id = None,
            temperature= 0.7
            )
            choices_answer = response_answer["choices"]
            completion_answer = choices_answer[0].message.content.strip()
            print(f"Answer: {completion_answer}")
            prompt_list.append(completion_question)
            completion_list.append(completion_answer)
            time.sleep(20)

    # 使用的是chatglm输出的结果格式
    output_df = pd.DataFrame({
        "instruction": "方案设计",#这个需要根据实际情况调整，对应书籍/文章分类
        "input": prompt_list,
        "output": completion_list,
    })

    # export as JSON file
    output_dict = output_df.to_json(orient="records",force_ascii=False)
    with open("test.json", "w",encoding='utf-8') as json_file:
        json.dump(output_dict, ensure_ascii=False, indent=4, fp=json_file)


if __name__ == "__main__":
    # set up OpenAI API credentials
    openai.api_key = "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"
    # initialize empty lists for prompts and completions
    str_name = "市场需求分析.txt"
    file_extension = os.path.splitext(str_name)[1].lower()
    if file_extension == ".pdf":
        print("Read PDF")
        prompts = pdf_reader(str_name)
        openai_auto_prompts(prompts)
    elif file_extension == ".md":
        print("Read MD")
        prompts = mk_reader(str_name)
        openai_auto_prompts(prompts)
    elif file_extension == ".txt":
        print("Read TXT")
        prompts = mk_reader(str_name)
        openai_auto_prompts(prompts)
