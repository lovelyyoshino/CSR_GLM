#WORD Version
import openai
import pandas as pd
import json
import re
import tempfile
import PyPDF2
import pdf2image
import os

def pdf_reader(pdf_file):
    #images = pdf2image.convert_from_path(pdf_file)
    with open(pdf_file, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file) 
        doc_text = ""
        for page_num in range(len(pdf_reader.pages)):
            doc_text += pdf_reader.pages[page_num].extract_text()
    prompts = doc_text.split("@")
    return prompts

def mk_reader(mk_file):
    with open(mk_file, "r", encoding="utf-8") as file:
        content = file.read()
        prompts = content.split("@")
    return content

def openai_auto_prompts(prompts):
    # generate completions using GPT-3 API
    prompt_list = []
    completion_list = []
    for prompt in prompts:
        question_prompt = "你现在是一个提问专家，请根据下面的一段文字用中文提炼出一个问题："+prompt
        if len(question_prompt) > 0:
            print("Starting........")
            # response = openai.Completion.create(
            #     engine="gpt-3.5-turbo",
            #     prompt=question_prompt,
            #     max_tokens=300,
            #     n=1,
            #     stop=None,
            #     temperature=0.7
            # )
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
            #completion = response.choices[0].text.strip()
            choices_question = response_question["choices"]  # type: ignore
            completion_question = choices_question[0].message.content.strip()
            print(f"Question: {completion_question}")

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
            break

    # create data frame for output
    output_df = pd.DataFrame({
        "instruction": "方案设计",
        "input": prompt_list,
        "output": completion_list,
    })

    # # export as Excel file
    # output_df.to_excel("<insert output file path here>.xlsx", index=False)

    # export as JSON file
    output_dict = output_df.to_json(orient="records",force_ascii=False)
    with open("test.json", "w",encoding='utf-8') as json_file:
        json.dump(output_dict, ensure_ascii=False, indent=4, fp=json_file)

    # # export as text file
    # with open("<insert output file path here>.txt", "w") as text_file:
    #     for i in range(len(prompt_list)):
    #         prompt = prompt_list[i]
    #         completion = completion_list[i]
    #         text_file.write(f"{{\"prompt\": \"{prompt}\", \"completion\": \"{completion}\"}}\n")

if __name__ == "__main__":
    # set up OpenAI API credentials
    openai.api_key = "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"
    # initialize empty lists for prompts and completions
    str_name = "市场需求分析.txt"
    file_extension = os.path.splitext(str_name)[1].lower()
    if file_extension == ".pdf":
        print("Read PDF")
        prompts = pdf_reader(str_name)
        #openai_auto_prompts(prompts)
    elif file_extension == ".md":
        print("Read MD")
        prompts = mk_reader(str_name)
        #openai_auto_prompts(prompts)
    elif file_extension == ".txt":
        print("Read TXT")
        prompts = mk_reader(str_name)
        print(prompts)
        #openai_auto_prompts(prompts)
