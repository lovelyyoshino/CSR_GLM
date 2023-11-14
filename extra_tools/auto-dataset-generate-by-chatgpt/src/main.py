import sys
import os
from tqdm.auto import tqdm
from utils import (
    generate_subtopic,
    generate_question_chatgpt,
    generate_answer_chatgpt, 
    generate_answer_pandora,
    generate_question_pandora,
    generate_question_pandora_by_paragraph,
    generate_question_chatgpt_by_paragraph,
    BudgetTracker,
)
from utils.format import DataSetArguments
tool_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(tool_path)
sys.path.append(tool_path+'\\tools')
sys.setrecursionlimit(100000)  #例如这里设置为十万
from tools.file_split import(
    breakdown_txt_to_satisfy_token_limit_using_advance_method_list,
    breakdown_txt_to_satisfy_token_limit_using_advance_method_list_by_paragraph,
    read_and_clean_pdf_text
)

def get_args(generalization_index:float = 0.75,
        generalization_basic:int = 10,
        number_of_dataset:int = 1,
        topic:str = None,
        dataset_output_path:str = 'dataset',
        proxy:str = None,
        api_key:str = None,
        api_base:str = None,
        api_key_list:list = None,
        tokenBudget:float = 100,
        retry_time:int = 5,
        api_index:int = 0,
        token_limit:int = 100):  #生成args，并对个参数进行赋值
    model_args = DataSetArguments(
        generalization_index=generalization_index,
        generalization_basic=generalization_basic,
        number_of_dataset=number_of_dataset,
        topic=topic,        
        dataset_output_path=dataset_output_path,
        proxy=proxy,
        api_key_list=api_key_list,
        api_index=api_index,
        api_key=api_key_list[api_index],
        api_base = api_base,
        tokenBudget=tokenBudget,
        retryTime=retry_time,
        tokenLimit=token_limit
    )
    return model_args

def get_prompts(file_path:str, limit:int, byparagraph:bool) :  #根据文件名读取文件，并对其进行分割

    def mk_reader(mk_file):
        with open(mk_file, "r", encoding="utf-8") as file:
            content = file.read()
        return content

    def get_token_fn1(txt: str): 
        return len(txt)
    
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == ".pdf":
        print("Read PDF")
        split_text,  page_one_meta= read_and_clean_pdf_text(file_path)
    elif file_extension == ".md":
        print("Read MD")
        split_text = mk_reader(file_path)
    elif file_extension == ".txt":
        print("Read TXT")
        split_text = mk_reader(file_path)
    else:
        print("Read TXT")
        split_text = mk_reader(file_path)

    if byparagraph:
        result = breakdown_txt_to_satisfy_token_limit_using_advance_method_list_by_paragraph(split_text, get_token_fn1, limit)
    else:
        result = breakdown_txt_to_satisfy_token_limit_using_advance_method_list(split_text, get_token_fn1, limit)
    return result

def dataset_generate(prompts:list, args:DataSetArguments, byparagraph:bool):  #根据API和文件分割获得的prompts，生成相关的dataset，并对其进行保存
    budget_tracker = BudgetTracker(total_budget=args.tokenBudget)
    for prompt in prompts:
        args.topic = prompt.replace('\n', '')
        if byparagraph:
            print("generating question......")
            if args.api_key.startswith('fk'):
                args.api_index, questions = generate_question_pandora_by_paragraph(
                                            args=args,
                                            budget_tracker=budget_tracker)
            else:
                questions = generate_question_chatgpt_by_paragraph(
                                            args=args,
                                            budget_tracker=budget_tracker)
            print(str(len(questions)) + ' question are generated')

            # Generate answers for the questions using the API key and update the budget tracker and progress bar
            print("generate_answer")
            if args.api_key.startswith('fk'):
                args.index = generate_answer_pandora(questions=questions,
                                budget_tracker=budget_tracker,
                                args=args)
            else:
                generate_answer_chatgpt(questions=questions,
                                budget_tracker=budget_tracker,
                                args=args)
        else:
            print("generating subtopic......")
            while True:
                args.api_index, sub_topic = generate_subtopic(args=args,
                                            budget_tracker=budget_tracker)
                if len(sub_topic) > 0:
                    break
            
            print(str(len(sub_topic)) + ' subtopics are generated')

            # Generate questions based on the topic, subtopic, and API key
            print("generating question......")
            if args.api_key.startswith('fk'):
                args.api_index, questions = generate_question_pandora(sub_topic=sub_topic,
                                            args=args,
                                            budget_tracker=budget_tracker)
            else:
                questions = generate_question_chatgpt(sub_topic=sub_topic,
                                            args=args,
                                            budget_tracker=budget_tracker)
            print(str(len(questions)) + ' question are generated')

            # Generate answers for the questions using the API key and update the budget tracker and progress bar
            print("generate_answer")
            if args.api_key.startswith('fk'):
                args.index = generate_answer_pandora(questions=questions,
                                budget_tracker=budget_tracker,
                                args=args)
            else:
                generate_answer_chatgpt(questions=questions,
                                budget_tracker=budget_tracker,
                                args=args)
            
def dataset_generate_by_paragraph(prompts:list, args:DataSetArguments):  #根据API和文件分割获得的prompts，生成相关的dataset，并对其进行保存
    budget_tracker = BudgetTracker(total_budget=args.tokenBudget)
    for prompt in prompts:
        args.topic = prompt
        # Generate questions based on the topic, and API key
        print("generating question......")
        if args.api_key.startswith('fk'):
            args.api_index, questions = generate_question_pandora_by_paragraph(
                                        args=args,
                                        budget_tracker=budget_tracker)
        else:
            questions = generate_question_chatgpt_by_paragraph(
                                        args=args,
                                        budget_tracker=budget_tracker)
        print(str(len(questions)) + ' question are generated')

        # Generate answers for the questions using the API key and update the budget tracker and progress bar
        print("generate_answer")
        if args.api_key.startswith('fk'):
            args.index = generate_answer_pandora(questions=questions,
                            budget_tracker=budget_tracker,
                            args=args)
        else:
            generate_answer_chatgpt(questions=questions,
                            budget_tracker=budget_tracker,
                            args=args)

if __name__ == '__main__':
    # args = get_args(api_key="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")
    # args = get_args(api_key="sk-IwHMBXeWyuLRp8kFEWZiNS6IlJJliO8MWXNMyhfXq35hFPEQ", api_base="https://api.chatanywhere.com.cn")
    # args = get_args(api_key="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", api_base="https://openai.451024.xyz/")
    # args = get_args(api_key_list=["fk-XCRfd6OKAWGxX9Lgr7iFkxM-nnpT4L7n0hCgUb9vzSM", "fk--JJiSZdnp6orTQ5XnBCRWXGtM2hb2GeuTdCffyswgZg", "fk-4o6U0arG0AQK4PFW8y5epnfS7w3JBoGUBkudoN5zlXc"], api_index=0)
    # args = get_args(api_key_list=["fk-XCRfd6OKAWGxX9Lgr7iFkxM-nnpT4L7n0hCgUb9vzSM","fk--JJiSZdnp6orTQ5XnBCRWXGtM2hb2GeuTdCffyswgZg", "fk-4o6U0arG0AQK4PFW8y5epnfS7w3JBoGUBkudoN5zlXc","fk-xjGaTrkiJ0ErilWzJ3aUcjhp70zu2RRv1RMDWnLZqzM", "fk-xZHsw62T1-4oVYuRrLo46yKRwvYJcFKqOb4m6n6Nbvo"], api_index=0)
    args = get_args(api_key_list=["fk-XCRfd6OKAWGxX9Lgr7iFkxM-nnpT4L7n0hCgUb9vzSM"], api_index=0)
    prompts = get_prompts('第一章.md', args.tokenLimit, True)
    # prompts = get_prompts('市场分析.txt', args.tokenLimit)
    dataset_generate(prompts, args, True)

