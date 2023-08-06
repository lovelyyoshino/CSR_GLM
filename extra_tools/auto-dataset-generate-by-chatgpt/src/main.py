import sys
import os
from tqdm.auto import tqdm
from utils import (
    generate_subtopic,
    generate_question,
    generate_answer,
    BudgetTracker,
)
from utils.format import DataSetArguments
tool_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(tool_path)
sys.path.append(tool_path+'\\tools')
from tools.file_split import(
    breakdown_txt_to_satisfy_token_limit_using_advance_method_list,
    read_and_clean_pdf_text
)

def get_args(generalization_index:float = 0.75,
        generalization_basic:int = 10,
        number_of_dataset:int = 5,
        topic:str = None,
        dataset_output_path:str = 'dataset',
        proxy:str = None,
        api_key:str = None,
        api_base:str = None,
        tokenBudget:float = 100,
        retry_time:int = 5,
        token_limit:int = 100):  #生成args，并对个参数进行赋值
    model_args = DataSetArguments(
        generalization_index=generalization_index,
        generalization_basic=generalization_basic,
        number_of_dataset=number_of_dataset,
        topic=topic,        
        dataset_output_path=dataset_output_path,
        proxy=proxy,
        api_key=api_key,
        api_base = api_base,
        tokenBudget=tokenBudget,
        retryTime=retry_time,
        tokenLimit=token_limit
    )
    return model_args

def get_prompts(file_path:str, limit:int):  #根据文件名读取文件，并对其进行分割

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
    result = breakdown_txt_to_satisfy_token_limit_using_advance_method_list(split_text, get_token_fn1, limit)
    return result

def dataset_generate(prompts:list, args:DataSetArguments):  #根据API和文件分割获得的prompts，生成相关的dataset，并对其进行保存
    budget_tracker = BudgetTracker(total_budget=args.tokenBudget)
    for prompt in prompts:
        args.topic = prompt
        print("generating subtopic......")
        sub_topic = generate_subtopic(args=args,
                                        budget_tracker=budget_tracker)
        print(str(len(sub_topic)) + ' subtopics are generated')

        # Generate questions based on the topic, subtopic, and API key
        print("generating question......")
        questions = generate_question(sub_topic=sub_topic,
                                        args=args,
                                        budget_tracker=budget_tracker)
        print(str(len(sub_topic)) + ' question are generated')

        # Generate answers for the questions using the API key and update the budget tracker and progress bar
        print("generate_answer")
        generate_answer(questions=questions,
                        budget_tracker=budget_tracker,
                        args=args)

if __name__ == '__main__':
    # args = get_args(api_key="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")
    args = get_args(api_key="sk-IwHMBXeWyuLRp8kFEWZiNS6IlJJliO8MWXNMyhfXq35hFPEQ", api_base="https://api.chatanywhere.com.cn")
    # args = get_args(api_key="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", api_base="https://openai.451024.xyz/")
    # prompts = get_prompts('extra_tools/auto-dataset-generate-by-chatgpt/src/市场分析.txt', args.tokenLimit)
    prompts = get_prompts('市场分析.txt', args.tokenLimit)
    dataset_generate(prompts, args)

