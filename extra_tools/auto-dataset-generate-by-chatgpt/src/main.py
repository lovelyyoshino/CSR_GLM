import PyPDF2
import os
from tqdm.auto import tqdm
from utils import (
    generate_subtopic,
    generate_question,
    generate_answer,
    prepare_args,
    BudgetTracker
)

args = prepare_args()

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
    return prompts

def main():
    # Create a budget tracker to keep track of the token budget
    budget_tracker = BudgetTracker(total_budget=args.tokenBudget)

    # Initialize a progress bar with the total number of datasets to be generated
    # args.api_key = "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"
    args.api_key = "sk-IwHMBXeWyuLRp8kFEWZiNS6IlJJliO8MWXNMyhfXq35hFPEQ"
    # args.api_key = "ZW5waGFXNXVNVEV4UURFMk15NWpiMjA9"

    # str_name = 'extra_tools/auto-dataset-generate-by-chatgpt/src/市场分析.txt'
    str_name = '市场分析.txt'
    file_extension = os.path.splitext(str_name)[1].lower()
    if file_extension == ".pdf":
        print("Read PDF")
        prompts = pdf_reader(str_name)
    elif file_extension == ".md":
        print("Read MD")
        prompts = mk_reader(str_name)
    elif file_extension == ".txt":
        print("Read TXT")
        prompts = mk_reader(str_name)
    # Generate a subtopic based on the provided topic and generalization parameters
    
    for prompt in prompts:
        # pbar = tqdm(total=args.number_of_dataset, desc="Processing")
        # while pbar.n < pbar.total:
        args.topic = prompt
        sub_topic = generate_subtopic(args=args,
                                        budget_tracker=budget_tracker)

        # Generate questions based on the topic, subtopic, and API key
        questions = generate_question(sub_topic=sub_topic,
                                        args=args,
                                        budget_tracker=budget_tracker)

        # Generate answers for the questions using the API key and update the budget tracker and progress bar
        # generate_answer(questions=questions,
        #                 budget_tracker=budget_tracker,
        #                 pbar=pbar,
        #                 args=args)
        generate_answer(questions=questions,
                        budget_tracker=budget_tracker,
                        args=args)


if __name__ == '__main__':
    main()

