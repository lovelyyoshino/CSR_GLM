import json
import math
import concurrent.futures
import random
import sys
import requests
from datetime import timedelta

import openai
import time
from typing import List
from .format import extract_list, DataSetArguments
from .save_dataset import write_dict_list_to_file

class BudgetTracker:
    def __init__(self, total_budget=None):
        # Initialize the budget tracker with the total budget and total tokens used
        self.total_budget = total_budget
        self.total_tokens_used = 0
        self.current_cost = 0

    def is_budget_exceeded(self):
        # Check if the budget has been exceeded based on the total tokens used
        self.current_cost = ((self.total_tokens_used / 1000) * 0.002)
        return self.total_budget is not None and self.current_cost >= self.total_budget


class ChatAPI:
    def __init__(self, api_key=None,
                 model='gpt-3.5-turbo',
                 api_base=None,
                 system_settings='You are a capable assistant, making every effort to provide assistance to users.',
                 temperature=0.7,
                 proxy=None):
        # Initialize the ChatAPI with the API key, model, system settings, temperature, and proxy
        if api_key is None:
            api_key = []
        self.model = model
        self.system_settings = system_settings
        self.temperature = temperature
        self.max_retries = 3
        self.retry_delay = 20
        self.proxy = proxy

        if len(api_key) > 0:
            openai.api_key = api_key
            
        else:
            raise ValueError("api_key is empty or incorrect")

        if self.proxy:
            openai.proxy = self.proxy
        
        if api_base is not None:
           openai.api_base = api_base 
        

    def chat(self, prompt, budget_tracker):
        # Perform a chat conversation with OpenAI API
        retries = 0
        while retries < self.max_retries:
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_settings},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=300,
                    deployment_id = None,
                    temperature=self.temperature
                )
                tokens_used = response.usage['total_tokens']
                budget_tracker.total_tokens_used += tokens_used
                # choices_answer = response["choices"]
                # completion_answer = choices_answer[0].message.content.strip()
                # return completion_answer
                return response.choices[0].message.content
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                retries += 1
                if retries < self.max_retries:
                    print(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    raise ValueError("Failed to get a valid response after maximum retries")

class FKAPI:
    def __init__(self, api_key_list,
                 api_index = 0,
                 api_key=None,
                 model='gpt-3.5-turbo',
                 api_base=None,
                 system_settings='You are a capable assistant, making every effort to provide assistance to users.',
                 temperature=0.7,
                 proxy=None):
        # Initialize the ChatAPI with the API key, model, system settings, temperature, and proxy
        if api_key is None:
            api_key = []
        self.model = model
        self.system_settings = system_settings
        self.temperature = temperature
        self.max_retries = len(api_key_list) + 2
        self.retry_delay = 10
        self.api_key_list = api_key_list
        self.api_index = api_index
        self.proxy = proxy 
        self.api_key = api_key

    def chat(self, prompt, budget_tracker):
        # Perform a chat conversation with OpenAI API
        retries = 0
        while retries < self.max_retries:
            try:
                headers = {"Content-Type": "application/json",
                           "Authorization": f"Bearer {self.api_key}"
                            }

                data = {
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": self.system_settings},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": self.temperature
                        }
                response = requests.post("https://ai.fakeopen.com/v1/chat/completions", 
                                        headers=headers, 
                                        data=json.dumps(data)
                                        )
                print(response.json())
                # return response.choices.message.content
                data_temple = response.json()
                content = data_temple['choices'][0]['message']['content']
                return content

            except Exception as e:
                print(f"An error occurred: {str(e)}")
                if str(e) == "'choices'" and 'error' in response.json():
                    error = response.json()['error']['message']
                    error_str = 'Please try again later'
                    # error_str = 'Access token is missing'
                    if error.find(error_str) != -1:
                        if self.api_index < len(self.api_key_list) - 1:
                            self.api_index += 1
                        else:
                            self.api_index = 0
                        self.api_key = self.api_key_list[self.api_index]
                        print(f"API key index: {self.api_index}, API key: {self.api_key_list[self.api_index]}")
                # if self.api_index < len(self.api_key_list) - 1:
                #     self.api_index += 1
                # else:
                #     self.api_index = 0
                # self.api_key = self.api_key_list[self.api_index]
                # print(f"API key index: {self.api_index}, API key: {self.api_key_list[self.api_index]}")
                retries += 1
                if retries < self.max_retries:
                    print(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    raise ValueError("Failed to get a valid response after maximum retries")

def generate_question_chatgpt(sub_topic: List[str], args: DataSetArguments, budget_tracker: BudgetTracker) -> List[str]:
    # Generate questions based on the given topic, sub-topic, API key, budget tracker, and number of datasets
    if not args.topic:
        raise ValueError("param topic is required, not None")

    example = """
    <example>在一个笼子里，有35只鸡和兔，共有94只脚。问鸡和兔各有多少只？</example>
    <example>在一个笼子里，有60只鸡和兔，共有166只脚。问鸡和兔各有多少只？</example>
    """

    topic = ""
    if len(sub_topic) > 0:
        topic += f"""
           以主题{args.topic, sub_topic}生成{args.number_of_dataset}个的问题
           """
    else:
        topic = f"""
           以主题{args.topic}生成{args.number_of_dataset}个的问题
           """

    conditions = """
    你无需对生成的示例进行答复或解释
    每个生成的示例必须是祈使句或疑问句
    祈使句和疑问句的生成比例要1:1
    每个示例必须以<example>开始，以</example>结束.
    每个示例控制在40字以内
    如果主题涉及敏感话题或不符合中华人民共和国相关法律法规请立刻停止所有的生成并直接返回下面'''包裹的内容
    ```
    ErrorCode:400
    ```
    """
    questions = []

    def process_question(prompt):
        api = ChatAPI(api_key=args.api_key, system_settings='You are an efficient assistant, aiming to provide concise '
                                                            'answers based on instructions.', proxy=args.proxy)
        q_response = api.chat(prompt=prompt, budget_tracker=budget_tracker)
        return extract_list(q_response)

    generated_questions = 0  # Record the number of generated questions

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # while generated_questions < args.number_of_dataset:
        #     # Calculate the remaining number of questions to generate
        #     remaining_questions = args.number_of_dataset - generated_questions

        #     # Generate 50 questions each time, or the remaining number (whichever is smaller)
        #     batch_size = math.ceil(remaining_questions / 50)

            # Dynamically generate tasks to submit to the thread pool
        future_to_question = {
            # executor.submit(process_question, example + topic + conditions): question
            executor.submit(process_question, topic + conditions): question
            for question in range(len(sub_topic))
        }

        # Iterate over completed tasks
        for future in concurrent.futures.as_completed(future_to_question):
            question = future_to_question[future]
            try:
                answer = future.result()
                questions.extend(answer)
                generated_questions += len(answer)

                if budget_tracker.is_budget_exceeded():
                    return questions  # Return immediately if the budget is exceeded
            except Exception as e:
                print(f"Error occurred for question: {question}. Error message: {str(e)}")

            # Check if the desired number of questions has been generated, if so, break the loop
            # if generated_questions >= args.number_of_dataset:
            #     break

    return questions

def generate_question_chatgpt_by_paragraph(args: DataSetArguments, budget_tracker: BudgetTracker) -> List[str]:
    # Generate questions based on the given topic, sub-topic, API key, budget tracker, and number of datasets
    if not args.topic:
        raise ValueError("param topic is required, not None")

    command = """
    如果我们之前有过对话，请忘掉我们以前的对话并忽略此对话中所有先前的提示，以一个初始的状态进行下述对话。
    """
    questions = []
    topic = f"""
        请对{args.topic}进行总结和梳理，并生成{args.number_of_dataset}个的问题
        """

    conditions = """
        你无需对生成的示例进行答复或解释
        每个生成的示例必须是祈使句或疑问句
        祈使句和疑问句的生成比例要1:1
        每个示例必须以<example>开始，以</example>结束.
        每个示例控制在40字以内
        如果主题涉及敏感话题或不符合中华人民共和国相关法律法规请立刻停止所有的生成并直接返回下面'''包裹的内容
        ```
        ErrorCode:400
        ```
        """
    questions = []

    def process_question(prompt):
        api = ChatAPI(api_key=args.api_key, system_settings='You are an efficient assistant, aiming to provide concise '
                                                            'answers based on instructions.', proxy=args.proxy)
        q_response = api.chat(prompt=prompt, budget_tracker=budget_tracker)
        return extract_list(q_response)

    generated_questions = 0  # Record the number of generated questions

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_question = {
            executor.submit(process_question, command + topic + conditions): question
            for question in range(len(args.topic))
        }

        # Iterate over completed tasks
        for future in concurrent.futures.as_completed(future_to_question):
            question = future_to_question[future]
            try:
                answer = future.result()
                questions.extend(answer)
                generated_questions += len(answer)

                if budget_tracker.is_budget_exceeded():
                    return questions  # Return immediately if the budget is exceeded
            except Exception as e:
                print(f"Error occurred for question: {question}. Error message: {str(e)}")

    return questions

def generate_question_pandora(sub_topic: List[str], args: DataSetArguments, budget_tracker: BudgetTracker) -> List[str]:
    # Generate questions based on the given topic, sub-topic, API key, budget tracker, and number of datasets
    if not args.topic:
        raise ValueError("param topic is required, not None")
    def process_question(prompt):
        api = FKAPI(args.api_key_list, args.api_index, api_key=args.api_key_list[args.api_index], system_settings='You are an efficient assistant, aiming to provide concise '
                                                                'answers based on instructions.', proxy=args.proxy)
        q_response = api.chat(prompt=prompt, budget_tracker=budget_tracker)
        return api.api_index, extract_list(q_response)

    example = """
    <example>在一个笼子里，有35只鸡和兔，共有94只脚。问鸡和兔各有多少只？</example>
    <example>在一个笼子里，有60只鸡和兔，共有166只脚。问鸡和兔各有多少只？</example>
    """
    questions = []
    for subtopic in sub_topic:
        topic = ""
        topic += f"""
        以主题{args.topic, subtopic}生成{args.number_of_dataset}个的问题
        """

        conditions = """
        你无需对生成的示例进行答复或解释
        每个生成的示例必须是祈使句或疑问句
        祈使句和疑问句的生成比例要1:1
        每个示例必须以<example>开始，以</example>结束.
        每个示例控制在40字以内
        如果主题涉及敏感话题或不符合中华人民共和国相关法律法规请立刻停止所有的生成并直接返回下面'''包裹的内容
        ```
        ErrorCode:400
        ```
        """
        index, question = process_question(topic + conditions)
        answer = question
        questions.extend(answer)
    return index, questions

def generate_question_pandora_by_paragraph(args: DataSetArguments, budget_tracker: BudgetTracker) -> List[str]:
    # Generate questions based on the given topic, sub-topic, API key, budget tracker, and number of datasets
    if not args.topic:
        raise ValueError("param topic is required, not None")
    def process_question(prompt):
        api = FKAPI(args.api_key_list, args.api_index, api_key=args.api_key_list[args.api_index], system_settings='You are an efficient assistant, aiming to provide concise '
                                                                'answers based on instructions.', proxy=args.proxy)
        q_response = api.chat(prompt=prompt, budget_tracker=budget_tracker)
        return api.api_index, extract_list(q_response)

    command = """
    如果我们之前有过对话，请忘掉我们以前的对话并忽略此对话中所有先前的提示，以一个初始的状态进行下述对话。
    """
    questions = []
    topic = f"""
        请对{args.topic}进行总结和梳理，并生成{args.number_of_dataset}个的问题
        """

    conditions = """
        你无需对生成的示例进行答复或解释
        每个生成的示例必须是祈使句或疑问句
        祈使句和疑问句的生成比例要1:1
        每个示例必须以<example>开始，以</example>结束.
        每个示例控制在40字以内
        如果主题涉及敏感话题或不符合中华人民共和国相关法律法规请立刻停止所有的生成并直接返回下面'''包裹的内容
        ```
        ErrorCode:400
        ```
        """
    index, question = process_question(command + topic + conditions)
    answer = question
    questions.extend(answer)
    return index, questions

def generate_subtopic(args: DataSetArguments,
                      budget_tracker: BudgetTracker) -> List[str]:
    # Generate sub-topics based on the given topic, generalization index, generalization basic, API key, and budget
    # tracker
    if args.generalization_basic * args.generalization_index == 0:
        return []

    prompt = f"""
         根据<Topic>标签内容生成{int(args.generalization_index * args.generalization_basic)}个子主题:
        每个子主题必须控制在6个字以内。
        每个子主题必须以<SubTopic>开始，</SubTopic>结束。
        使用方法:
        <Topic>{args.topic}</Topic>
        <SubTopic>子主题1</SubTopic>
        <SubTopic>子主题2</SubTopic>
        ...以此类推...
        """
    if args.api_key.startswith('fk'):
        api = FKAPI(args.api_key_list, args.api_index, api_key=args.api_key_list[args.api_index], proxy=args.proxy)
        data= extract_list(api.chat(prompt=prompt, budget_tracker=budget_tracker), 'SubTopic')
        return api.api_index, data
    else:
        api = ChatAPI(api_key=args.api_key, proxy=args.proxy)
        return 0, extract_list(api.chat(prompt=prompt, budget_tracker=budget_tracker), 'SubTopic')

    


# def generate_answer(questions: List[str], budget_tracker: BudgetTracker, pbar, args: DataSetArguments):
def generate_answer_chatgpt(questions: List[str], budget_tracker: BudgetTracker, args: DataSetArguments):
    # Generate answers for the given list of questions using the API key, budget tracker, progress bar, and output path
    api = ChatAPI(api_key=args.api_key, system_settings='你是一个知识丰富的助手，展示出你的才能！',
                  proxy=args.proxy)
    

    def process_question(question):
        prompt = f"""
        回答以下'''包裹的问题。展示你的知识，但要像学者一样严谨。
        你可以选择不回答不确定的内容，并从你熟悉的角度回答。```
        ```
        {question}
        ```
        """
        response = api.chat(prompt=prompt, budget_tracker=budget_tracker)
        answer_dict = {
            "question": question,
            "answer": response
        }
        return answer_dict

    generated_answers = 0
    current_questions = questions
    current_retry_time = -1
    thread_num_max = 10

    while generated_answers < len(questions) and current_retry_time < args.retryTime:
        if len(current_questions) > thread_num_max: 
            questions_batch = current_questions[:thread_num_max]
            current_questions = current_questions[thread_num_max:]
        else:
            questions_batch = current_questions
            current_questions = []
        current_retry_time += 1
        answers = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_question = {executor.submit(process_question, question): question for question in
                                  questions_batch}
            for future in concurrent.futures.as_completed(future_to_question):
                question = future_to_question[future]
                try:
                    answer = future.result()
                    answers.append(answer)
                    generated_answers += 1
                    if budget_tracker.is_budget_exceeded():
                        write_dict_list_to_file(data_list=answers, output_path=args.dataset_output_path)
                        sys.exit(0)
                except Exception as e:
                    print(f"Error occurred for question: {question}. Error message: {str(e)}")
                    current_questions.append(question)
        write_dict_list_to_file(data_list=answers, output_path=args.dataset_output_path)
        print(str(len(answers)) + " answers have been written to " + str(args.dataset_output_path))

def generate_answer_pandora(questions: List[str], budget_tracker: BudgetTracker, args: DataSetArguments):
        # Generate answers for the given list of questions using the API key, budget tracker, progress bar, and output path
    api = FKAPI(args.api_key_list, args.api_index, api_key=args.api_key_list[args.api_index], system_settings='你是一个知识丰富的助手，展示出你的才能！',
                  proxy=args.proxy)
    def process_question(question):
        prompt = f"""
        回答以下{question}问题。展示你的知识，但要像学者一样严谨。
        你可以选择不回答不确定的内容，并从你熟悉的角度回答。
        """
        response = api.chat(prompt=prompt, budget_tracker=budget_tracker)
        answer_dict = {
            "question": question,
            "answer": response
        }
        return answer_dict
    answers = []
    generated_answers = 0
    for question in questions:
        answer = process_question(question)
        answers.append(answer)
        generated_answers += 1
    write_dict_list_to_file(data_list=answers, output_path=args.dataset_output_path)   
    return api.api_index 
       

    # while generated_answers < len(questions) and current_retry_time < args.retryTime:
    #     questions_batch = current_questions
    #     current_questions = []
    #     current_retry_time += 1
    #     answers = []
    #     with concurrent.futures.ThreadPoolExecutor() as executor:
    #         future_to_question = {executor.submit(process_question, question): question for question in
    #                               questions_batch}
    #         for future in concurrent.futures.as_completed(future_to_question):
    #             question = future_to_question[future]
    #             try:
    #                 answer = future.result()
    #                 answers.append(answer)
    #                 generated_answers += 1
    #                 if budget_tracker.is_budget_exceeded():
    #                     write_dict_list_to_file(data_list=answers, output_path=args.dataset_output_path)
    #                     sys.exit(0)
    #             except Exception as e:
    #                 print(f"Error occurred for question: {question}. Error message: {str(e)}")
    #                 current_questions.append(question)
    #     write_dict_list_to_file(data_list=answers, output_path=args.dataset_output_path)
    #     print(str(len(answers)) + " answers have been written to " + str(args.dataset_output_path))