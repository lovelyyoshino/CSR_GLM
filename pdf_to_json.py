import os
import sys
import json
import pdfplumber
import spacy
import re
import requests
from tqdm import tqdm

nlp = spacy.load("zh_core_web_sm")

def fullwidth_to_halfwidth(s):
    return ''.join([chr(ord(c) - 0xFEE0) if 0xFF01 <= ord(c) <= 0xFF5E else c for c in s])

def table_to_corpus(table):
    processed_table = [[re.sub(r'\s', '', str(cell)) if cell is not None else '' for cell in row] for row in table]
    corpus = '\n'.join(['\t'.join(row) for row in processed_table])
    corpus = fullwidth_to_halfwidth(corpus)
    corpus = corpus.replace('', '.')
    return corpus

def extract_text_from_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        text_data = []
        table_data = []
        for page in tqdm(pdf.pages, desc="Processing pages"):
            text = page.extract_text().replace('\n', ' ')
            text = fullwidth_to_halfwidth(text.replace(' ', ''))
            text = text.replace('', '.')
            doc = nlp(text)
            for sent in doc.sents:
                clean_text = ' '.join(sent.text.split())
                text_data.append(clean_text)

            for table in page.extract_tables():
                if table:
                    table_data.append(table_to_corpus(table))
                else:
                    table_data.append("n.a.")

    return text_data, table_data

def create_corpus_dicts(text_data, table_data):
    corpus_dicts = []
    for text in text_data:
        corpus_dict = {
            'input': text,
            'is_table': False
        }
        corpus_dicts.append(corpus_dict)

    for table in table_data:
        corpus_dict = {
            'input': table,
            'is_table': True
        }
        corpus_dicts.append(corpus_dict)

    return corpus_dicts

def is_question(sentence):
    question_keywords = ["吗", "是否", "是不是", "能否", "可以", "怎么样", "如何", "为什么", "为何", "哪里", "哪个", "哪种", "多少", "多久", "什么", "何时", "何地", "怎样", "如何", "能否", "可否", "是否可以", "是否能够", "是否有", "是否存在", "是谁", "是什么", "有没有", "是否需要", "需要不需要", "是否可行", "可行不可行", "能不能", "能不能够"]
    for keyword in question_keywords:
        if keyword in sentence:
            return True
    return False

def generate_questions(answer, num_questions=5):
    questions = []
    for _ in range(num_questions):
        prompt = "请根据答案生成一个问题："
        data = {"prompt": prompt + answer, "history": []}

        url = "http://127.0.0.1:8000"
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, headers=headers, json=data)
        result = response.json()

        questions.append(result["response"])

    return questions

def main():
    directory_path = sys.argv[1]
    all_files = [file for file in os.listdir(directory_path) if file.endswith('.pdf')]
    total_files = len(all_files)

    for i, file_name in enumerate(all_files):
        file_path = os.path.join(directory_path, file_name)
        try:
            text_data, table_data = extract_text_from_pdf(file_path)
            corpus_dicts = create_corpus_dicts(text_data, table_data)

            dir_path = os.path.dirname(file_path)
            base_file_name = os.path.splitext(os.path.basename(file_path))[0]
            base_file_name_processed = base_file_name.replace('+', '/')
            parsed_json_file_path = os.path.join(dir_path, "parsed_" + base_file_name + '.json')
            qa_json_file_path = os.path.join(dir_path, "QA_" + base_file_name + '.json')

            parsed_corpus = []
            total = len(corpus_dicts)
            for j, corpus_dict in enumerate(tqdm(corpus_dicts, desc="Processing parsed data", total=total)):
                input_text = corpus_dict['input']
                if is_question(input_text) or len(input_text) < 12:
                    continue

                input_text = input_text.replace('\n', ' ')
                input_text = input_text.replace('\r', ' ')

                if not corpus_dict['is_table']:
                    input_text = input_text.replace('\t', ' ')

                input_text = input_text.strip()
                input_text = input_text.replace('\t ', '\n')
                input_text = input_text.replace(' ', '\n')
                parsed_corpus.append({
                    'input': input_text
                })

            with open(parsed_json_file_path, 'w', encoding='utf-8') as parsed_json_file:
                json.dump(parsed_corpus, parsed_json_file, ensure_ascii=False, indent=4)

            qa_corpus = []
            total = len(parsed_corpus)
            for k, corpus_dict in enumerate(tqdm(parsed_corpus, desc="Processing QA data", total=total)):
                input_text = corpus_dict['input']
                instructions = generate_questions(input_text)
                for instruction in instructions:
                    output = input_text
                    qa_corpus.append({
                        'instruction': instruction,
                        'input': base_file_name_processed,
                        'output': output
                    })

            with open(qa_json_file_path, 'w', encoding='utf-8') as qa_json_file:
                json.dump(qa_corpus, qa_json_file, ensure_ascii=False, indent=4)

            print(f"Completed {i+1}/{total_files} files. {total_files-i-1} remaining.")
        except Exception as e:
            print(f"Error processing file {file_name}. Error message: {e}. Skipping file.")

if __name__ == "__main__":
    main()