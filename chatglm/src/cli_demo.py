# coding=utf-8
# Implements stream chat in command line for ChatGLM fine-tuned with PEFT.
# Usage: python cli_demo.py --checkpoint_dir path_to_checkpoint [--quantization_bit 4]

from glmtuner import ChatModel
from glmtuner.tuner import get_infer_args


def main():
    chat_model = ChatModel(*get_infer_args())
    history = []
    print("欢迎使用大模型，输入内容即可对话，clear 清空对话历史，stop 终止程序")

    while True:
        try:
            query = input("\nInput: ")
        except UnicodeDecodeError:
            print("Detected decoding error at the inputs, please set the terminal encoding to utf-8.")
            continue
        except Exception:
            raise

        if query.strip() == "stop":
            break

        if query.strip() == "clear":
            history = []
            print("History has been removed.")
            continue

        print("模型输入: ", end="", flush=True)
        gen_kwargs = {
            "top_p": 0.75,
            "top_k": 0.1,
            "temperature": 0.95,
            "num_beams": 1,
            "max_length":4096,
            "max_new_tokens": 1024,
            "repetition_penalty": 1.0,
        }
        response = ""
        for new_text in chat_model.stream_chat(query, history,input_kwargs=gen_kwargs):
            print(new_text, end="", flush=True)
            response += new_text
        print()

       #  history = history + [(query, response)]


if __name__ == "__main__":
    main()
