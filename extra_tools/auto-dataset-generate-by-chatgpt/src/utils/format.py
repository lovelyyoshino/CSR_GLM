import re
from argparse import ArgumentParser
from dataclasses import dataclass, field
from typing import List


def extract_list(answers, tag='example'):
    if "ErrorCode:400" in answers:
        raise ValueError("The topic content does not comply with platform guidelines. Please try again.")
    pattern = rf"<{tag}>(.*?)</{tag}>"
    texts = re.findall(pattern, answers, re.DOTALL)
    return texts


@dataclass
class DataSetArguments:
    generalization_index: float = field(
        default=0.75,
        metadata={
            "help": "Generalization Index, the higher it is, the more diverse the dataset is. Default value is 0.0"}
    )

    generalization_basic: int = field(
        default=10,
        metadata={"help": "The scaling factor, represented by k in the equation y = kx + b. Default value is 10"}
    )

    number_of_dataset: int = field(
        default=5,
        metadata={"help": "The number of generated datasets. Default is 5"}
    )

    topic: str = field(
        default=None,
        metadata={"help": "Dataset topic. Default is None"}
    )

    dataset_output_path: str = field(
        default='data',
        metadata={"help": "Path to save the dataset"}
    )

    proxy: str = field(
        default=None,
        metadata={
            "help": "If you are unable to access OpenAI, please provide a proxy address. e.g., http://127.0.0.1:7890"}
    )

    api_key: List[str] = field(
        default=None,
        metadata={"help": "The OpenAI API-KEY"}
    )

    api_base: str = field(
        default=None,
        metadata={"help": "The OpenAI API-BASE"}
    )

    tokenBudget: float = field(
        default=100,
        metadata={"help": "Token budget. Default is 100"}
    )

    retryTime: int = field(
        default=5,
        metadata={"help": "The number of retry times to the chat_gpt API Default is 5"}
    )

    tokenLimit: int = field(
        default = 100,
        metadata={"help": "Not less than 50, default is 100"}
    )

    api_key_list: List[str] = field(
        default=None,
        metadata={"help": "api_key_list."}
    )

    api_index: int = field(
        default=0,
        metadata={"help": "The index of api key in the api_key_list. Default is 0"}
    )


def prepare_args() -> DataSetArguments:
    parser = ArgumentParser()
    parser.add_argument('--generalization_index', type=float, default=0.75)
    parser.add_argument('--generalization_basic', type=int, default=10)
    parser.add_argument('--number_of_dataset', type=int, default=5)
    parser.add_argument('--topic', type=str, default=None)
    parser.add_argument('--dataset_output_path', type=str, default='data')
    parser.add_argument('--proxy', type=str, default=None)
    parser.add_argument('--api_key', nargs='+', default=None)
    parser.add_argument('--tokenBudget', type=float, default=1)
    parser.add_argument('--retry_time', type=int, default=5)
    args = parser.parse_args()
    model_args = DataSetArguments(
        generalization_index=args.generalization_index,
        generalization_basic=args.generalization_basic,
        number_of_dataset=args.number_of_dataset,
        topic=args.topic,
        dataset_output_path=args.dataset_output_path,
        proxy=args.proxy,
        api_key=args.api_key,
        tokenBudget=args.tokenBudget,
        retry_time=args.retry_time
    )
    return model_args
