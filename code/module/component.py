"""123
"""
import json
import configparser
from about_model import get_full_prompt, call_model

def hyde(raw_str, hyde_type, model_type, model_and_tokenizer=None):
    """123
    """
    print("enter hyde")

    config = configparser.ConfigParser()
    config.read('/user_data/DG/hyde_113_0321/global_variable/config.ini')
    datapath = config['path']['hyde_prompt']

    print("\tafter user_prompt")
    with open(datapath, 'r', encoding='utf-8') as file:
        temp = json.load(file)
        user_prompt = temp[hyde_type][0]
    print("\tafter user_prompt")

    full_prompt = get_full_prompt(model_type, user_prompt, raw_str)
    hyde_str = call_model(full_prompt, model_type, model_and_tokenizer)

    print("exit hyde")
    return hyde_str
