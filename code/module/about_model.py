"""Build a searcher from a pre-built index; download the index if necessary.

Parameters
----------
prebuilt_index_name : str
    Prebuilt index name.
verbose : bool
    Print status information.

Returns
-------
LuceneSearcher
    Searcher built from the prebuilt index.
"""

import os
import json

import configparser
import openai
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
import requests


def get_full_prompt(model_type, user_prompt, raw_str):
    r"""

    Parameters
    ----------
    model : str or pointer
        'api \model' or pointer to a loaded model

    Returns
    -------
    String
    """
    print("enter get_full_prompt")

    print(f"raw_str = {raw_str}")
    if 'breeze' in model_type:
        print("enter if breeze")
        config = configparser.ConfigParser()
        config.read('/user_data/DG/hyde_113_0321/global_variable/config.ini')
        system_prmpt = config['prompt']['breeze_system_prompt']
        
        # full_prompt = \
        #     f"<s> {system_prmpt} [INST] {user_prompt} markdown格式的表格：{raw_str}，總結： [/INST] "
        
        # full_prompt = \
        #     f"<s> {system_prmpt} [INST] 你現在是一位農業病蟲害防治專家，請舉出有關於{raw_str}的危害： [/INST] "
        
        full_prompt = \
            f"<s> {system_prmpt} [INST] 你現在是一位農業病蟲害防治專家，請具體說明若作物遭受{raw_str}危害，會有哪些情況發生： [/INST] "

        print("exit if breeze")

    elif 'taide' in model_type:
        print("enter elif taide")

        full_prompt = \
            f"<s> {user_prompt} [INST] markdown格式的表格：{raw_str}，總結： [/INST] "

        print("exit elif taide")

    elif 'gpt' in model_type:
        print("enter elif gpt")

        full_prompt = \
            f" {user_prompt} markdown格式的表格：{raw_str}，總結："

        print("exit elif gpt")

    print("exit get_full_prompt")
    return full_prompt


def load_model(model_type):
    """123
    """
    print("enter load_model")

    if model_type=='breeze':
        print("enter if breeze")

        torch.cuda.empty_cache()
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("\t\tGPU get")

        # 信彰的
        path_2_model = os.getenv("PATH_2_breeze")
        print("\t\tPATH_2_MODEL get")
        model = AutoModelForCausalLM.from_pretrained(
            path_2_model,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2", # optional for inference (有些顯卡不能用)
            # 會報錯要去 huggingface 下載
            # pip install flash-attn --no-build-isolation
            trust_remote_code=True, # MUST
        )
        print("\t\tmodel get")

        model.config.use_cache = False
        model.config.pretraining_tp = 1

        tokenizer = AutoTokenizer.from_pretrained(path_2_model, trust_remote_code=True)
        tokenizer.padding_side = "right"
        print("\t\ttokenizer get")

        model_and_tokenizer = [model, tokenizer]

        print("exit if breeze")

    elif model_type=='taide':
        print("enter elif taide")



        print("exit elif taide")

    else:
        model_and_tokenizer = None

    print("exit load_model")
    return model_and_tokenizer


def call_model(full_prompt, model_type, model_and_tokenizer=None):
    r"""

    Parameters
    ----------
    model : str or pointer
        'api \model' or pointer to a loaded model

    Returns
    -------
    String
    """
    print("enter call_model")

    if 'api' in model_type:
        print("\tenter if api")

        if 'breeze' in model_type:
            print("\t\tenter if breeze")

            host = os.getenv("breeze_url")
            headers = {
                'Content-Type': 'application/json',
                'accept': 'application/json'
            }
            data = json.dumps(
                {
                    "inputs": full_prompt,
                    "parameters": {
                        "do_sample": True,
                        "temperature": 0.01,
                        "top_p": 0.95,
                        # 'max_new_tokens':200,
                    }
                }
            )
            r = requests.request("POST", host, headers=headers, data=data)
            r = json.loads(r.text)
            if "generated_text" in r.keys():
                # str
                response = r['generated_text']

            print("\t\texit if breeze")

        elif 'taide' in model_type:
            print("\t\tenter elif taide")
            token = os.getenv("TAIDE_api_key")
            host = os.getenv("taide_url")
            headers = {
                "Authorization": "Bearer " + token
            }
            data = {
                "model": "TAIDE/b.11.0.0",
                "prompt": full_prompt, # assigned in the funciton api_TAIDE()
                "temperature": 0,
                "top_p": 0.9,
                "presence_penalty": 1,
                "frequency_penalty": 1,
                "max_tokens": 200,  # 他是 max_new_tokens
                "repetition_penalty":1.2
            }

            r = requests.post(host+"/completions", json=data, headers=headers)

            print(f"\tr = {r}")
            r = r.json()
            print(f"\tr = {r}")
            if "choices" in r:
                response = r["choices"][0]["text"]
                response = str(response)

            print("\t\texit elif taide")

        elif 'gpt' in model_type:
            print("\t\tenter elif gpt")

            openai.api_key = os.getenv("openai_api_key")
            messages = [{"role": "user", "content": full_prompt}]
            r = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-1106",
                messages=messages,
                temperature=0,
                max_tokens=4000,
            )

            # list
            # r = [r.choices[0].message.content]
            # response = r[0]
            response = r.choices[0].message.content
            print("\t\texit elif gpt")

        print("exit if api")

    else:
        print("\tenter else not api")

        if model_type=='breeze':
            print("\t\tenter if breeze")
            model, tokenizer = model_and_tokenizer

            print("\t\tbefore tokenize")
            sentence_input = tokenizer(
                full_prompt,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(model.device)
            print("\t\tafter tokenize")

            print("\t\tbefore GenerationConfig")
            generation_config = GenerationConfig(
                # max_new_tokens=10,
                # do_sample=True,
                # temperature=1,
                num_beams = 6,
                num_return_sequences=1, # < num_beams

                # no_repeat_ngram_size=2,
                early_stopping=False,
                max_length=4096, #CUDA oom 輸出有關
                # top_p=0.92,   # 前 p 可能，共多少個不知道
                # top_k=15,     # 前 k 個，共佔多少機率不知道
            )
            print("\t\tafter GenerationConfig")

            print("\t\tbefore generate")
            response = model.generate(
                **sentence_input,
                generation_config=generation_config,
            )
            print("\t\tafter generate")

            print("\t\tbefore tokenize decode")
            # list
            response = tokenizer.batch_decode(
                response[:, sentence_input.input_ids.size(1) :], skip_special_tokens=True
            )
            response = response[0]
            print("\t\tafter tokenize decode")

            print("\t\texit if breeze")

        elif model_type=='taide':
            print("\t\tenter elif taide")
            model, tokenizer = model_and_tokenizer


            print("\t\texit elif taide")

        print("\texit else")

    print("exit call_model")

    return response


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv("/user_data/DG/hyde_113_0321/global_variable/.env")

    # model_and_tokenizer = load_model("breeze")
    # res = call_model("hi 你好", "breeze", model_and_tokenizer)

    res = call_model("hi 你好", 'taideapi')
    print(f"res = {res}")
    print(f"type(res) = {type(res)}")
