"""123
"""

import os
import json
import subprocess
import configparser
import numpy as np
import torch
from FlagEmbedding import BGEM3FlagModel
from module.split_doc import extract_keyword
from module.about_retriever import init_retriever, apply_retriever
from module.about_model import get_full_prompt, call_model, load_model



def hyde_single(raw_str, hyde_type, model_and_tokenizer=None):
    """123
    """
    print("enter hyde_single")

    # load user_prompt
    config = configparser.ConfigParser()
    config.read('/user_data/DG/hyde_113_0321/global_variable/config.ini')
    datapath = config['prompt']['hyde_prompt']
    model_type = config['mode']['model_type']


    # # create hyde prompt

    # full_prompt = get_full_prompt(model_type, user_prompt, raw_str)

    keyword, keyword_type = extract_keyword(raw_str)
    print("\tbefore user_prompt")
    with open(datapath, 'r', encoding='utf-8') as file:
        temp = json.load(file)
        user_prompt = temp[hyde_type]['test']
    print("\tafter user_prompt")
    
    print(f"keyword = {keyword}\tkeyword_type = {keyword_type}")
    full_prompt = get_full_prompt(model_type, user_prompt, keyword, keyword_type)

    # hyde data
    hyde_str = call_model(full_prompt, model_type, model_and_tokenizer)

    print("exit hyde_single")
    return hyde_str


def hyde_total(now, hyde_type, path_2_raw_data, total_docid_2_hyde=None):
    """123
    """
    print("enter hyde_total")

    # load model_type
    config = configparser.ConfigParser()
    config.read('/user_data/DG/hyde_113_0321/global_variable/config.ini')
    model_and_tokenizer = load_model()

    # hyde data
    with open(path_2_raw_data, 'r', encoding='utf-8') as file:
        total_raw_data = json.load(file)
    for idx in total_docid_2_hyde:
        print(f"\tin for-loop idx={idx}")
        contents = total_raw_data[idx]['contents']
        hyde_contents = hyde_single(
            raw_str=contents,
            hyde_type=hyde_type,
            model_and_tokenizer=model_and_tokenizer
        )
        total_raw_data[idx]['contents'] += f"\n\n\n{hyde_contents}"

    # save hyde_data
    dir_2_hyde_dir = config['data']['hyde_dir']
    dir_2_hyde_data = f"{dir_2_hyde_dir}/{now}"
    os.makedirs(f"{dir_2_hyde_dir}/{now}", exist_ok=True)
    path_2_hyde_data = f"{dir_2_hyde_data}/{now}.json"
    with open(path_2_hyde_data, 'w', encoding='utf-8') as file:
        json.dump(total_raw_data, file, indent=4, ensure_ascii=False)

    print("exit hyde_total")
    return total_raw_data, path_2_hyde_data


def summarize_tatol(now, path_2_raw_data):
    """123
    """
    print("enter summarize_total")
    
    # load model_type
    config = configparser.ConfigParser()
    config.read('/user_data/DG/hyde_113_0321/global_variable/config.ini')
    model_and_tokenizer = load_model()
    
    with open(path_2_raw_data, 'r', encoding='utf-8') as file:
        total_raw_data = json.load(file)
    
    for idx, raw_data in enumerate(total_raw_data):
        contents = raw_data['contents']
        summary = hyde_single(contents, hyde_type='summary', model_and_tokenizer=model_and_tokenizer)
        raw_data['contents'] = summary
    
    # save hyde_data
    dir_2_summary_dir = config['data']['summary_dir']
    dir_2_summary_data = f"{dir_2_summary_dir}/{now}"
    os.makedirs(f"{dir_2_summary_dir}/{now}", exist_ok=True)
    path_2_summary_data = f"{dir_2_summary_data}/{now}.json"
    with open(path_2_summary_data, 'w', encoding='utf-8') as file:
        json.dump(total_raw_data, file, indent=4, ensure_ascii=False)
    
    # os.makedirs(output_folder, exist_ok=True)
    # output_path = os.path.join(output_folder, 'summaries.json')
    # with open(output_path, 'w', encoding='utf-8') as file:
    #     json.dump(total_raw_data, file, indent=4, ensure_ascii=False)
    
    print("exit summarize_total")
    return total_raw_data, path_2_summary_data


def corpus_build(path_2_ttl_data, path_2_corpus):
    """

    Parameters
    ----------
    retriever_type : str
        'BM25' or 'BGEM3'

    Returns
    -------
    None
    """
    print("enter corpus_build")

    config = configparser.ConfigParser()
    config.read('/user_data/DG/hyde_113_0321/global_variable/config.ini')
    retriever_type = config['mode']['retriever_type']

    # create corpus
    if retriever_type=='BM25':
        print("\tenter if 'BM25'")

        command = [
            'python', '-m', 'pyserini.index.lucene',
            '--collection', 'JsonCollection',
            '--input', path_2_ttl_data,
            '--language', 'zh',
            '--index', path_2_corpus,
            '--generator', 'DefaultLuceneDocumentGenerator',
            '--threads', '1',
            '--storePositions', '--storeDocvectors', '--storeRaw'
        ]

        # 執行命令
        subprocess.run(command, check=False)
        # The ``check`` keyword is set to False by default. It means the process launched by ``subprocess.run`` can exit with a non-zero exit code and fail silently. It's better to set it explicitly to make clear what the error-handling behavior is.

        print("\texit if 'BM25'")

    elif retriever_type=='BGEM3':
        print("\tenter elif 'BGEM3'")

        # activate gpu
        if torch.cuda.is_available():
            device = torch.device("cuda")

        # prepocess ttl_docs
        print("\tbefore loading ttl_doc")
        with open(path_2_ttl_data, 'r', encoding='utf-8') as file:
            total_data = json.load(file)
        ttl_doc = [data['contents'] for data in total_data]

        # create corpus
        print("\tbefore create encoder")
        encoder = BGEM3FlagModel(
            'BAAI/bge-m3',
            use_fp16=True,
            device=device     # https://github.com/FlagOpen/FlagEmbedding/issues/419
        )
        print("\tbefore create corpus")
        corpus = encoder.encode(
            ttl_doc,
            batch_size=12,
            max_length=8192, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
            )['dense_vecs']
        print("\tbefore transform corpus")
        corpus = corpus.astype(np.float32)

        # save corpus
        print("\tbefore save corpus")
        np.save(path_2_corpus, corpus)

        print("\texit elif 'BGEM3'")

    print("exit corpus_build")
    return 0    # 表示成功


def retrieve_related_docid(dir_2_corpus, ttl_query, top_k):
    """123
    """

    retriever = init_retriever(dir_2_corpus)
    ttl_hints_id = apply_retriever(
        retriever,
        ttl_query,
        top_k=top_k
    )
    return ttl_hints_id


def save_logging(now, dir_2_corpus, query_docid, ttl_hints_id):
    """123
    """
    config = configparser.ConfigParser()
    config.read('/user_data/DG/hyde_113_0321/global_variable/config.ini')
    retriever = init_retriever(dir_2_corpus)

    logging = {'dir_2_corpus' : dir_2_corpus}
    for query, hints_id in zip(query_docid , ttl_hints_id):
        gold_id = query_docid[query]

        gold_doc = retriever.doc(int(gold_id)).raw()
        gold_doc = json.loads(gold_doc)['contents']

        hit = [i for i, x in enumerate(hints_id) if x == gold_id]

        logging[query] = {
            "gold_id" : gold_id,
            "gold_doc" : [gold_doc],
            "rank" : None if hit==[] else hit[0],
            "hints_id" : hints_id
        }

    path_2_logging = os.path.join(config['logging']['logging'], f"{now}.json")
    with open(path_2_logging, 'w', encoding='utf-8') as file:
        json.dump(logging, file, ensure_ascii=False)

    return path_2_logging
