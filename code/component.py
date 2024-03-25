"""123
"""
import json
import subprocess
import configparser
import numpy as np
from FlagEmbedding import BGEM3FlagModel
from module.about_model import get_full_prompt, call_model
from module.split_doc import extract_keyword
# import about_model

def hyde(raw_str, hyde_type, model_type, model_and_tokenizer=None):
    """123
    """
    print("enter hyde")

    config = configparser.ConfigParser()
    config.read('/user_data/DG/hyde_113_0321/global_variable/config.ini')
    datapath = config['prompt']['hyde_prompt']

    print("\tafter user_prompt")
    with open(datapath, 'r', encoding='utf-8') as file:
        temp = json.load(file)
        user_prompt = temp[hyde_type]['test']
    print("\tafter user_prompt")

    # full_prompt = get_full_prompt(model_type, user_prompt, raw_str)
    
    keyword = extract_keyword(raw_str)
    print(f"keyword = {keyword}")
    full_prompt = get_full_prompt(model_type, user_prompt, keyword)
    
    hyde_str = call_model(full_prompt, model_type, model_and_tokenizer)

    print("exit hyde")
    return hyde_str



def corpus_build(retriever_type, path_2_ttl_data, path_2_corpus):
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

        # prepocess ttl_docs
        with open(path_2_ttl_data, 'r', encoding='utf-8') as file:
            total_data = json.load(file)
        ttl_doc = [data['contents'] for data in total_data]

        # create corpus
        encoder = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        corpus = encoder.encode(
            ttl_doc,
            batch_size=12,
            max_length=8192, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
            )['dense_vecs']
        corpus = corpus.astype(np.float32)

        # save corpus
        np.save(path_2_corpus, corpus)

        print("\texit elif 'BGEM3'")

    print("exit corpus_build")
    return 0    # 表示成功
