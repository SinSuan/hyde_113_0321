"""123
"""
import os
import json
import configparser
from dotenv import load_dotenv

from component import hyde, corpus_build
from module.about_model import load_model
from module.time_now import time_now
from module.about_retriever import init_retriever, apply_retriever

# TTL_QUERY = {
#     "伏滅鼠餌劑的施藥量" : "144559",
#     "伏滅鼠餌劑的施藥克數" : "144559",
#     "比芬諾乳劑的稀釋倍數" : "146830",
#     "聚乙醛餌劑每公頃每次施藥量是多少" : "147611",
#     "耐克螺可濕性粉劑施藥有限制什麼田的種類嗎" : "147611",
#     "有什麼殺蟲劑可以拿來處理積穀害蟲" : "145213"
# }

TTL_QUERY = {
    "伏滅鼠餌劑的施藥量" : "144559",
    "伏滅鼠餌劑的施藥克數" : "144559",
    "比芬諾乳劑的稀釋倍數" : "146830",
    "聚乙醛餌劑每公頃每次施藥量是多少" : "147611",
    "耐克螺可濕性粉劑施藥有限制什麼田的種類嗎" : "147611",
    "有什麼殺蟲劑可以拿來處理積穀害蟲" : "145213"
}

DOCID = TTL_QUERY.values()
DOCID = [int(idx) for idx in set(DOCID)]

# DOCID = [1, 2, 5, 10]
# DOCID = [144559, 146830, 147611, 145213]

MODEL_TYPE = 'breezeapi'

RETRIEVER_TYPE = 'BM25'
# RETRIEVER_TYPE = 'BGEM3'

def main():
    """123
    """

    model_and_tokenizer = load_model(MODEL_TYPE)

    config = configparser.ConfigParser()
    config.read('/user_data/DG/hyde_113_0321/global_variable/config.ini')


    # hyde corpus
    # path_2_raw_data = config['data']['raw_data']
    path_2_raw_data = "/user_data/DG/hyde_113_0321/data/raw_data/clean_agri_with_table.json"

    with open(path_2_raw_data, 'r', encoding='utf-8') as file:
        total_raw_data = json.load(file)

    for idx in DOCID:
        print(f"\tin for-loop idx={idx}")
        contents = total_raw_data[idx]['contents']
        hyde_contents = hyde(
            raw_str=contents,
            hyde_type='document',
            model_type=MODEL_TYPE,
            model_and_tokenizer=model_and_tokenizer
        )
        total_raw_data[idx]['contents'] += f"\n\n\n{hyde_contents}"

    path_2_hyde_dir = config['data']['hyde_data']
    now = time_now()
    dir_2_hyde_data = f"{path_2_hyde_dir}/{now}"
    os.makedirs(dir_2_hyde_data, exist_ok=True)
    path_2_hyde_data = f"{dir_2_hyde_data}/{now}.json"
    with open(path_2_hyde_data, 'w', encoding='utf-8') as file:
        json.dump(total_raw_data, file, indent=4, ensure_ascii=False)


    # create corpus
    path_2_corpus = config['corpus_dir'][RETRIEVER_TYPE]
    dir_2_corpus = f"{path_2_corpus}/{now}"
    os.makedirs(dir_2_corpus, exist_ok=True)
    corpus_build(RETRIEVER_TYPE, dir_2_hyde_data, dir_2_corpus)

    # retrieve
    retriever_info = init_retriever(RETRIEVER_TYPE, dir_2_corpus)
    ttl_hints_id = apply_retriever(
        retriever_info,
        list(TTL_QUERY.keys()),
        top_k=15
    )

    # sava log path
    logging = {'dir_2_corpus' : dir_2_corpus}
    _, retriever = retriever_info
    for query, hints_id in zip(TTL_QUERY, ttl_hints_id):
        gold_id = TTL_QUERY[query]
        
        gold_doc = retriever.doc(int(gold_id)).raw()
        gold_doc = json.loads(gold_doc)['contents']
        
        hit = [i for i, x in enumerate(hints_id) if x == gold_id]
        
        logging[query] = {
            "gold_id" : gold_id,
            "gold_doc" : [gold_doc],
            "rank" : None if hit==[] else hit[0],
            "hints_id" : hints_id
        }
        # print(query , " ", gold_id)
        # for i, x in enumerate(hints_id):
        #     print(x, " ", x == gold_id)

    # logging = { query : hints_id for query,hints_id in zip(TTL_QUERY, ttl_hints_id)}
    path_2_logging = f"/user_data/DG/hyde_113_0321/data/logging/{now}.json"
    with open(path_2_logging, 'w', encoding='utf-8') as file:
        json.dump(logging, file, ensure_ascii=False)

    print(f"ttl_hints_id = {ttl_hints_id}")


if __name__ == "__main__":
    load_dotenv("/user_data/DG/hyde_113_0321/global_variable/.env")
    main()
